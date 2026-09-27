"""Token-weighted causal language-model NLL/PPL over independent documents.

The caller supplies fixed token IDs from the same tokenizer for every candidate.
No BOS/EOS, padding, chat template, concatenation, or truncation is added here.
Sliding-window reference: https://huggingface.co/docs/transformers/en/perplexity
We count *shifted* valid labels directly: a masked overlap must not trigger an
extra subtraction of one token, as that would undercount subsequent windows.
"""

from __future__ import annotations

import math
import sys

import torch
from torch.nn import functional as F

LOSS_CHUNK_SIZE = 64


def _validate_documents(documents: list[dict], window: int, stride: int) -> None:
    if type(window) is not int or type(stride) is not int or not 0 < stride < window:
        raise ValueError("window/stride must be integers satisfying 0 < stride < window")
    if not isinstance(documents, list) or not documents:
        raise ValueError("documents must be a nonempty list; no documents are silently skipped")
    identifiers = set()
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("each document must be a dict with id and token_ids")
        identifier = document.get("id")
        if not isinstance(identifier, str) or not identifier.strip() or identifier in identifiers:
            raise ValueError("document ids must be unique nonempty strings")
        identifiers.add(identifier)
        tokens = document.get("token_ids")
        if not isinstance(tokens, list) or len(tokens) < 2:
            raise ValueError(f"{identifier}: token_ids must contain at least two tokens")
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError(f"{identifier}: token_ids must be nonnegative integers")


def _metrics(nll_sum: float, count: int) -> dict:
    nll = nll_sum / count
    if not math.isfinite(nll):
        raise ValueError("model produced a nonfinite language-model loss")
    overflow = nll > math.log(sys.float_info.max)
    return {
        "nll_sum": nll_sum,
        "valid_token_count": count,
        "nll": nll,
        # JSON has no infinity value. Preserve the NLL for pathological models.
        "ppl": None if overflow else math.exp(nll),
        "ppl_overflow": overflow,
    }


@torch.inference_mode()
def evaluate_ppl(model, documents: list[dict], device, window: int = 512, stride: int = 256) -> dict:
    """Score positions 1..N-1 exactly once in each pretokenized document.

    Windows start at 0, stride, 2*stride, ... and end at min(start+window,N).
    The first window scores all targets after the document's initial token.
    Later windows score only positions at or beyond the preceding window end.
    Stop immediately when the end reaches N. Context and position IDs restart
    for each forward pass, no KV cache is reused, and document boundaries are
    never crossed. The first token has no supplied predecessor and is unscored.

    The model must expose Hugging Face-style ``outputs.logits``. It remains on
    the caller-selected device; this helper does not load or move model weights.
    PPL is exp(total NLL / valid targets), not a mean of document PPL values.
    A PPL exceeding floating-point range is represented by null with an explicit
    overflow flag, while a nonfinite model loss raises rather than being skipped.
    """
    _validate_documents(documents, window, stride)
    eval_device = torch.device(device)
    was_training = model.training
    model.eval()
    rows = []
    try:
        for document_index, document in enumerate(documents, 1):
            tokens = document["token_ids"]
            previous_end = 0
            window_rows = []
            losses = []
            count = 0
            for start in range(0, len(tokens), stride):
                end = min(start + window, len(tokens))
                input_ids = torch.tensor([tokens[start:end]], dtype=torch.long, device=eval_device)
                first_target = max(1, previous_end)
                local_first_target = first_target - start
                # All targets form one contiguous range. This equals counting
                # labels[:, 1:] != -100 after masking the overlap, without a
                # CUDA-to-host synchronization or a dynamic boolean gather.
                valid_count = end - first_target
                if not 1 <= local_first_target < end - start or valid_count <= 0:
                    raise RuntimeError("sliding-window target coverage invariant failed")
                print(f"LM document {document_index}/{len(documents)} {document['id']}: "
                      f"window {len(window_rows) + 1} [{start}, {end}), "
                      f"targets {count + valid_count}/{len(tokens) - 1}", flush=True)

                outputs = model(input_ids=input_ids, use_cache=False)
                logits = outputs.logits
                if logits.ndim != 3 or logits.shape[:2] != input_ids.shape:
                    raise ValueError("model logits must have shape [1, window_tokens, vocabulary]")
                # The large vocabulary makes a full-window float32 CE copy
                # expensive. Basic slices are views; upcast at most 64 target
                # rows at once, keeping the same per-token float32 formula.
                window_nll = torch.zeros((), dtype=torch.float64, device=eval_device)
                for local_target in range(local_first_target, end - start, LOSS_CHUNK_SIZE):
                    target_end = min(local_target + LOSS_CHUNK_SIZE, end - start)
                    target_logits = logits[:, local_target - 1:target_end - 1, :].reshape(-1, logits.shape[-1]).float()
                    target_ids = input_ids[:, local_target:target_end].reshape(-1)
                    token_losses = F.cross_entropy(target_logits, target_ids, reduction="none")
                    # Accumulate on-device; only synchronize once per window.
                    window_nll.add_(token_losses.double().sum())
                    del target_logits, target_ids, token_losses
                nll_sum = float(window_nll.item())
                if not math.isfinite(nll_sum):
                    raise ValueError(f"{document['id']}: nonfinite loss in window [{start}, {end})")
                losses.append(nll_sum)
                count += valid_count
                window_rows.append({
                    "start": start,
                    "end": end,
                    "first_target": first_target,
                    "target_count": valid_count,
                })
                previous_end = end
                # Release large vocabulary logits before the next forward pass.
                del outputs, logits, input_ids, window_nll
                if end == len(tokens):
                    break

            if count != len(tokens) - 1:
                raise RuntimeError("document target count must equal token_count - 1")
            rows.append({
                "id": document["id"],
                "token_count": len(tokens),
                **_metrics(math.fsum(losses), count),
                "windows": window_rows,
            })
            # No window tensors remain referenced here. Return unused cached
            # blocks to the driver between documents on a shared 8GB GPU.
            if eval_device.type == "cuda":
                with torch.cuda.device(eval_device):
                    torch.cuda.empty_cache()
            print(f"LM document {document_index}/{len(documents)} {document['id']} completed: "
                  f"{count} targets, NLL={rows[-1]['nll']:.6f}", flush=True)
    finally:
        model.train(was_training)

    return {
        "schema_version": 1,
        **_metrics(math.fsum(row["nll_sum"] for row in rows), sum(row["valid_token_count"] for row in rows)),
        "document_count": len(rows),
        "documents": rows,
        "protocol": {
            "window": window,
            "stride": stride,
            "batch_size": 1,
            "document_context": "independent; reset at every document; no concatenation",
            "special_tokens": "use supplied token_ids unchanged; add no BOS/EOS or chat template",
            "window_offsets": "zero-based, start inclusive, end exclusive; start advances by stride",
            "position_ids": "model default; restart at each forward pass",
            "target_rule": "score every document position 1..N-1 exactly once; overlap is context only",
            "valid_token_count": "end - first_target; equals count labels[:, 1:] != -100 after causal label shift and overlap masking",
            "aggregation": "sum token NLL / sum valid tokens; PPL = exp(pooled NLL)",
            "loss_precision": "float32 cross entropy; float64 token-loss sum",
            "loss_chunk_size": LOSS_CHUNK_SIZE,
            "target_selection": "contiguous basic slices; no boolean gather",
            "cuda_cache_clearing": "empty_cache after each document, after all temporary tensors are released; CPU does not call CUDA",
            "use_cache": False,
            "reference": "https://huggingface.co/docs/transformers/en/perplexity",
        },
    }
