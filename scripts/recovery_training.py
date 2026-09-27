"""Token-weighted LoRA gradient accumulation for one complete optimizer group."""

from __future__ import annotations

import math

import torch

from trial_lora import autocast, batch_tensors, is_adapter


def batch_groups(rows, size=8):
    """Return consecutive groups without shuffle or a silently shorter tail."""
    if type(size) is not int or size < 1 or not isinstance(rows, list) or not rows:
        raise ValueError("Expected nonempty row list and positive integer group size")
    if len(rows) % size:
        raise ValueError("Training row count must be exactly divisible by accumulation size")
    return [rows[start:start + size] for start in range(0, len(rows), size)]


def _validate_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError("An accumulation group must contain at least one complete row")
    seen, total = set(), 0
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip() or row["id"] in seen:
            raise ValueError("Group rows require unique nonempty IDs")
        seen.add(row["id"])
        ids, labels, mask = (row.get(key) for key in ("input_ids", "labels", "attention_mask"))
        if not isinstance(ids, list) or not 2 <= len(ids) <= 512 or any(type(token) is not int or token < 0 for token in ids):
            raise ValueError("Invalid complete input token IDs")
        if not isinstance(labels, list) or len(labels) != len(ids) or any(type(token) is not int or (token < 0 and token != -100) for token in labels):
            raise ValueError("Labels must match complete input length")
        if not isinstance(mask, list) or len(mask) != len(ids) or any(type(value) is not int or value != 1 for value in mask):
            raise ValueError("Accumulation supports unpadded batch-one rows only")
        count = sum(token != -100 for token in labels[1:])
        if type(row.get("supervised_token_count")) is not int or count < 1 or count != row["supervised_token_count"]:
            raise ValueError("Supervised token count must match valid shifted labels")
        if type(row.get("sequence_length")) is not int or row["sequence_length"] != len(ids):
            raise ValueError("sequence_length must match the untruncated example")
        first = next((index for index, value in enumerate(labels) if value != -100), len(ids))
        if first < 1 or labels[first:] != ids[first:]:
            raise ValueError("Labels must mask the complete prompt and supervise the unchanged answer suffix")
        if "prompt_length" in row and row["prompt_length"] != first:
            raise ValueError("Prompt mask disagrees with prompt_length")
        if "eos_token_id" in row and (type(row["eos_token_id"]) is not int or ids[-1] != row["eos_token_id"]):
            raise ValueError("Complete example must end with its recorded EOS token")
        total += count
    return total


def train_accumulated_group(model, rows, optimizer, device, max_grad_norm=1.0):
    """Backpropagate sum_i(mean_loss_i * target_count_i) / total_targets.

    No optimizer update occurs until every forward/backward and gradient check
    succeeds. Any exception clears all model/optimizer gradients. An exception
    *inside* optimizer.step or a nonfinite post-update parameter is fatal and
    cannot be described as an unattempted or rolled-back update: this helper
    deliberately makes no optimizer-state rollback guarantee.

    Frozen base gradients are forbidden. Trainable A/B parameters and their
    gradients must be finite; frozen base content is audited by the run's base
    fingerprints, while nonfinite computations are rejected at the loss check.
    """
    device = torch.device(device)
    try:
        total = _validate_rows(rows)
        if isinstance(max_grad_norm, bool) or not isinstance(max_grad_norm, (float, int)) or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        named_trainable = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                if parameter.grad is not None:
                    raise ValueError(f"Frozen base has a gradient: {name}")
                continue
            if not is_adapter(name):
                raise ValueError(f"Unexpected trainable base parameter: {name}")
            if not bool(torch.isfinite(parameter).all()):
                raise ValueError(f"Nonfinite adapter parameter before update: {name}")
            named_trainable.append((name, parameter))
        if not named_trainable:
            raise ValueError("No trainable LoRA A/B parameters")
        trainable = [parameter for _, parameter in named_trainable]
        optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        if len(optimizer_parameters) != len(trainable) or {id(p) for p in optimizer_parameters} != {id(p) for p in trainable}:
            raise ValueError("Optimizer must contain every trainable adapter parameter exactly once and no base parameters")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        per_item = []
        for row in rows:
            with autocast(device):
                outputs = model(**batch_tensors(row, device))
                loss = outputs.loss
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise ValueError(f"{row['id']}: nonfinite or nonscalar training loss")
            loss_value = float(loss.detach())
            weight = row["supervised_token_count"] / total
            (loss * weight).backward()
            per_item.append({"id": row["id"], "sequence_length": row["sequence_length"],
                             "supervised_token_count": row["supervised_token_count"],
                             "mean_loss": loss_value, "backward_weight": weight})
            del outputs, loss
        nonzero = {"A_nonzero_tensor_count": 0, "B_nonzero_tensor_count": 0}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                if parameter.grad is not None:
                    raise ValueError(f"Frozen base has a gradient: {name}")
                continue
            if not is_adapter(name) or parameter.grad is None:
                raise ValueError(f"Missing or unexpected trainable gradient: {name}")
            if not bool(torch.isfinite(parameter.grad).all()):
                raise ValueError(f"Nonfinite accumulated gradient: {name}")
            key = "A_nonzero_tensor_count" if ".lora_A." in name else "B_nonzero_tensor_count"
            nonzero[key] += int(bool(torch.count_nonzero(parameter.grad)))
        norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm, error_if_nonfinite=True)
        norm_value = float(norm)
        if not math.isfinite(norm_value) or norm_value <= 0:
            raise ValueError("Accumulated gradient norm must be finite and nonzero")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if any(not bool(torch.isfinite(parameter).all()) for parameter in trainable):
            raise ValueError("Nonfinite adapter after optimizer update; update was already attempted")
        return {"weighted_loss": math.fsum(item["mean_loss"] * item["supervised_token_count"] for item in per_item) / total,
                "supervised_token_count": total, "sample_ids": [row["id"] for row in rows],
                "sequence_lengths": [row["sequence_length"] for row in rows], "per_item": per_item,
                "gradient_norm_before_clip": norm_value, **nonzero}
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        raise
