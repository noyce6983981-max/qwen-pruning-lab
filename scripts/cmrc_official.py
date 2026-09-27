"""Strict adapter for the pinned CMRC2018 v5-special official scorer.

The upstream Python 2 file is retained byte-for-byte. Only Python 3 language
compatibility is applied in memory; no metric or normalization is reimplemented.
All reported EM/F1 numbers, including per-item values, are percentages (0–100).
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import types


VENDOR_DIR = Path(__file__).resolve().parents[1] / "third_party" / "cmrc2018"
OFFICIAL_PATH = VENDOR_DIR / "cmrc2018_evaluate.py"
UPSTREAM_REVISION = "c0eb1b6ba219847457e6af3180da722bbeb656af"
OFFICIAL_SHA256 = "56da606758e4d8607c37af6acecfb4811ede6eb10dd54031deb75231324ba783"
NLTK_VERSION = "3.9.2"

# Exact, checked substitutions against the hash-verified Python 2 source.
# Python 3 strings are already Unicode: removing decode is equivalent for the
# string-only public interface below. The regex itself and metric stay intact.
PY3_REPLACEMENTS = (
    ("reload(sys)\nsys.setdefaultencoding('utf8')\n", "", 1),
    ("str(in_str).decode('utf-8')", "str(in_str)", 2),
    ("ur'[\\u4e00-\\u9fa5]'", "r'[\\u4e00-\\u9fa5]'", 1),
    ("str(prediction_file[query_id]).decode('utf-8')", "str(prediction_file[query_id])", 1),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_source() -> str:
    if _sha256(OFFICIAL_PATH) != OFFICIAL_SHA256:
        raise RuntimeError("Vendored CMRC2018 scorer hash mismatch; restore the pinned source.")
    source = OFFICIAL_PATH.read_text(encoding="utf-8")
    for old, new, expected_count in PY3_REPLACEMENTS:
        if source.count(old) != expected_count:
            raise RuntimeError("Unexpected upstream source; Python 3 compatibility patch rejected.")
        source = source.replace(old, new)
    return source


def _prepare_nltk() -> dict:
    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError(f"Install the evaluation dependency: python -m pip install nltk=={NLTK_VERSION}") from exc
    if nltk.__version__ != NLTK_VERSION:
        raise RuntimeError(f"This protocol pins nltk=={NLTK_VERSION}; found {nltk.__version__}.")
    provenance = json.loads((VENDOR_DIR / "nltk_punkt_provenance.json").read_text(encoding="utf-8"))
    for row in provenance["files"]:
        if not (VENDOR_DIR / row["path"]).is_file():
            raise RuntimeError("Pinned Punkt resources are missing; run python third_party/cmrc2018/fetch_punkt.py once.")
        if _sha256(VENDOR_DIR / row["path"]) != row["sha256"]:
            raise RuntimeError(f"Pinned NLTK tokenizer resource hash mismatch: {row['path']}")
    # Restrict Punkt to the bundled resources even if the user's global NLTK
    # data directory contains a different copy. Prime NLTK's normal cache, then
    # restore global search paths; the scorer still calls nltk.word_tokenize.
    previous = nltk.data.path[:]
    try:
        nltk.data.path[:] = [str(VENDOR_DIR / "nltk_data")]
        nltk.tokenize._get_punkt_tokenizer.cache_clear()
        nltk.tokenize._get_punkt_tokenizer("english")
    finally:
        nltk.data.path[:] = previous
    return provenance


@lru_cache(maxsize=1)
def load_official_scorer() -> types.ModuleType:
    """Load the official functions after hash and dependency verification."""
    _prepare_nltk()
    module = types.ModuleType("cmrc2018_official_pinned")
    module.__file__ = str(OFFICIAL_PATH)
    exec(compile(_verified_source(), str(OFFICIAL_PATH), "exec"), module.__dict__)
    return module


def scoring_provenance() -> dict:
    """Return auditable source, compatibility, dependency and resource details."""
    load_official_scorer()
    punkt = json.loads((VENDOR_DIR / "nltk_punkt_provenance.json").read_text(encoding="utf-8"))
    return {
        "metric": "cmrc2018_official_v5_special",
        "unit": "percent",
        "source_url": f"https://github.com/ymcui/cmrc2018/blob/{UPSTREAM_REVISION}/squad-style-data/cmrc2018_evaluate.py",
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_sha256": _sha256(OFFICIAL_PATH),
        "adapter_sha256": _sha256(Path(__file__)),
        "python3_compatibility": "four checked substitutions; upstream file unmodified",
        "nltk_version": NLTK_VERSION,
        "punkt": punkt,
    }


def _validate(examples: list[dict], predictions: Mapping[str, str]) -> None:
    if not isinstance(examples, list) or not examples:
        raise ValueError("examples must be a nonempty list")
    ids = set()
    for index, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"example {index} must be an object")
        qid = example.get("id")
        if not isinstance(qid, str) or not qid or qid != qid.strip():
            raise ValueError(f"example {index} needs a nonempty string id without outer whitespace")
        if qid in ids:
            raise ValueError(f"duplicate example id: {qid}")
        ids.add(qid)
        answers = example.get("answers")
        if not isinstance(answers, list) or not answers or any(not isinstance(a, str) or not a.strip() for a in answers):
            raise ValueError(f"answers for {qid} must be a nonempty list of nonblank strings")
    if not isinstance(predictions, Mapping):
        raise ValueError("predictions must map ids to answer strings")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in predictions.items()):
        raise ValueError("prediction ids and answers must be strings")
    missing, extra = ids - predictions.keys(), predictions.keys() - ids
    if missing or extra:
        raise ValueError(f"prediction id mismatch: missing={sorted(missing)}, extra={sorted(extra)}")


def score_examples(examples: list[dict], predictions: Mapping[str, str]) -> dict:
    """Score flat {id, answers: [str]} records with exact prediction coverage.

    Empty model predictions are allowed and normally score zero. There is no
    answer extraction or model-output cleanup here: that belongs to the locked
    inference protocol. Extra example metadata is ignored.
    """
    _validate(examples, predictions)
    official = load_official_scorer()
    qas = [{"id": e["id"], "question": "", "answers": [{"text": a} for a in e["answers"]]} for e in examples]
    dataset = {"data": [{"paragraphs": [{"qas": qas}]}]}
    f1, em, total, skipped = official.evaluate(dataset, predictions)
    per_item = [
        {"id": e["id"],
         "em": 100.0 * official.calc_em_score(e["answers"], predictions[e["id"]]),
         "f1": 100.0 * official.calc_f1_score(e["answers"], predictions[e["id"]])}
        for e in examples
    ]
    return {"metric": "cmrc2018_official", "count": total, "skipped": skipped,
            "em": em, "f1": f1, "average": (em + f1) / 2, "per_item": per_item}


def _unique_object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json_strict(path: str | Path):
    """Reject duplicate JSON object keys before they can overwrite predictions."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("examples", type=Path, help="JSON list of flat examples")
    parser.add_argument("predictions", type=Path, help="JSON object mapping id to prediction")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score_examples(load_json_strict(args.examples), load_json_strict(args.predictions))
    result["provenance"] = scoring_provenance()
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
