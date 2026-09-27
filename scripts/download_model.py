"""Download one pinned official Qwen checkpoint. Does not execute model code."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-1.7B"
REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} at {REVISION}", flush=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=str(args.model_dir.resolve()),
        allow_patterns=["*.json", "*.safetensors", "*.txt", "LICENSE", "README.md"],
        max_workers=2,
        token=False,
    )
    index_path = args.model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    required = sorted(set(index["weight_map"].values()))
    required += ["config.json", "tokenizer.json", "tokenizer_config.json"]
    missing = [name for name in required if not (args.model_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"Incomplete model download: {missing}")
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "source": "https://huggingface.co/" + MODEL_ID,
        "download_completed_utc": datetime.now(timezone.utc).isoformat(),
        "files": [
            {"name": p.name, "bytes": p.stat().st_size}
            for p in sorted(args.model_dir.iterdir())
            if p.is_file() and p.name != "download_manifest.json"
        ],
    }
    (args.model_dir / "download_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Complete: {args.model_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
