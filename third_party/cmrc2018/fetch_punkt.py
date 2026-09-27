"""Fetch the pinned English-only Punkt resource cache used by the scorer.

Run explicitly once; scoring itself never downloads anything. The upstream
Punkt model license is unclarified, so this local cache is excluded from Git.
"""

import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def main():
    manifest = json.loads((ROOT / "nltk_punkt_provenance.json").read_text(encoding="utf-8"))
    if all((ROOT / row["path"]).is_file() and hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest() == row["sha256"] for row in manifest["files"]):
        print("Pinned English Punkt resources already present and verified.")
        return
    with urllib.request.urlopen(manifest["source_url"], timeout=60) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != manifest["archive_sha256"]:
        raise RuntimeError("Punkt archive hash mismatch")
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        for row in manifest["files"]:
            raw = source.read(row["archive_path"])
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise RuntimeError(f"Punkt file hash mismatch: {row['path']}")
            target = (ROOT / row["path"]).resolve()
            if not target.is_relative_to(ROOT / "nltk_data"):
                raise RuntimeError("Resource manifest path escapes cache")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    print("Fetched and verified the pinned English Punkt resources.")


if __name__ == "__main__":
    main()
