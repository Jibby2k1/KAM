#!/usr/bin/env python3
"""Stage immutable, bounded TinyStories splits for Phase 6 confirmation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import urllib.request
from pathlib import Path


TRAIN_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt?download=true"
VALID_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt?download=true"
TRAIN_BYTES = 128 * 1024 * 1024
MAX_VALID_BYTES = 64 * 1024 * 1024


def _download_prefix(url: str, limit: int) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"Range": f"bytes=0-{limit - 1}", "User-Agent": "KAM-Phase6-confirmation/2"},
    )
    chunks: list[bytes] = []
    remaining = limit
    with urllib.request.urlopen(request, timeout=120) as response:
        while remaining > 0:
            chunk = response.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
    payload = b"".join(chunks)
    if len(payload) < 4096:
        raise RuntimeError(f"downloaded corpus is undersized: {len(payload)} bytes from {url}")
    return payload


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def prepare(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "TinyStoriesV2-GPT4-train.128MiB.txt"
    validation_path = root / "TinyStories-valid.validation.txt"
    test_path = root / "TinyStories-valid.test.txt"
    manifest_path = root / "corpus_manifest.json"
    if manifest_path.exists() and train_path.exists() and validation_path.exists() and test_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current = {
            "train": _sha(train_path.read_bytes()),
            "validation": _sha(validation_path.read_bytes()),
            "test": _sha(test_path.read_bytes()),
        }
        if current == manifest.get("sha256"):
            return manifest
        raise RuntimeError("existing confirmation corpus files do not match corpus_manifest.json")
    train = _download_prefix(TRAIN_URL, TRAIN_BYTES)
    if len(train) != TRAIN_BYTES:
        raise RuntimeError(f"expected exactly {TRAIN_BYTES} training bytes, received {len(train)}")
    valid = _download_prefix(VALID_URL, MAX_VALID_BYTES)
    midpoint = len(valid) // 2
    validation = valid[:midpoint]
    test = valid[midpoint:]
    if min(len(validation), len(test)) < 1_000_000:
        raise RuntimeError("TinyStories validation/test split is too small")
    train_path.write_bytes(train)
    validation_path.write_bytes(validation)
    test_path.write_bytes(test)
    manifest = {
        "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "license": "cdla-sharing-1.0",
        "source": {"train": TRAIN_URL, "validation_and_test": VALID_URL},
        "bytes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "sha256": {"train": _sha(train), "validation": _sha(validation), "test": _sha(test)},
        "split_overlap": False,
        "tokenizer": "immutable_byte_256",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/phase6_confirmation")
    args = parser.parse_args()
    print(json.dumps(prepare(Path(args.root)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
