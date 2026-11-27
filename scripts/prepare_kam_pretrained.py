#!/usr/bin/env python3
"""Tokenize the frozen training mixture once for both adapter arms."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

DOMAINS = ("wiki", "code", "math")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(root):
    tokenizer = AutoTokenizer.from_pretrained(root / "model", local_files_only=True)
    entries = {}
    old = json.loads((root / "corpus" / "manifest.json").read_text())
    for split in ("train", "validation", "test"):
        for domain in DOMAINS:
            name = f"{domain}-{split}"
            source = root / "corpus" / f"{name}.bin"
            assert digest(source) == old["corpus"][name]["sha256"]
            documents = source.read_bytes().decode("utf-8").split("\n\x00\n")
            path = root / f"{name}.u32"
            limit = 2_000_000 if split == "train" else 80_000
            tokens = 0
            with path.open("wb") as out:
                for start in range(0, len(documents), 128):
                    encoded = tokenizer(documents[start:start+128], add_special_tokens=False)["input_ids"]
                    flat = [index for row in encoded if row for index in [*row, tokenizer.eos_token_id]]
                    payload = np.asarray(flat[:limit-tokens], dtype="<u4")
                    out.write(payload.tobytes())
                    tokens += len(payload)
                    if tokens >= limit:
                        break
            assert tokens > 1024
            entries[name] = {"tokens": tokens, "sha256": digest(path), "source_sha256": old["corpus"][name]["sha256"]}
            print(name, tokens, flush=True)
    (root / "token-manifest.json").write_text(json.dumps(entries, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    prepare(parser.parse_args().root)
