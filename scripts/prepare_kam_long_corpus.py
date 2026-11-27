#!/usr/bin/env python3
"""Build the frozen prose, Python, and arithmetic mixture."""
import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq

DOMAINS = ("wiki", "code", "math")
SPLITS = ("train", "validation", "test")
LIMITS = {"wiki": 64 << 20, "code": 48 << 20, "math": 16 << 20}
SEED = 731109


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def test_strings(data):
    field = {"mmlu": "question", "gsm8k": "question", "humaneval": "prompt", "mbpp": "text"}
    prompts = [row[key] for name, key in field.items()
               for row in pq.read_table(data / f"{name}.parquet", columns=[key]).to_pylist()]
    with (data / "gpqa.csv").open(newline="", encoding="utf-8-sig") as source:
        prompts.extend(row["Question"] for row in csv.DictReader(source))
    blocked = {}
    for text in prompts:
        text = normalized(text)
        words = text.split()
        if len(words) >= 8:
            blocked.setdefault(" ".join(words[:8]), set()).add(text)
    return blocked


def overlaps(text, blocked):
    words = normalized(text).split()
    for i in range(len(words) - 7):
        prefix = " ".join(words[i:i + 8])
        matches = blocked.get(prefix)
        if matches:
            remaining = " ".join(words[i:])
            if any(remaining == question or remaining.startswith(question + " ") for question in matches):
                return True
    return False


def source_docs(root, domain, split):
    path = root / f"{domain}-{split}.parquet"
    if domain == "code":
        for batch in pq.ParquetFile(path).iter_batches(columns=["whole_func_string"]):
            for value in batch.column(0).to_pylist():
                if value and len(value) > 32:
                    yield value
        return
    article = []
    for batch in pq.ParquetFile(path).iter_batches(columns=["text"]):
        for line in batch.column(0).to_pylist():
            if re.match(r"^\s*=\s+[^=].*?\s+=\s*$", line):
                if article:
                    yield "".join(article)
                article = []
            article.append(line)
    if article:
        yield "".join(article)


def math_docs(split):
    rng = random.Random(SEED + SPLITS.index(split))
    for i in range(500_000):
        a, b, c = rng.sample(range(2, 201), 3)
        kind = i % 4
        if kind == 0:
            q = f"A store sells {a} crates with {b} bottles in each and {c} loose bottles. How many bottles did it sell?"
            equation, value = f"{a} * {b} + {c}", a * b + c
        elif kind == 1:
            q = f"A community hall has {a} rows of {b} chairs. If {c} more chairs are brought in, how many chairs are there?"
            equation, value = f"{a} * {b} + {c}", a * b + c
        elif kind == 2:
            q = f"An orchard begins with {a*b+c} apples. It gives away {a} bags of {b} apples each. How many apples remain?"
            equation, value = f"{a*b+c} - {a} * {b}", c
        else:
            q = f"A shipment contains {a*b} plates distributed evenly among {a} boxes. How many plates are in each box?"
            equation, value = f"{a*b} / {a}", b
        yield f"Question: {q}\nAnswer (end with #### and the number): {equation} = {value}.\n#### {value}\n"


def build(root):
    blocked = test_strings(root / "data")
    seen = set()
    metadata = {"overlap_filter": "normalized exact test prompt substring, minimum eight words, within candidate document",
                "sources": {}, "corpus": {}}
    for split in ("test", "validation", "train"):
        for domain in DOMAINS:
            name = f"{domain}-{split}"
            if domain == "math":
                docs = math_docs(split)
            else:
                path = root / "source" / f"{name}.parquet"
                metadata["sources"][name] = sha(path)
                docs = source_docs(root / "source", domain, split)
            limit = LIMITS[domain] if split == "train" else 512 << 10
            target = root / "corpus" / f"{name}.bin"
            kept = duplicates = overlap = 0
            with target.open("wb") as stream:
                for text in docs:
                    payload = text.encode("utf-8").replace(b"\x00", b"")
                    if not payload.strip():
                        continue
                    key = hashlib.sha256(normalized(text).encode()).digest()
                    if key in seen:
                        duplicates += 1
                        continue
                    if split == "train" and overlaps(text, blocked):
                        overlap += 1
                        continue
                    seen.add(key)
                    stream.write(payload + b"\n\x00\n")
                    kept += 1
                    if stream.tell() >= limit:
                        break
            assert target.stat().st_size >= min(limit, 128 << 10), name
            metadata["corpus"][name] = {
                "bytes": target.stat().st_size, "sha256": sha(target), "documents": kept,
                "duplicate_documents_skipped": duplicates, "overlap_documents_skipped": overlap,
            }
            print(name, metadata["corpus"][name], flush=True)
    (root / "corpus" / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    build(parser.parse_args().root)
