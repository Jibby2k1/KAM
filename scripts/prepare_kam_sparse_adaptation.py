#!/usr/bin/env python3
"""Freeze independent 4K PG-19 training windows for the sparse adaptation pilot."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

from audit_kam_natural_data import stable
from kam_adapter_preservation import digest
from kam_math_kl_protocol import write_json
from prepare_kam_attention_transfer import download_books, read
from prepare_kam_long_sparse import CONTEXT, load_locked_tokenizer

PROTOCOL = "kam-sparse-adaptation-v1"
STEPS = 128
BOOKS = 96
MAX_WINDOWS_PER_BOOK = 3


def acquire(work):
    values = []
    options = {"prefix": "train/", "maxResults": "1000", "fields": "nextPageToken,items(name,size,md5Hash)"}
    while True:
        url = "https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o?" + urlencode(options)
        with urlopen(url, timeout=30) as response:
            batch = json.load(response)
        values.extend(row for row in batch["items"] if row["name"].endswith(".txt"))
        if "nextPageToken" not in batch:
            break
        options["pageToken"] = batch["nextPageToken"]
    assert len(values) == len(set(row["name"] for row in values)) and len(values) > 25000
    chosen = sorted(values, key=lambda r: stable(PROTOCOL + r["name"]))[:BOOKS]
    save = work / "training-books.json"
    write_json(save, {"available": len(values), "source": "PG-19 train", "objects": chosen})
    download_books(chosen, work / "pg19", "train")


def encode(root, work):
    tok, tokenizer_path = load_locked_tokenizer(root)
    manifest = work / "training-books.json"
    examples, books = [], []
    for item in read(manifest)["objects"]:
        path = work / "pg19" / Path(item["name"]).name
        full = tok.encode(path.read_text(errors="replace"), add_special_tokens=False, verbose=False)
        width = CONTEXT + 1
        ranges = sorted(range(len(full)//width), key=lambda i: stable(PROTOCOL+path.stem+":"+str(i)))
        starts = [index*width for index in ranges[:MAX_WINDOWS_PER_BOOK]]
        books.append({"id": path.stem, "source_sha256": digest(path), "starts": starts})
        examples.extend({"id": path.stem, "start": start, "tokens": full[start:start+width]} for start in starts)
    assert len(examples) >= STEPS*1.5
    inputs = work / "sparse-train-inputs.json"
    write_json(inputs, examples)
    write_json(work / "sparse-train-manifest.json", {"protocol": PROTOCOL, "source_books_sha256": digest(manifest),
        "tokenizer_sha256": digest(tokenizer_path), "inputs_sha256": digest(inputs), "width": CONTEXT,
        "steps": STEPS, "books": books, "available_windows": len(examples)})


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="mode", required=True)
    get = sub.add_parser("download")
    get.add_argument("work", type=Path)
    enc = sub.add_parser("encode")
    enc.add_argument("root", type=Path)
    enc.add_argument("work", type=Path)
    a = cli.parse_args()
    if a.mode == "download":
        a.work.mkdir(parents=True, exist_ok=True)
        acquire(a.work)
    else:
        encode(a.root, a.work)
