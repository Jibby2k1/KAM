#!/usr/bin/env python3
"""Freeze independent fit and development book groups in PG-19 validation."""
import argparse
import json
from pathlib import Path

from audit_kam_natural_data import stable
from kam_adapter_preservation import digest
from kam_math_kl_protocol import write_json
from prepare_kam_attention_transfer import download_books
from prepare_kam_long_sparse import CONTEXT, load_locked_tokenizer

PROTOCOL = "kam-route-distillation-v1"


def read(path):
    return json.loads(path.read_text())


def acquire(work, index):
    pools = read(index)
    assert len(pools["items"]) == 50 and "nextPageToken" not in pools
    rows = sorted(pools["items"], key=lambda r: stable(PROTOCOL+r["name"]))
    download_books(rows, work/"pg19", "validation")
    write_json(work/"pg19"/"sample.json", rows)


def encode(root, work):
    tokenizer, tokenizer_path = load_locked_tokenizer(root)
    source = work/"pg19"/"sample.json"
    books = []
    skipped = []
    for row in read(source):
        path = work/"pg19"/Path(row["name"]).name
        values = tokenizer.encode(path.read_text(errors="replace"), add_special_tokens=False)
        n = len(values)//CONTEXT
        if not n:
            skipped.append(path.stem)
            continue
        order = sorted(range(n), key=lambda i: stable(PROTOCOL+path.stem+":"+str(i)))[:2]
        books.append({"id": path.stem, "source_sha256": digest(path),
                      "blocks": order, "windows": [values[i*CONTEXT:(i+1)*CONTEXT] for i in order]})
    assert len(books)+len(skipped)==50 and len(books)>=36
    groups = {"fit": [b for i,b in enumerate(books) if i%4], "dev": books[::4]}
    path = work/"routing-inputs.json"
    write_json(path, groups)
    write_json(work/"routing-input-manifest.json", {"protocol": PROTOCOL, "source":"PG-19 validation",
        "source_manifest_sha256": digest(source), "tokenizer_sha256": digest(tokenizer_path),
        "inputs_sha256": digest(path), "skipped": skipped, "context": CONTEXT,
        "groups": {name: [{k:r[k] for k in ("id","source_sha256","blocks")} for r in rows] for name,rows in groups.items()}})


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    c=p.add_subparsers(dest="action",required=True)
    a=c.add_parser("books")
    a.add_argument("work",type=Path)
    a.add_argument("index",type=Path)
    b=c.add_parser("encode")
    b.add_argument("root",type=Path)
    b.add_argument("work",type=Path)
    args=p.parse_args()
    if args.action=="books":
        acquire(args.work,args.index)
    else:
        encode(args.root,args.work)
