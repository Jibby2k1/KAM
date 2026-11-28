#!/usr/bin/env python3
"""Build independent PG-19 and MuSiQue panels at matched 4K and 8K lengths."""
import argparse
import base64
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from audit_kam_natural_data import stable
from kam_adapter_preservation import digest
from kam_math_kl_protocol import audit, write_json
from prepare_kam_long_sparse import CONTEXT, TARGETS, freeze_retrieval, load_locked_tokenizer, tokenize_retrieval
from prepare_kam_natural_router import title_key

WIDTHS = (CONTEXT, 2*CONTEXT)
QUESTIONS = 128
PROTOCOL = "kam-attention-transfer-v1"


def read(path):
    return json.loads(path.read_text())


def acquire_books(index, work, prior):
    raw = read(index)
    assert len(raw["items"]) == 100 and "nextPageToken" not in raw
    used = {b["id"] for path in prior for b in read(path)["pg19"]}
    candidates = [row for row in raw["items"] if Path(row["name"]).stem not in used]
    assert len(used) == 72 and len(candidates) == 28
    pg = work / "pg19"
    pg.mkdir(parents=True, exist_ok=True)

    def download(row):
        name = row["name"]
        assert name.startswith("test/") and Path(name).suffix == ".txt"
        path = pg / Path(name).name
        valid = lambda b: len(b) == int(row["size"]) and base64.b64encode(hashlib.md5(b).digest()).decode() == row["md5Hash"]
        if path.exists() and valid(path.read_bytes()):
            return
        with urlopen("https://storage.googleapis.com/deepmind-gutenberg/" + quote(name), timeout=30) as reply:
            content = reply.read()
        assert valid(content), name
        path.write_bytes(content)

    candidates.sort(key=lambda row: stable(PROTOCOL + row["name"]))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, candidates))
    write_json(pg / "sample.json", candidates)


def freeze(source, reservations, work, earlier_pools):
    raw = [read(p) for p in earlier_pools]
    assert all(row["source_sha256"] == digest(source) for row in raw)
    used = [r for x in raw for r in x["cases"]]
    held_ids = set(reservations.read_text().split()) | {r["id"] for r in used}
    titles = {title_key(r[key]["title"]) for r in used for key in ("earlier", "gold")}
    families = {r["family"] for r in used}
    path = work / "transfer-reservations.txt"
    path.write_text("\n".join(sorted(held_ids)) + "\n")
    freeze_retrieval(source, path, work, count=QUESTIONS, excluded_titles=titles, excluded_families=families)


def encode(root, work):
    tokenizer, tokenizer_path = load_locked_tokenizer(root)
    widest = WIDTHS[-1]
    pg = work / "pg19"
    books, ineligible = [], []
    for row in read(pg / "sample.json"):
        path = pg / Path(row["name"]).name
        seq = tokenizer.encode(path.read_text(errors="replace"), add_special_tokens=False)
        if len(seq) < widest+1:
            ineligible.append({"id": path.stem, "tokens": len(seq)})
            continue
        starts = range(0, len(seq)-widest, widest+1)
        start = min(starts, key=lambda i: stable(PROTOCOL + path.stem + str(i)))
        books.append({"id": path.stem, "window": seq[start:start+widest+1], "start": start, "source_sha256": digest(path)})
    assert len(books)+len(ineligible) == 28 and len(books) >= 16
    raw = read(work / "retrieval.json")
    fixtures = {}
    for width in WIDTHS:
        retrieval, cases = tokenize_retrieval(tokenizer, raw, QUESTIONS, context=width)
        output = work / f"transfer-{width}.json"
        write_json(output, {"pg19": [{"id": b["id"], "windows": [b["window"][-width-1:]]} for b in books],
                            "retrieval": retrieval})
        fixtures[str(width)] = {"sha256": digest(output), "questions": cases, "splits": Counter(r["split"] for r in cases)}
    assert [r["id"] for r in fixtures[str(WIDTHS[0])]["questions"]] == [r["id"] for r in fixtures[str(widest)]["questions"]]
    write_json(work / "transfer-manifest.json", {"protocol": PROTOCOL, "tokenizer_sha256": digest(tokenizer_path),
        "pg19": [{k: b[k] for k in ("id", "start", "source_sha256")} for b in books], "short_books": ineligible,
        "scored_tokens_per_window": TARGETS, "source_sha256": raw["source_sha256"],
        "reservations_sha256": raw["reserved_sha256"], "fixture": fixtures})


def source_audit(root, work):
    manifest = read(work / "transfer-manifest.json")
    selected = {r["id"] for r in manifest["fixture"][str(WIDTHS[-1])]["questions"]}
    docs = {}
    for row in read(work / "retrieval.json")["cases"]:
        if row["id"] not in selected:
            continue
        for side in ("earlier", "gold", "foil"):
            doc = row[side]
            docs[doc["id"]] = doc["body"]
    inputs = work / "source-audit"
    inputs.mkdir(exist_ok=True)
    write_json(inputs / "dev.json", [])
    write_json(inputs / "test.json", [{"id": ident, "problem": body} for ident, body in sorted(docs.items())])
    audit(root, inputs, work / "source-audit.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    cli = p.add_subparsers(dest="action", required=True)
    a = cli.add_parser("books")
    a.add_argument("index", type=Path)
    a.add_argument("work", type=Path)
    a.add_argument("prior", type=Path, nargs=2)
    b = cli.add_parser("freeze")
    for name in ("source", "reservations", "work"):
        b.add_argument(name, type=Path)
    b.add_argument("earlier_pools", type=Path, nargs=2)
    c = cli.add_parser("encode")
    for name in ("root", "work"):
        c.add_argument(name, type=Path)
    d = cli.add_parser("audit")
    for name in ("root", "work"):
        d.add_argument(name, type=Path)
    args = p.parse_args()
    if args.action == "books":
        acquire_books(args.index, args.work, args.prior)
    elif args.action == "freeze":
        freeze(args.source, args.reservations, args.work, args.earlier_pools)
    elif args.action == "encode":
        encode(args.root, args.work)
    else:
        source_audit(args.root, args.work)
