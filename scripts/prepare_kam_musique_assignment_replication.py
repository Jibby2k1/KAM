#!/usr/bin/env python3
"""Freeze unseen MuSiQue questions for the distinct-assignment replication."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZipFile

from audit_kam_natural_data import digest, stable, words
from prepare_kam_musique_decomposition import title_sha
from prepare_kam_native_musique import MEMBERS, canonical_paragraph, reserve
from run_kam_native_musique import corpus_and_fixture


def freeze(a, bank, base):
    fitted = json.loads(gzip.decompress(a.training.read_bytes()))
    source = digest(a.source)
    base_hash = digest(a.base_fixture)
    assert fitted["fixture_sha256"] == base_hash and fitted["source_sha256"] == base["source_sha256"] == source
    reservation_hashes = list(map(digest, a.reserve))
    assert sorted(reservation_hashes) == sorted(base["reservations"]["fixture_sha256"])
    ids, spellings = reserve(a.reserve)
    assert len(ids) == base["reservations"]["questions"] and len(spellings) == base["reservations"]["normalized_wordings"]
    prior_titles = {title_sha(bank[i]["title"]) for q in base["rows"] for i in q["gold"]}
    prior_titles.update(t for group in fitted["splits"].values() for q in group for t in q["support_title_sha256"])
    for group in [base["rows"], *fitted["splits"].values()]:
        ids.update(q["id"] for q in group)
        spellings.update(tuple(words(q["question"])) for q in group)

    exact = {canonical_paragraph({"title": p["title"], "paragraph_text": p["body"]}): p["id"] for p in bank}
    candidates = []
    raw_hash = sha256()
    with ZipFile(a.source) as archive, archive.open(MEMBERS["dev"]) as stream:
        for line in stream:
            raw_hash.update(line)
            q = json.loads(line)
            spelling = tuple(words(q["question"]))
            if not q["answerable"] or q["id"] in ids or spelling in spellings or not spelling:
                continue
            evidence = [p for p in q["paragraphs"] if p["is_supporting"]]
            hops = int(q["id"][0])
            gold = sorted({exact[canonical_paragraph(p)] for p in evidence})
            if len(gold) != hops or hops not in (2, 3, 4):
                continue
            if any(title_sha(p["title"]) in prior_titles for p in evidence):
                continue
            candidates.append({"id": q["id"], "question": q["question"], "hops": hops, "gold": gold})
    assert raw_hash.hexdigest() == base["member_sha256"]["dev"]
    rows, chains = [], set()
    for row in sorted(candidates, key=lambda q: stable("kam-music-assignment-confirm-v1:" + q["id"])):
        wording = tuple(words(row["question"]))
        chain = tuple(row["gold"])
        if chain in chains or wording in spellings:
            continue
        rows.append(row)
        chains.add(chain)
        spellings.add(wording)
    counts = Counter(q["hops"] for q in rows)
    assert counts[2] >= 128
    meta = {k: base[k] for k in ("source_sha256", "member_sha256", "corpus_sha256", "documents", "training_documents")}
    return meta | {"base_fixture_sha256": base_hash, "decomposer_data_sha256": digest(a.training),
                   "historical_reservation_sha256": sorted(reservation_hashes), "blocked_support_titles": len(prior_titles),
                   "hop_counts": counts, "primary_hops": 2, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "corpus", "base-fixture", "training", "output", "manifest"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--reserve", action="append", required=True, type=Path)
    a = p.parse_args()
    a.fixture = a.base_fixture
    bank, base = corpus_and_fixture(a)
    sealed = freeze(a, bank, base)
    a.output.write_text(json.dumps(sealed, ensure_ascii=False, separators=(",", ":")) + "\n")
    public = sealed | {"fixture_sha256": digest(a.output),
                       "rows": [{k: v for k, v in r.items() if k != "question"} for r in sealed["rows"]]}
    a.manifest.write_bytes(gzip.compress(json.dumps(public, separators=(",", ":")).encode(), mtime=0))
    print(sealed["hop_counts"])


if __name__ == "__main__":
    main()
