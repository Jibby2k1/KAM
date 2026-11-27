#!/usr/bin/env python3
"""Freeze disjoint evidence spans and hard same-question negatives for MuSiQue."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import ZipFile

from audit_kam_natural_data import digest, stable, words
from prepare_kam_musique_decomposition import title_sha
from prepare_kam_native_musique import MEMBERS, canonical_paragraph
from train_kam_musique_decomposer import REFERENCE

MAX_ANSWER_WORDS = 12


def literal_spans(text, answer):
    if not answer:
        return []
    phrase = r"\s+".join(re.escape(word) for word in answer.split())
    return [m.span() for m in re.finditer(r"(?<!\w)" + phrase + r"(?!\w)", text, flags=re.IGNORECASE)]


def read_split(archive, member, reserve_ids):
    result, stamp = {}, sha256()
    with archive.open(member) as source:
        for line in source:
            stamp.update(line)
            row = json.loads(line)
            if row["id"] in reserve_ids:
                result[row["id"]] = row
    assert result.keys() == reserve_ids
    return result, stamp.hexdigest()


def groups(q, blocked_titles, blocked_bodies):
    previous = []
    evidence = q["paragraphs"]
    citations = q["question_decomposition"]
    available = []
    for paragraph in evidence:
        title, body = canonical_paragraph(paragraph)
        available.append(title_sha(title) not in blocked_titles and stable(body) not in blocked_bodies)
    for step, hop in enumerate(citations):
        assert all(0 < int(j) <= step for j in REFERENCE.findall(hop["question"]))
        filled = REFERENCE.sub(lambda m: previous[int(m.group(1)) - 1], hop["question"]).strip()
        reply = " ".join(hop["answer"].split())
        previous.append(reply)
        index = hop["paragraph_support_idx"]
        support = evidence[index]
        assert support["is_supporting"]
        if not (filled and available[index] and 1 <= len(words(reply)) <= MAX_ANSWER_WORDS):
            continue
        passage = support["title"] + ". " + support["paragraph_text"]
        if not literal_spans(passage, reply):
            continue
        question = set(words(filled))
        answer_words = " " + " ".join(words(reply)) + " "
        plausible = []
        for i, distractor in enumerate(evidence):
            if not available[i] or distractor["is_supporting"]:
                continue
            body = distractor["title"] + ". " + distractor["paragraph_text"]
            if answer_words in (" " + " ".join(words(body)) + " ") or literal_spans(body, reply):
                continue
            plausible.append((len(question.intersection(words(body))), stable(f"{q['id']}:{step}:{i}"), i, body))
        if not plausible:
            continue
        _, _, neg, body = max(plausible)
        for active, slot, text, answer in ((True, index, passage, reply), (False, neg, body, None)):
            title, prose = canonical_paragraph(evidence[slot])
            yield {"id": q["id"], "step": step + 1, "hops": len(citations), "has_answer": active,
                   "title_sha256": title_sha(title), "body_sha256": stable(prose), "slot": slot,
                   "question": filled, "evidence": text, "answer": answer}


def prepare(a):
    def load(path):
        with gzip.open(path, "rt") as stream:
            return json.load(stream)

    test, fresh, track = [load(f) for f in (a.native, a.fresh, a.partition)]
    official = digest(a.source)
    assert official == test["source_sha256"] == fresh["source_sha256"] == track["source_sha256"]
    reserved_ids = {q["id"] for source in (test, fresh) for q in source["rows"]}
    assert len(reserved_ids) == sum(len(t["rows"]) for t in (test, fresh))
    fit_ids, monitor_ids = [{q["id"] for q in track["ids"][name]} for name in ("fit", "validation")]
    assert fit_ids.isdisjoint(monitor_ids) and reserved_ids.isdisjoint(fit_ids | monitor_ids)
    with ZipFile(a.source) as source:
        held, held_hash = read_split(source, MEMBERS["dev"], reserved_ids)
        trained, train_hash = read_split(source, MEMBERS["train"], fit_ids | monitor_ids)
    assert held_hash == test["member_sha256"]["dev"] and train_hash == track["training_member_sha256"]

    def support_fingerprint(questions):
        descriptions = [canonical_paragraph(doc) for q in questions for doc in q["paragraphs"] if doc["is_supporting"]]
        return {title_sha(title) for title, _ in descriptions}, {stable(body) for _, body in descriptions}

    test_titles, test_bodies = support_fingerprint(held.values())
    monitor_titles, monitor_bodies = support_fingerprint(trained[q] for q in monitor_ids)
    tests_wording = {tuple(words(x["question"])) for x in held.values()}
    data = {}
    for setting, ids in (("monitor", monitor_ids), ("fit", fit_ids)):
        blocked_titles = test_titles if setting == "monitor" else test_titles | monitor_titles
        blocked_bodies = test_bodies if setting == "monitor" else test_bodies | monitor_bodies
        rows = []
        for id_ in sorted(ids, key=lambda item: stable(f"kam-span-{setting}-v1:{item}")):
            q = trained[id_]
            assert q["answerable"] and tuple(words(q["question"])) not in tests_wording
            rows.extend(groups(q, blocked_titles, blocked_bodies))
        assert len(rows) >= 200 and len(rows) % 2 == 0
        assert sum(r["has_answer"] for r in rows) == len(rows) // 2
        data[setting] = rows
        if setting == "monitor":
            monitor_titles.update(r["title_sha256"] for r in rows)
            monitor_bodies.update(r["body_sha256"] for r in rows)
    assert {q["title_sha256"] for q in data["fit"]}.isdisjoint(q["title_sha256"] for q in data["monitor"])
    assert {q["body_sha256"] for q in data["fit"]}.isdisjoint(q["body_sha256"] for q in data["monitor"])
    meta = {"source_sha256": official, "native_sha256": test["fixture_sha256"], "fresh_sha256": fresh["fixture_sha256"],
            "partition_sha256": digest(a.partition), "fit_member_sha256": train_hash, "dev_member_sha256": held_hash,
            "rows_by_depth": {setting: dict(Counter(r["hops"] for r in rows)) for setting, rows in data.items()},
            "heldout_support_titles": len(test_titles)}
    a.output.write_bytes(gzip.compress(json.dumps(meta | {"splits": data}, separators=(",", ":"), ensure_ascii=False).encode(), mtime=0))
    published = {setting: [{k: v for k, v in r.items() if k not in ("question", "evidence", "answer")} for r in rows]
                 for setting, rows in data.items()}
    a.manifest.write_bytes(gzip.compress(json.dumps(meta | {"data_sha256": digest(a.output), "splits": published},
                                                          separators=(",", ":")).encode(), mtime=0))
    print(meta["rows_by_depth"], meta["heldout_support_titles"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "native", "fresh", "partition", "output", "manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
