#!/usr/bin/env python3
"""Stream LongMemEval-M into a deduplicated session bank and sealed query panels."""

import argparse
from collections import Counter
from datetime import datetime
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re

from audit_kam_natural_data import digest

DATE = re.compile(r"^(\d{4})/(\d\d)/(\d\d) \([^)]*\) (\d\d):(\d\d)$")
SOURCE_HASH = "9d79e5524794a2e6900a3aa9cb7d9152c5a3e8319c9a87c25494ba1eacee495f"

def minute(value):
    match = DATE.fullmatch(value)
    assert match, value
    d = datetime(*map(int, match.groups()))
    return d.toordinal() * 1440 + d.hour * 60 + d.minute


def records(path):
    decoder = json.JSONDecoder()
    with path.open() as source:
        first = source.read(1)
        assert first == "[", first
        buf = ""
        ended = False
        while not ended:
            buf = buf.lstrip(" \n\r\t,")
            if buf.startswith("]"):
                assert not buf[1:].strip() and not source.read().strip()
                return
            try:
                item, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                block = source.read(16 * 1024 * 1024)
                if not block:
                    ended = True
                buf += block
            else:
                yield item
                buf = buf[end:]
        raise ValueError("unclosed LongMemEval array")


def text_turns(chat):
    return [[turn["role"], turn["content"]] for turn in chat]


def text_signature(chat):
    return sha256(json.dumps(text_turns(chat), ensure_ascii=False, separators=(",", ":")).encode()).digest()


def answer_records(case):
    ids = case["haystack_session_ids"]
    gold = set(case["answer_session_ids"])
    out = {}
    for i, chat in zip(ids, case["haystack_sessions"], strict=True):
        if i in gold:
            out.setdefault(i, set()).add(text_signature(chat))
    return out


def prepare(a):
    assert digest(a.medium) == SOURCE_HASH
    identity = ("question", "question_type", "answer", "answer_session_ids", "question_date")
    small = {r["question_id"]: ({k: r[k] for k in identity}, answer_records(r)) for r in records(a.small)}
    assert len(small) == 500
    cases, sessions, counters = [], {}, Counter()
    a.out.mkdir(exist_ok=False)
    bank = a.out / "sessions.jsonl.gz"
    with bank.open("wb") as compressed, gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0,
                                                     filename="", compresslevel=4) as out:
        for row in records(a.medium):
            id_ = row["question_id"]
            held, evidence = small[id_]
            assert row["question_type"] == held["question_type"]
            align = {key: row[key] == held[key] for key in ("question", "answer")}
            align["evidence_ids"] = set(row["answer_session_ids"]) == set(held["answer_session_ids"])
            for key, agrees in align.items():
                counters["same_" + key + "_as_small"] += agrees
            gold = set(row["answer_session_ids"])
            assert gold <= set(row["haystack_session_ids"])
            observed = answer_records(row)
            same_text = observed == evidence
            counters["identical_gold_session_text_to_small"] += same_text
            counters["same_question_date_to_small"] += row["question_date"] == held["question_date"]
            id_list = row["haystack_session_ids"]
            ref, flags, dates = [], [], []
            for sid, chat, date in zip(id_list, row["haystack_sessions"], row["haystack_dates"], strict=True):
                fingerprint = text_signature(chat)
                if fingerprint not in sessions:
                    i = sessions[fingerprint] = len(sessions)
                    item = {"id": i, "turns": text_turns(chat)}
                    out.write((json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
                    counters["distinct_turns"] += len(chat)
                ref.append(sessions[fingerprint])
                flags.append([i for i, t in enumerate(chat) if t.get("has_answer", False)])
                dates.append(minute(date))
                if flags[-1] and sid not in gold:
                    counters["answer_turn_marks_outside_gold_sessions"] += len(flags[-1])
            counters["session_occurrences"] += len(id_list)
            counters["question_types/" + row["question_type"]] += 1
            counters["answer_turn_marks"] += sum(map(len, flags))
            counters["session_id_duplicate_cases"] += len(id_list) != len(set(id_list))
            cases.append({"id": id_, "kind": row["question_type"], "question": row["question"],
                          "time": minute(row["question_date"]), "gold_ids": sorted(gold), "slots": ref,
                          "author_turn_ids": flags, "dates": dates, "source_session_ids": id_list,
                          "gold_text_identical_to_small": same_text, "same_as_small": align})
    assert len(cases) == len({r["id"] for r in cases}) == len(small)
    counters["distinct_sessions"] = len(sessions)
    fixture = {"source_sha256": SOURCE_HASH, "small_sha256": digest(a.small), "session_bank_sha256": digest(bank),
               "summary": counters, "rows": cases}
    (a.out / "fixture.json").write_text(json.dumps(fixture, separators=(",", ":"), ensure_ascii=False) + "\n")
    public = {k: v for k, v in fixture.items() if k != "rows"}
    public["fixture_sha256"] = digest(a.out / "fixture.json")
    public["rows"] = [{k: v for k, v in r.items() if k != "question"} for r in cases]
    (a.out / "manifest.json.gz").write_bytes(gzip.compress(json.dumps(public, separators=(",", ":")).encode(), mtime=0))
    print(counters)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("small", "medium", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    prepare(a)


if __name__ == "__main__":
    main()
