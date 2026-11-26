#!/usr/bin/env python3
"""Score dated chat-session evidence retrieval on the official LongMemEval-S sample."""

import argparse
from collections import Counter
from datetime import datetime
import gzip
import json
from pathlib import Path
import re
from time import perf_counter

import torch

from audit_kam_natural_data import digest, rank
from run_kam_hotpot_independent import read, save
from run_kam_natural_feasibility import Encoder

DATE = re.compile(r"^(\d{4})/(\d\d)/(\d\d) \([^)]*\) (\d\d):(\d\d)$")
DEPTHS = (1, 3, 5, 10)
FUTURE_ID = "__future_echo__"
PAST_ID = "__past_echo__"


def minutes(value):
    match = DATE.fullmatch(value)
    if not match:
        raise ValueError(value)
    moment = datetime(*map(int, match.groups()))
    return moment.toordinal() * 1440 + moment.hour * 60 + moment.minute


def passage_chunks(block):
    content = block["content"].split()
    if not content:
        return
    prefix = f"{block['role']}: "
    for start in range(0, len(content), 96):
        yield prefix + " ".join(content[start:start + 128])
        if start + 128 >= len(content):
            break


def group_input(rows):
    cases, passages = [], []
    for r in rows:
        ids = r["haystack_session_ids"]
        dates = list(map(minutes, r["haystack_dates"]))
        now = minutes(r["question_date"])
        gold_ids = set(r["answer_session_ids"])
        assert len(ids) == len(dates) == len(r["haystack_sessions"])
        assert gold_ids <= set(ids) and {FUTURE_ID, PAST_ID}.isdisjoint(ids)
        offset = len(passages)
        counts, roles, session_texts = [], [], []
        for session in r["haystack_sessions"]:
            n = len(passages)
            for turn in session:
                chunks = list(passage_chunks(turn))
                passages.extend(chunks)
                roles.extend(turn["role"] == "user" for _ in chunks)
            counts.append(len(passages) - n)
            session_texts.append(" ".join(t["content"] for t in session))
        echo = f"The user raised the following question: {r['question']} This entry records the question without an answer."
        cases.append({"row": r, "dates": dates, "now": now, "sizes": counts, "roles": roles,
                      "start": offset, "end": len(passages), "echo": echo, "session_texts": session_texts})
        passages.append(echo)
    return cases, passages


def top(scores, ids, gold):
    order = scores.argsort(descending=True, stable=True).tolist()
    finite = scores.isfinite().tolist()
    unique = list(dict.fromkeys(ids[j] for j in order if finite[j]))
    pool = unique[:max(DEPTHS)]
    target = set(gold)
    return {"ranks": pool,
            "scores": {k: {"all": target <= set(pool[:k]), "recall": len(target & set(pool[:k])) / len(target)
                           if target else None} for k in DEPTHS}}


@torch.inference_mode()
def evaluate(a, rows):
    enc = Encoder(a.assets / "encoder", a.encode_batch)
    device = torch.device("cuda")
    cases, passages = group_input(rows)
    qv = enc([r["question"] for r in rows], "query")
    started = perf_counter()
    keys = enc(passages, "passage")
    encoding_seconds = perf_counter() - started
    records = []
    for i, item in enumerate(cases):
        r = item["row"]
        dates = torch.tensor(item["dates"], device=device)
        query_time = item["now"]
        ids, sizes = r["haystack_session_ids"], item["sizes"]
        base = torch.tensor(sizes, device=device)
        mapping = torch.repeat_interleave(torch.arange(len(ids), device=device), base)
        user = torch.tensor(item["roles"], device=device)
        scores = qv[i] @ keys[item["start"]:item["end"]].T
        echo_affinity = qv[i] @ keys[item["end"]]
        assert len(scores) == len(mapping) == len(user)
        highest = torch.full((len(ids),), -torch.inf, device=device).scatter_reduce_(0, mapping, scores, reduce="amax")
        user_best = torch.full_like(highest, -torch.inf).scatter_reduce_(
            0, mapping, scores.masked_fill(~user, -torch.inf), reduce="amax")
        mean = torch.zeros_like(highest).scatter_add_(0, mapping, scores) / base.clamp_min(1)
        mean.masked_fill_(base == 0, -torch.inf)
        recency = 0.025 * ((dates - query_time).clamp_max(0).float() / (30 * 1440)).exp()
        gold = r["answer_session_ids"]
        arms = {name: top(values, ids, gold) for name, values in
                (("dense_max", highest), ("dense_mean", mean), ("user_only", user_best),
                 ("dense_recent", highest + recency), ("recent_only", dates),
                 ("dense_asof_original", highest.masked_fill(dates > query_time, -torch.inf)))}
        lexical_order = rank(r["question"], item["session_texts"], max(DEPTHS))
        lexical_scores = torch.full_like(highest, -torch.inf)
        lexical_scores[lexical_order] = -torch.arange(len(lexical_order), dtype=torch.float32, device=device)
        arms["bm25_session"] = top(lexical_scores, ids, gold)
        noisy = torch.cat((highest, echo_affinity.reshape(1)))
        arms["future_echo_unfiltered"] = top(noisy, ids + [FUTURE_ID], gold)
        arms["future_echo_asof"] = top(noisy.masked_fill(
            torch.cat((dates, dates.new_tensor([query_time + 7 * 1440]))) > query_time, -torch.inf),
            ids + [FUTURE_ID], gold)
        arms["past_echo_asof"] = top(noisy.masked_fill(
            torch.cat((dates, dates.new_tensor([query_time - 7 * 1440]))) > query_time, -torch.inf),
            ids + [PAST_ID], gold)
        assert arms["dense_asof_original"] == arms["future_echo_asof"]
        gold_display = {id_: max(d for j,d in zip(ids, item["dates"], strict=True) if j == id_) for id_ in set(gold)}
        records.append({"id": r["question_id"], "kind": r["question_type"], "gold": gold,
                        "abstention_item": r["question_id"].endswith("_abs"),
                        "latest_gold": max(gold_display, key=gold_display.get, default=None),
                        "dated_post_cutoff_sessions": sum(d > query_time for d in item["dates"]),
                        "unavailable_gold_under_cutoff": sorted(set(gold) - {j for j, d in zip(ids, item["dates"], strict=True)
                                                                          if d <= query_time}),
                        "n_sessions": len(ids), "n_chunks": len(scores), "arms": arms})
    return records, encoding_seconds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--encode-batch", type=int, default=128)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    with gzip.open(a.source, "rb") as infile:
        raw = infile.read()
    from hashlib import sha256
    assert sha256(raw).hexdigest() == a.source_sha256
    full = json.loads(raw)
    assert len(full) == len({r["question_id"] for r in full})
    cases = full[a.shard::a.shards]
    rows, duration = evaluate(a, cases)
    save(a.output, {"source_sha256": a.source_sha256, "gzip_sha256": digest(a.source), "n_source": len(full),
                    "models": read(a.assets / "revisions.json"), "device": torch.cuda.get_device_name(),
                    "torch": str(torch.__version__), "counts": dict(Counter(r["kind"] for r in rows)),
                    "encoding_seconds": duration, "elapsed_seconds": perf_counter() - start, "rows": rows})


if __name__ == "__main__":
    main()
