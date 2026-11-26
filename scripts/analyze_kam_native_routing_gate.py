#!/usr/bin/env python3
"""Seal ranked pools, then fit and evaluate inference-visible routing policies."""

import argparse
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from analyze_kam_coverage_gate import choose, fit
from analyze_kam_sentence_verifier import paired

LINK_FIELDS = ("broad_pair_link", "linked_pair_link", "pool_link_fraction")
ARMS = ("broad", "linked", "frozen_context")
POLICIES = ARMS + ("observed_link_rule", "link_only_gate", "learned_gate", "oracle_choice")
MEASURES = ("support", "answer_exact", "joint_exact")


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()))


def write(path, value):
    path.write_bytes(gzip.compress(json.dumps(value, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


def load_shards(directory, prefix):
    result = [read(p) for p in sorted(directory.glob(prefix + "-*.json.gz"))]
    assert result
    assert sorted(s["shard"] for s in result) == list(range(result[0]["shards"]))
    return result


def seal(shards, fixture):
    fixture_sha = digest(fixture)
    assert all(s["fixture_sha256"] == fixture_sha and s["index_sha256"] == shards[0]["index_sha256"] for s in shards)
    data = json.loads(fixture.read_bytes())
    ids = {r["id"] for rows in data["rows"].values() for r in rows}
    rows = {r["id"]: r["row_ids"] for s in shards for r in s["rows"]}
    assert ids == rows.keys() and len(rows) == sum(len(s["rows"]) for s in shards)
    assert all(len(p) == len(set(p)) == 64 and all(isinstance(x, int) and x >= 0 for x in p) for p in rows.values())
    return {"fixture_sha256": fixture_sha, "index_sha256": shards[0]["index_sha256"],
            "gpu": sorted({s["gpu"] for s in shards}), "rows": dict(sorted(rows.items()))}


def train_gate(rows, indices):
    observations = []
    for r in rows:
        pair = {"features": [r["features"][i] for i in indices], "full": r["arms"]["linked"]["support"],
                "without": r["arms"]["broad"]["support"], "same": r["agree"]}
        observations.append({"variants": {"clean": pair}})
    return fit(observations, ("clean",))


def decisions(rows, parameters, link_indices):
    result = []
    for row in rows:
        x = row["features"]
        learned = "linked" if choose({"features": x}, parameters["learned_gate"]) == "full" else "broad"
        link_gate = "linked" if choose({"features": [x[i] for i in link_indices]}, parameters["link_only_gate"]) == "full" else "broad"
        result.append({**{k: k for k in ARMS},
                       "observed_link_rule": "linked" if x[0] else "broad",
                       "learned_gate": learned, "link_only_gate": link_gate})
    return result


def score(rows, routes):
    picks = [routes[r["id"]] for r in rows]
    by_policy = {name: {m: [] for m in MEASURES} for name in POLICIES}
    for row, route in zip(rows, picks, strict=True):
        def score_arm(name):
            hit = row["arms"][name]["support"]
            answer = row["arms"][name]["exact"]
            return hit, answer, hit and answer

        for policy in POLICIES:
            if policy == "oracle_choice":
                results = [a or b for a, b in zip(score_arm("broad"), score_arm("linked"), strict=True)]
            else:
                results = score_arm(route[policy])
            for m, v in zip(MEASURES, results, strict=True):
                by_policy[policy][m].append(v)
    return {"n": len(rows), "same_pair": sum(r["agree"] for r in rows),
            "linked_routes": {p: sum(r[p] == "linked" for r in picks)
                              for p in ("observed_link_rule", "link_only_gate", "learned_gate")},
            "correct": {p: {m: sum(x) for m, x in by_policy[p].items()} for p in POLICIES},
            "gate_vs": {p: {m: paired(by_policy["learned_gate"][m], by_policy[p][m]) for m in MEASURES}
                        for p in ("broad", "linked", "frozen_context", "observed_link_rule", "link_only_gate")}}


def summarize(shards, pools, fixture_path, fitted_path):
    sealed = read(pools)
    fields = ("fixture_sha256", "index_sha256", "pools_sha256", "fitted_sha256", "heads_sha256", "ablations_sha256", "features")
    reference = shards[0]
    assert all(all(s[k] == reference[k] for k in fields) for s in shards)
    fixture = gzip.decompress(fixture_path.read_bytes()) if fixture_path.suffix == ".gz" else fixture_path.read_bytes()
    assert reference["fixture_sha256"] == sealed["fixture_sha256"] == sha256(fixture).hexdigest()
    assert reference["index_sha256"] == sealed["index_sha256"]
    assert reference["pools_sha256"] == digest(pools)
    assert reference["fitted_sha256"] == digest(fitted_path)
    case_rows = json.loads(fixture)["rows"]
    partitions = {r["id"]: split for split, cases in case_rows.items() for r in cases}
    gold_ids = {r["id"]: set(r["gold_ids"]) for cases in case_rows.values() for r in cases}
    rows = sorted([r for s in shards for r in s["rows"]], key=lambda r: r["id"])
    assert {r["id"]: r["split"] for r in rows} == partitions and len(rows) == len(partitions)
    assert len(rows) == len(sealed["rows"])
    for r in rows:
        ordered = sealed["rows"][r["id"]]
        assert r["pool_hash"] == sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest()
        assert len(r["features"]) == len(reference["features"])
        for v in r["arms"].values():
            assert len(v["pair"]) == len(set(v["pair"])) == 2 and all(0 <= j < 64 for j in v["pair"])
            assert v["support"] == ({ordered[j] for j in v["pair"]} == gold_ids[r["id"]])
        if r["split"] == "test":
            assert all("exact" in v for v in r["arms"].values())
            assert not r["agree"] or r["arms"]["broad"] == r["arms"]["linked"]
        else:
            assert all("exact" not in v for v in r["arms"].values())
    fit_rows = [r for r in rows if r["split"] == "training"]
    eval_rows = [r for r in rows if r["split"] == "test"]
    assert (len(fit_rows), len(eval_rows)) == (1536, 1024)
    link_indices = [reference["features"].index(k) for k in LINK_FIELDS]
    params = {"learned_gate": train_gate(fit_rows, list(range(len(reference["features"])))),
              "link_only_gate": train_gate(fit_rows, link_indices)}
    by_id = dict(zip([r["id"] for r in eval_rows], decisions(eval_rows, params, link_indices), strict=True))
    panels = {
        "all": eval_rows,
        "gold_in_pool": [r for r in eval_rows if r["gold_in_pool"] == 2],
        "natural_linked_distractor": [r for r in eval_rows if r["gold_linked"] is True and r["pool_link_count"] > 1],
        "gold_unlinked": [r for r in eval_rows if r["gold_linked"] is False],
        "missing_gold": [r for r in eval_rows if r["gold_in_pool"] < 2],
        "reranker_disagreement": [r for r in eval_rows if not r["agree"]],
    }
    label_counts = Counter("linked_only" if r["arms"]["linked"]["support"] else "broad_only" for r in fit_rows
                           if r["arms"]["linked"]["support"] != r["arms"]["broad"]["support"])
    report = {**{k: reference[k] for k in fields}, "freeze_gpu": sealed["gpu"], "score_gpu": sorted({s["gpu"] for s in shards}),
              "training": {"n": len(fit_rows), "same_pair": sum(r["agree"] for r in fit_rows),
                           "supervision": dict(sorted(label_counts.items()))},
              "tests": {label: score(subset, by_id) for label, subset in panels.items()}}
    return report, params


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "analyze"))
    parser.add_argument("input", type=Path)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--pools", type=Path)
    parser.add_argument("--fitted", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-replay", type=Path)
    parser.add_argument("--write-fit", type=Path)
    args = parser.parse_args()
    if args.mode == "seal":
        write(args.output, seal(load_shards(args.input, "freeze"), args.fixture))
        return
    if args.pools is None or args.fitted is None:
        parser.error("Analysis requires the sealed pools and the frozen native reranker weights")
    shards = load_shards(args.input, "score") if args.input.is_dir() else read(args.input)
    report, parameters = summarize(shards, args.pools, args.fixture, args.fitted)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.write_fit:
        args.write_fit.write_text(json.dumps(parameters, indent=2, allow_nan=False) + "\n")
    if args.write_replay:
        write(args.write_replay, shards)


if __name__ == "__main__":
    main()
