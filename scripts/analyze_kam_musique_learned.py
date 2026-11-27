#!/usr/bin/env python3
"""Compare inference-generated MuSiQue queries with the frozen retrieval controls."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re

from analyze_kam_native_musique import COHORTS, compare, shared_evidence_clusters
from analyze_kam_shared_serving import unpack

REFERENCE = re.compile(r"#(\d+)\b")
SEEDS = 3


def verified(paths, baselines, manifest, gold, templates):
    chunks = sorted(map(unpack, paths), key=lambda x: x["shard"])
    assert [c["shard"] for c in chunks] == list(range(4))
    fst = chunks[0]
    assert all(c["weights_sha256"] == fst["weights_sha256"] and c["generation_sha256"] == fst["generation_sha256"]
               and c["corpus_sha256"] == manifest["corpus_sha256"] and c["fixture_sha256"] == manifest["fixture_sha256"]
               and c["device"] == "NVIDIA L4" for c in chunks)
    output = {}
    for part in chunks:
        base = baselines[part["baseline_sha256"]]
        assert base["shard"] == part["shard"] and base["fixture_sha256"] == manifest["fixture_sha256"]
        for raw, new, truth in zip(base["rows"], part["rows"], manifest["rows"][part["shard"]::4], strict=True):
            qid = raw["id"]
            assert qid == new["id"] == truth["id"]
            generated = templates[qid]
            assert len(generated["steps"]) == new["predicted_steps"] and generated["valid"] != new["fallback"]
            pool = new["pool"]
            assert len(pool) == len(set(pool)) == 128 and min(pool) >= 0 and max(pool) < manifest["documents"]
            if new["fallback"]:
                assert pool == raw["hybrid"]
            else:
                assert pool[:64] == raw["dense"][:64] and len(new["read_from"]) == new["predicted_steps"] - 1
            output[qid] = new | {"reference_signature_equal": (
                [REFERENCE.findall(s) for s in generated["steps"]] == [REFERENCE.findall(s) for s in gold[qid]])}
    return output


def summarize(cases, runs, base, bind, generations, annotated):
    result = {}
    for index, (rows, generation) in enumerate(zip(runs, generations, strict=True)):
        flags = {arm: [] for arm in ("learned", "hybrid", "annotated_top1", "resolved_oracle")}
        totals = {arm: Counter() for arm in flags}
        structural = Counter()
        for item in cases:
            id_ = item["id"]
            support = set(item["gold"])
            pools = {"learned": rows[id_]["pool"], "hybrid": base[id_]["hybrid"],
                     "annotated_top1": bind[id_]["arms"]["top1"]["pool"],
                     "resolved_oracle": annotated[id_]["resolved_answers"]}
            for arm, p in pools.items():
                q = len(support & set(p))
                flags[arm].append(q == len(support))
                totals[arm].update({"full_support": q == len(support), "support_paragraphs": q})
            row = rows[id_]
            structural.update({"valid": not row["fallback"], "all": 1,
                               "same_number_of_steps": row["predicted_steps"] == item["hops"],
                               "same_reference_signature": row["reference_signature_equal"],
                               "unknown_bindings": row["unknown_bindings"],
                               "fallback_full_support": row["fallback"] and flags["learned"][-1]})
            structural[f"prediction_depth/{row['predicted_steps']}"] += 1
        groups = shared_evidence_clusters(cases)
        result[str(index)] = {"weights_sha256": generation["weights_sha256"], "counts": totals, "structure": structural,
                              "paired_vs_hybrid": compare(flags["learned"], flags["hybrid"], groups),
                              "paired_vs_author_top1": compare(flags["learned"], flags["annotated_top1"], groups)}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for arm in ("manifest", "output", "author-templates", "train-manifest"):
        p.add_argument("--" + arm, type=Path, required=True)
    for arm in ("baseline", "binding", "oracle"):
        p.add_argument("--" + arm, type=Path, nargs=4, required=True)
    p.add_argument("--generate", type=Path, nargs=SEEDS, required=True)
    p.add_argument("--zero-shot", type=Path, required=True)
    p.add_argument("--learned", type=Path, nargs=4, action="append", required=True)
    p.add_argument("--fit", type=Path, nargs=SEEDS, required=True)
    a = p.parse_args()
    fixture = unpack(a.manifest)
    training = unpack(a.train_manifest)
    assert training["fixture_sha256"] == fixture["fixture_sha256"]
    logged = [json.loads(file.read_text()) for file in a.fit]
    assert all(x["data_sha256"] == training["data_sha256"] and x["fixture_sha256"] == training["fixture_sha256"]
               for x in logged)
    keys = [r["id"] for r in fixture["rows"]]
    gold = json.loads(a.author_templates.read_text())["templates"]
    base = {sha256(file.read_bytes()).hexdigest(): unpack(file) for file in a.baseline}
    generations = list(map(unpack, a.generate))
    assert len(a.learned) == len(generations) == SEEDS
    assert all(g["fixture_sha256"] == training["fixture_sha256"] and g["weights_sha256"]
               and [q["id"] for q in g["rows"]] == keys for g in generations)
    zero = unpack(a.zero_shot)
    assert zero["weights_sha256"] is None and [q["id"] for q in zero["rows"]] == keys
    call = lambda x: {r["id"]: r for data in x for r in data["rows"]}
    b = call(base.values())
    controls = [list(map(unpack, files)) for files in (a.binding, a.oracle)]
    assert all(c["fixture_sha256"] == fixture["fixture_sha256"] and c["corpus_sha256"] == fixture["corpus_sha256"]
               for group in controls for c in group)
    bind, oracle = map(call, controls)
    runs = []
    for ranked, g, path in zip(a.learned, generations, a.generate, strict=True):
        assert all(unpack(s)["weights_sha256"] == g["weights_sha256"]
                   and unpack(s)["generation_sha256"] == sha256(path.read_bytes()).hexdigest() for s in ranked)
        runs.append(verified(ranked, base, fixture, gold, call([g])))
    cases = fixture["rows"]
    panels = {"all": cases, **{str(i): [q for q in cases if q["hops"] == i] for i in COHORTS}}
    out = {"fixture_sha256": fixture["fixture_sha256"], "training": logged,
           "zero_shot_valid": sum(bool(q["valid"]) for q in zero["rows"]),
           "scopes": {key: summarize(p, runs, b, bind, generations, oracle) for key, p in panels.items()}}
    a.output.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
