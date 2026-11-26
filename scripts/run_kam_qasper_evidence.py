#!/usr/bin/env python3
"""Measure pair transfer and section-matched noise on research-paper QA."""

import argparse
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_fullwiki_support_links import ContextScorer, relation_features
from run_kam_hotpot_head_to_head import route
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import Reader, pair_selection, prepared
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, candidate_texts, measure_answer


def evidence_metrics(picked, gold_sets):
    proposed = set(picked)
    hits = [len(proposed & set(gold)) for gold in gold_sets]
    return {
        "complete": any(n == len(gold) for n, gold in zip(hits, gold_sets, strict=True)),
        "evidence_f1": max(2 * n / (len(proposed) + len(gold)) for n, gold in zip(hits, gold_sets, strict=True)),
    }


@torch.inference_mode()
def evaluate(a, rows):
    enc, reader = Encoder(a.assets / "encoder", 64), Reader(a.assets / "reader")
    cfg = read(a.config)
    adapter, original = restore_checkpoint(a.baseline, a.config, cfg, None, None,
                                           weights_sha256=a.baseline_sha256)
    frozen = torch.load(a.heads, weights_only=True, map_location="cpu")
    assert frozen["provenance"]["weights_sha256"] == a.baseline_sha256
    assert frozen["provenance"]["models"] == read(a.assets / "revisions.json")
    context_head = ContextScorer().cuda().eval()
    context_head.load_state_dict(frozen["heads"]["context_only"])
    full_cases = [{"question": r["question"], "context": r["context"] + r["noise"]} for r in rows]
    document_sizes = [len(c["context"]) for c in full_cases]
    passages = enc([s for case in full_cases for s in candidate_texts(case)], "passage").split(document_sizes)
    queries = enc([r["question"] for r in rows], "query")
    requests, results, pair_indices = [], [], {}
    for i, (row, case, bank) in enumerate(zip(rows, full_cases, passages, strict=True)):
        key_scores = queries[i] @ bank.T
        adapted = adapter(queries[i:i+1])
        n = len(row["context"])
        gold_sets = row["gold_sets"]
        result = {"id": row["id"], "paper": row["paper"], "kind": row["kind"],
                  "gold_sets": gold_sets, "n_clean": n, "scenarios": {}}
        for scenario, size in (("clean", n), ("noise", len(bank))):
            ranks = key_scores[:size].argsort(descending=True, stable=True)[:64].tolist()
            pool = {"question": case["question"], "context": [case["context"][r] for r in ranks]}
            keys = bank[ranks].unsqueeze(0)
            depth = len(ranks)
            if depth not in pair_indices:
                cpu = torch.triu_indices(depth, depth, 1)
                pair_indices[depth] = cpu, cpu.cuda()
            pairs, gpu_pairs = pair_indices[depth]
            graph, lexical = relation_features(pool, pairs)
            mentions = graph.cuda().unsqueeze(0)
            base, features, affinities = prepared(adapted, keys, mentions, gpu_pairs)
            logits = context_head(base[0], torch.cat((features[0], lexical.cuda()), -1))
            left, right = pairs[:, logits.argmax().item()].tolist()
            context_selected = [left, right] if affinities[0, left] >= affinities[0, right] else [right, left]
            seq, _ = route(enc, adapter, [pool], queries[i:i+1], keys, include_frozen=False)
            choices = {"dense_top1": [0],
                       "dense_top2": [0, 1],
                       "learned_sequential": seq["learned_sequential"][0],
                       "one_pass_pair": pair_selection(adapted, keys, mentions, original, gpu_pairs)[0].tolist(),
                       "context_pair": context_selected}
            arms = {}
            for arm, local in choices.items():
                choice = [ranks[k] for k in local]
                metrics = {**evidence_metrics(choice, gold_sets), "picked": choice,
                           "noise_selected": sum(j >= n for j in choice)}
                arms[arm] = metrics
                requests.append((reader.prompt(pool, local), metrics, row["answers"]))
            result["scenarios"][scenario] = {"ranks": ranks, "pool_can_cover": evidence_metrics(ranks, gold_sets)["complete"],
                                             "arms": arms}
        shortest = min(gold_sets, key=lambda group: (len(group), group))
        oracle = shortest[:2]
        result["oracle"] = evidence_metrics(oracle, gold_sets)
        requests.append((reader.prompt(case, oracle), result["oracle"], row["answers"]))
        results.append(result)
    prompts = list(dict.fromkeys(p for p, _, _ in requests))
    outputs = dict(zip(prompts, reader.answers(prompts), strict=True))
    for prompt, metric, truths in requests:
        scores = [measure_answer(outputs[prompt], truth) for truth in truths]
        metric.update(answer_em=max(d["em"] for d in scores), answer_f1=max(d["f1"] for d in scores))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "fixture", "config", "baseline", "heads", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    fixture = read(a.fixture)
    rows = evaluate(a, fixture["rows"][a.shard::a.shards])
    save(a.output, {"fixture_sha256": digest(a.fixture), "source_sha256": fixture["source_sha256"],
                    "heads_sha256": digest(a.heads), "gpu": torch.cuda.get_device_name(),
                    "torch": str(torch.__version__), "rows": rows, "elapsed_seconds": perf_counter() - start})


if __name__ == "__main__":
    main()
