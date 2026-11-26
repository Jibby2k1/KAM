#!/usr/bin/env python3
"""Inject matched irrelevant cross-references into held-out support pools."""

import argparse
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest, words
from run_kam_fullwiki_support_links import ContextScorer, relation_features
from run_kam_hotpot_head_to_head import route
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import Reader, pair_selection, prepared
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, candidate_texts, measure_answer

SCENARIOS = ("clean", "sham_text", "linked_text", "graph_only")


def modified(case, donor, title, template="See also {title}."):
    docs = list(case["context"])
    original_title, evidence = docs[donor]
    docs[donor] = original_title, [template.format(title=title), *evidence]
    return {**case, "context": docs}


def challenge(case, graph, affinities):
    gold = set(case["gold"])
    ordering = affinities.argsort(descending=True, stable=True).tolist()
    for donor in ordering:
        if donor in gold:
            continue
        for target in ordering:
            if target not in gold or graph[donor, target] or graph[target, donor]:
                continue
            title = case["context"][target][0]
            tokens = words(title)
            if len(" ".join(tokens)) >= 5:
                return donor, target, title, " ".join(f"catalogue{i+917}" for i in range(len(tokens)))
    return None


@torch.inference_mode()
def evaluate(a, data, cases):
    cfg = read(a.config)
    enc, reader = Encoder(a.assets / "encoder", 64), Reader(a.assets / "reader")
    adapter, original = restore_checkpoint(a.baseline, a.config, cfg, None, None,
                                           weights_sha256=a.baseline_sha256)
    fitted = torch.load(a.heads, weights_only=True, map_location="cpu")
    assert fitted["provenance"]["weights_sha256"] == a.baseline_sha256
    contextual = ContextScorer().cuda().eval()
    contextual.load_state_dict(fitted["heads"]["context_only"])
    alternatives = {}
    if a.robust_heads:
        extra = torch.load(a.robust_heads, weights_only=True, map_location="cpu")
        assert extra["baseline_sha256"] == a.baseline_sha256
        assert extra["config_sha256"] == digest(a.config)
        for name, item in extra["arms"].items():
            head = ContextScorer().cuda().eval()
            head.load_state_dict(item["weights"])
            alternatives[name] = head, torch.tensor(item["mask"], device="cuda")
    queries = enc([c["question"] for c in cases], "query")
    docs = [candidate_texts(c) for c in cases]
    keys = enc([text for snippets in docs for text in snippets], "passage").split(list(map(len, docs)))
    chunks, graph_checks = [], []
    pair_indices = {}
    for i, (case, bank) in enumerate(zip(cases, keys, strict=True)):
        depth = len(bank)
        if depth not in pair_indices:
            pair_indices[depth] = torch.triu_indices(depth, depth, 1)
        pairs = pair_indices[depth]
        graph, x = relation_features(case, pairs)
        q = adapter(queries[i:i+1])
        order = (q @ bank.T)[0]
        description = challenge(case, graph, order)
        if description is None:
            continue
        donor, target, title, fake = description
        sham = modified(case, donor, fake, a.injection_template)
        injected = modified(case, donor, title, a.injection_template)
        sham_graph, sham_x = relation_features(sham, pairs)
        linked_graph, linked_x = relation_features(injected, pairs)
        if sham_graph[donor, target] or not linked_graph[donor, target]:
            continue
        assert len(words(sham["context"][donor][1][0])) == len(words(injected["context"][donor][1][0]))
        assert all(sham["context"][g] == injected["context"][g] == case["context"][g] for g in case["gold"])
        only_graph = graph.clone()
        only_graph[donor, target] = True
        chunks.append((i, case, donor, target, bank, q, pairs,
                       ((case, graph, x), (sham, sham_graph, sham_x), (injected, linked_graph, linked_x),
                        (case, only_graph, x))))
        graph_checks.append({"changed_directed_links": int((sham_graph != linked_graph).sum()),
                             "sham_changes": int((graph != sham_graph).sum()),
                             "gold_both_linked": bool(graph[case["gold"][0], case["gold"][1]] |
                                                      graph[case["gold"][1], case["gold"][0]])})
    changed_docs = [candidate_texts(variant[0])[donor] for _, _, donor, _, _, _, _, variants in chunks
                    for variant in variants[1:3]]
    changed = enc(changed_docs, "passage")
    assert len(changed) == 2 * len(chunks)
    records, requests = [], []
    for j, (i, case, donor, target, bank, query, pairs, variants) in enumerate(chunks):
        depth = len(bank)
        gpu_pairs = pairs.cuda()
        out = {}
        pure_sequential = None
        for s, (mode, (example, graph, context)) in enumerate(zip(SCENARIOS, variants, strict=True)):
            if mode in ("sham_text", "linked_text"):
                bank_mode = bank.clone()
                bank_mode[donor] = changed[2 * j + s - 1]
            else:
                bank_mode = bank
            keys_mode = bank_mode.unsqueeze(0)
            mentions = graph.cuda().unsqueeze(0)
            base, x, affinity = prepared(query, keys_mode, mentions, gpu_pairs)
            x_all = torch.cat((x[0], context.cuda()), -1)
            winner = int(contextual(base[0], x_all).argmax())
            left, right = pairs[:, winner].tolist()
            selected = {
                "learned_single_shot": affinity[0].argsort(descending=True, stable=True)[:2].tolist(),
                "one_pass_pair": pair_selection(query, keys_mode, mentions, original, gpu_pairs)[0].tolist(),
                "context_only": [left, right] if affinity[0, left] >= affinity[0, right] else [right, left],
            }
            for name, (head, active) in alternatives.items():
                left, right = pairs[:, head(base[0], x_all * active).argmax().item()].tolist()
                selected[name] = [left, right] if affinity[0, left] >= affinity[0, right] else [right, left]
            if mode == "graph_only":
                selected["learned_sequential"] = pure_sequential
            else:
                seq, _ = route(enc, adapter, [example], queries[i:i+1], keys_mode, include_frozen=False)
                selected["learned_sequential"] = seq["learned_sequential"][0]
                if mode == "clean":
                    pure_sequential = seq["learned_sequential"][0]
            choices = {}
            for arm, selection in selected.items():
                entry = {"selection": selection, "both_support": set(selection) == set(case["gold"]),
                         "injected_pair": set(selection) == {donor, target},
                         "contains_donor": donor in selection}
                choices[arm] = entry
                requests.append((reader.prompt(example, selection), entry, [case["answer"], *case.get("aliases", [])]))
            out[mode] = choices
        records.append({"id": case["id"], "gold": case["gold"], "donor": donor, "target": target,
                        "scenarios": out, **graph_checks[j]})
    prompts = list(dict.fromkeys(p for p, _, _ in requests))
    predictions = dict(zip(prompts, reader.answers(prompts), strict=True))
    for prompt, entry, truths in requests:
        entry["em"] = max(measure_answer(predictions[prompt], answer)["em"] for answer in truths)
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for item in ("assets", "config", "baseline", "heads", "fixture", "output"):
        p.add_argument("--" + item, required=True, type=Path)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--robust-heads", type=Path)
    p.add_argument("--injection-template", default="See also {title}.")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    data = read(a.fixture)
    cases = data["rows"][a.shard::a.shards]
    rows = evaluate(a, data, cases)
    save(a.output, {"dataset": data["dataset"], "fixture_sha256": digest(a.fixture),
                    "robust_heads_sha256": digest(a.robust_heads) if a.robust_heads else None,
                    "injection_template": a.injection_template,
                    "heads_sha256": digest(a.heads), "baseline_sha256": a.baseline_sha256,
                    "eligible": len(rows), "total": len(cases), "rows": rows,
                    "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
                    "elapsed_seconds": perf_counter() - start})


if __name__ == "__main__":
    main()
