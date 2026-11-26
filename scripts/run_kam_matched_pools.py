#!/usr/bin/env python3
"""Separate source-domain effects from candidate count and distractor hardness."""

import argparse
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from audit_kam_natural_data import digest, stable, words
from run_kam_fullwiki_support_links import ContextScorer, relation_features
from run_kam_hotpot_head_to_head import route
from run_kam_hotpot_independent import read, save
from run_kam_hotpot_pool_scaling import pair_selection, prepared
from run_kam_link_noise import modified
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, candidate_texts

CONDITIONS = (("supplied", 10), ("unrelated", 20), ("hard", 20), ("unrelated", 64), ("hard", 64))
GATE_COLUMNS = ("full_link", "drop_link", "full_margin", "drop_margin", "full_preference",
                "drop_preference", "share_document", "full_affinity", "drop_affinity", "link_fraction")


def probes(case, keys, q, pairs, enc, full, without, keep, gold, split):
    """Score paired text interventions and record features available at serving time."""
    cpu_pairs = pairs.cpu()
    graph, lexical = relation_features(case, cpu_pairs)
    target_set = set(gold)
    _, _, affinity = prepared(q, keys, graph.unsqueeze(0).cuda(), pairs)
    order = affinity[0].argsort(descending=True, stable=True).tolist()
    challenge = next(((d, t, case["context"][t][0]) for d in order if d not in target_set
                      for t in order if t in target_set and not (graph[d,t] or graph[t,d])
                      and len(" ".join(words(case["context"][t][0]))) >= 5), None)
    variants = [("clean", graph, lexical, keys)]
    if challenge is not None:
        donor, target, title = challenge
        fake = " ".join(f"registry{i+724}" for i, _ in enumerate(words(title)))
        template = "Additional references include {title}." if split.endswith("-fit") else "Related reading: {title}."
        sham, linked = [modified(case, donor, label, template) for label in (fake, title)]
        edits = [relation_features(variant, cpu_pairs) for variant in (sham, linked)]
        if not edits[0][0][donor,target] and edits[1][0][donor,target]:
            assert all(sham["context"][g] == linked["context"][g] == case["context"][g] for g in gold)
            assert len(words(sham["context"][donor][1][0])) == len(words(linked["context"][donor][1][0]))
            vectors = enc([candidate_texts({"context": [v["context"][donor]]})[0] for v in (sham, linked)], "passage")
            for name, (g, x), vector in zip(("sham", "linked"), edits, vectors, strict=True):
                copy = keys.clone()
                copy[0, donor] = vector
                variants.append((name, g, x, copy))
    output = {}
    for name, g, x, bank in variants:
        base, common, affinities = prepared(q, bank, g.unsqueeze(0).cuda(), pairs)
        features = torch.cat((common[0], x.cuda()), -1)
        scores = torch.stack((full(base[0], features), without(base[0], features * keep)))
        scores = (scores - scores.mean(1, keepdim=True)) / scores.std(1, keepdim=True).clamp_min(1e-6)
        winners = scores.argsort(dim=1, descending=True, stable=True)[:, :2]
        ix, iy = winners[:, 0].tolist()
        coords = pairs[:, [ix, iy]].T.tolist()
        edges = g | g.T
        values = scores.gather(1, winners)
        margins = (values[:, 0] - values[:, 1]).tolist()
        a, b = coords
        signals = [int(edges[a[0], a[1]]), int(edges[b[0], b[1]]), *margins,
                   float(scores[0, ix] - scores[0, iy]), float(scores[1, iy] - scores[1, ix]),
                   int(bool(set(a) & set(b))), float(affinities[0, a].mean()), float(affinities[0, b].mean()),
                   float(edges.triu(1).sum()) / pairs.shape[1]]
        output[name] = {"features": signals, "full": set(a) == target_set, "without": set(b) == target_set,
                        "same": ix == iy}
    return output


def donors_by_quality(q, donor_keys, donors, case):
    semantic = (q @ donor_keys.T)[0]
    order = semantic.argsort(descending=True, stable=True).tolist()
    answer = words(case["answer"])
    blocked = f" {' '.join(answer)} " if len(answer) >= 3 else None
    allowed = lambda idx: blocked is None or blocked not in donors[idx]
    hard = [idx for idx in order[:1024] if allowed(idx)][:54]
    rng = Random(int(stable("kam-matched-donors-v1:" + case["id"])[:16], 16))
    random_choice = [idx for idx in rng.sample(range(len(donors)), 512) if allowed(idx)][:54]
    assert len(hard) == len(random_choice) == 54
    return {"hard": hard, "unrelated": random_choice, "supplied": []}, semantic


@torch.inference_mode()
def evaluate(a, fixture, rows):
    enc = Encoder(a.assets / "encoder", 128)
    cfg = read(a.config)
    adapter, original = restore_checkpoint(a.baseline, a.config, cfg, None, None,
                                           weights_sha256=a.baseline_sha256)
    fitted = torch.load(a.heads, map_location="cpu", weights_only=True)
    no_coverage = torch.load(a.ablations, map_location="cpu", weights_only=True)
    assert fitted["provenance"]["models"] == read(a.assets / "revisions.json")
    assert no_coverage["reference_heads_sha256"] == digest(a.heads)
    context = ContextScorer().cuda().eval()
    context.load_state_dict(fitted["heads"]["context_only"])
    remove_coverage = ContextScorer().cuda().eval()
    control = no_coverage["arms"]["refit_no_coverage"]
    remove_coverage.load_state_dict(control["weights"])
    keep = torch.tensor(control["mask"], device="cuda")
    donors = [(doc["title"], [doc["text"]]) for doc in fixture["donors"]]
    parsed = [f" {' '.join(words(doc['text']))} " for doc in fixture["donors"]]
    donor_keys = enc(candidate_texts({"context": donors}), "passage")
    qbank = enc([r["question"] for r in rows], "query")
    context_bank = enc([t for r in rows for t in candidate_texts(r)], "passage").reshape(len(rows), 10, -1)
    conditions = (("hard", 64),) if a.probe_splits else CONDITIONS
    pair_indices = {depth: torch.triu_indices(depth, depth, 1) for _, depth in conditions}
    splits = read(a.probe_splits) if a.probe_splits else None
    gpu_pairs = {depth: pairs.cuda() for depth, pairs in pair_indices.items()}
    results = []
    for i, row in enumerate(rows):
        query, supplied = qbank[i:i+1], context_bank[i]
        adapted = adapter(query)
        proposals, affinities = donors_by_quality(query, donor_keys, parsed, row)
        data = {}
        for family, depth in conditions:
            links = proposals[family][:depth - 10]
            assert len(links) == depth - 10
            items = [-(j + 1) for j in range(10)] + links
            items.sort(key=lambda j: stable(f"kam-matched-order-v1:{row['id']}:{j}"))
            case = {"question": row["question"], "context": [row["context"][-x - 1] if x < 0 else donors[x] for x in items]}
            keys = torch.stack([supplied[-x - 1] if x < 0 else donor_keys[x] for x in items]).unsqueeze(0)
            targets = {j for j, x in enumerate(items) if x < 0 and (-x - 1) in row["gold"]}
            assert len(targets) == 2
            if splits is not None:
                split = splits[row["id"]]
                outcome = probes(case, keys, adapted, gpu_pairs[depth], enc, context, remove_coverage, keep, targets, split)
                results.append({"id": row["id"], "split": split, "variants": outcome})
                continue
            graph, new = relation_features(case, pair_indices[depth])
            g = graph.unsqueeze(0).cuda()
            base, x, aff = prepared(adapted, keys, g, gpu_pairs[depth])
            features = torch.cat((x[0], new.cuda()), -1)
            seq, _ = route(enc, adapter, [case], query, keys, include_frozen=False)
            choices = {a: picks[0] for a, picks in seq.items()}
            choices["one_pass_pair"] = pair_selection(adapted, keys, g, original, gpu_pairs[depth])[0].tolist()
            for arm, logits in (("context_only", context(base[0], features)),
                                ("no_document_coverage", remove_coverage(base[0], features * keep))):
                choices[arm] = pair_indices[depth][:, logits.argmax().item()].tolist()
            s1, s2 = sorted(targets)
            data[f"{family}:{depth}"] = {
                "gold_pairs_linked": bool(graph[s1, s2] | graph[s2, s1]),
                "n_linked_pairs": int((graph | graph.T).triu(1).sum()),
                "donor_relevance": float(affinities[links].mean()) if links else None,
                "pool": items,
                "arms": {arm: {"selected": selected, "complete": set(selected) == targets,
                               "outside_candidate_count": sum(items[j] >= 0 for j in selected)}
                         for arm, selected in choices.items()},
            }
        if splits is None:
            results.append({"id": row["id"], "kind": row["kind"], "gold_base": row["gold"], "conditions": data})
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "fixture", "config", "baseline", "heads", "ablations", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--dataset", choices=("twowiki", "musique"), required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--probe-splits", type=Path)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    fixture = read(a.fixture)
    cases = fixture["rows"][a.dataset][a.shard::a.shards]
    rows = evaluate(a, fixture, cases)
    save(a.output, {"dataset": a.dataset, "fixture_sha256": digest(a.fixture), "heads_sha256": digest(a.heads),
                    "ablations_sha256": digest(a.ablations), "donor_count": len(fixture["donors"]),
                    "device": torch.cuda.get_device_name(), "elapsed_seconds": perf_counter() - start, "rows": rows}
                    | ({"gate_columns": GATE_COLUMNS, "split_sha256": digest(a.probe_splits)} if a.probe_splits else {}))


if __name__ == "__main__":
    main()
