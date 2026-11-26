#!/usr/bin/env python3
"""Fit listwise sentence rerankers on earlier 2Wiki cases and replay held-out results."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

SCENARIOS = ("clean", "sham", "linked")
CONTEXTUAL = ("contextual_sham", "contextual_linked")
MODELS = {
    "base_reranker": (5, SCENARIOS),
    "lexical_sentence": (11, SCENARIOS),
    "semantic_sentence": (16, SCENARIOS),
    "semantic_clean_fit": (16, ("clean",)),
}
SENTENCE_FEATURE_GROUPS = {
    "relevance_without_mentions": ("weak_max_lexical", "max_lexical", "joint_lexical",
                                   "weak_first_sentence_lexical", "weak_max_semantic",
                                   "max_semantic", "joint_top_semantic_lexical"),
    "mention_only": ("mention_lexical", "mention_same_sentence_count", "mention_semantic", "mention_margin"),
    "mention_without_margin": ("mention_lexical", "mention_same_sentence_count", "mention_semantic"),
}


def fit(rows, dimensions, scenarios, penalty=.025):
    features, labels, weights = [], [], []
    for row in rows:
        variants = [row["variants"][v] for v in scenarios if v in row["variants"] and row["variants"][v]["gold"] >= 0]
        for item in variants:
            features.append(np.asarray(item["features"], dtype=np.float64)[:, :dimensions])
            labels.append(item["gold"])
            weights.append(1 / len(variants))
    labels = np.asarray(labels)
    weights = np.asarray(weights) / sum(weights)
    flat = np.concatenate(features)
    center, scale = flat.mean(0), flat.std(0).clip(1e-6)
    batch = np.zeros((len(features), max(map(len, features)), dimensions))
    valid = np.zeros(batch.shape[:2], dtype=bool)
    for i, x in enumerate(features):
        batch[i, :len(x)] = (x - center) / scale
        valid[i, :len(x)] = True
    index = np.arange(len(labels))

    def loss(w):
        logits = batch @ w
        logits[~valid] = -np.inf
        norm = logsumexp(logits, axis=1)
        likelihood = norm - logits[index, labels]
        p = np.exp(logits - norm[:, None])
        p[index, labels] -= 1
        grad = np.einsum("np,npd,n->d", p, batch, weights)
        return weights @ likelihood + penalty * (w @ w) / 2, grad + penalty * w

    result = minimize(loss, np.zeros(dimensions), jac=True, method="L-BFGS-B",
                      options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 500})
    assert result.success, result.message
    return center, scale, result.x, len(features)


def select(item, params):
    center, scale, coefficients, _ = params
    x = np.asarray(item["features"], dtype=np.float64)[:, :len(center)]
    return int((((x - center) / scale) @ coefficients).argmax())


def linked_distractor_run(rows, fitted):
    wins = Counter()
    eligible = 0
    for row in rows:
        item = row["variants"]["clean"]
        x = np.asarray(item["features"], dtype=np.float64)
        gold = item["gold"]
        linked = np.flatnonzero(x[:, 4])
        if gold < 0 or gold not in linked or len(linked) < 2:
            continue
        eligible += 1
        for name, scores in (("full", x[:, 0]), ("without", x[:, 1])):
            wins[name] += linked[scores[linked].argmax()] == gold
        for name, (center, scale, coefficients, _) in fitted.items():
            scores = ((x[linked, :len(center)] - center) / scale) @ coefficients
            wins[name] += linked[scores.argmax()] == gold
    return {"n": eligible, "gold_ranked_first_among_shortlisted_linked_pairs": {k: int(v) for k, v in wins.items()}}


def shortlist_frontier(items, params):
    center, scale, coefficients, _ = params
    result = {}
    for depth in (1, 2, 4, 8, 16):
        paired_hits, ceiling, pair_count = 0, 0, []
        for _, item in items:
            x = np.asarray(item["features"], dtype=np.float64)
            ranks = np.argsort(-x[:, :2], axis=0, kind="stable")[:depth]
            indices = np.unique(np.concatenate((ranks.ravel(), item["winners"])))
            score = ((x[indices] - center) / scale) @ coefficients
            paired_hits += int(indices[score.argmax()] == item["gold"])
            ceiling += int(item["gold"] in indices)
            pair_count.append(len(indices))
        result[str(depth)] = {"correct": paired_hits, "gold_in_shortlist": ceiling,
                              "mean_pairs": round(float(np.mean(pair_count)), 3)}
    return result


def feature_group_ablation(rows, names, primary):
    """Refit after removing sentence relevance or exact-title mention features."""
    training, heldout = ([r for r in rows if r["split"].endswith("-" + split)] for split in ("fit", "test"))
    output = {}
    for label, features in SENTENCE_FEATURE_GROUPS.items():
        indices = [*range(5), *[names.index(name) for name in features]]

        def project(item):
            return item | {"features": [[pair[i] for i in indices] for pair in item["features"]]}

        train_rows = [r | {"variants": {kind: project(item) for kind, item in r["variants"].items()}}
                      for r in training]
        model = fit(train_rows, len(indices), SCENARIOS)
        metrics = {}
        for split in ("twowiki-test", "musique-test"):
            for scenario in SCENARIOS + CONTEXTUAL:
                items = [r["variants"][scenario] for r in heldout
                         if r["split"] == split and scenario in r["variants"]]
                if not items:
                    continue
                hits = [select(project(item), model) == item["gold"] for item in items]
                full = [select(item, primary) == item["gold"] for item in items]
                metrics[f"{split}:{scenario}"] = {"correct": sum(hits), "n": len(hits),
                                                  "versus_all_sentence_features": paired(hits, full)}
        output[label] = {"features": [names[i] for i in indices], "tests": metrics}
    return output


def paired(a, b):
    d = np.asarray(a, dtype=np.int16) - np.asarray(b, dtype=np.int16)
    rng = np.random.default_rng(41729)
    index = rng.integers(0, len(d), size=(5000, len(d)))
    bootstrap = d[index].mean(1)
    return {"delta": round(float(d.mean()), 8), "bootstrap95": np.round(np.quantile(bootstrap, [.025, .975]), 8).tolist(),
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum())}


def summarize(shards):
    metadata = ("fixture_sha256", "split_sha256", "heads_sha256", "ablations_sha256",
                "baseline_sha256", "shortlist_per_head", "columns")
    assert all(all(s[k] == shards[0][k] for k in metadata) for s in shards)
    assert len(shards[0]["columns"]) == 16
    rows = sorted([r for s in shards if not s.get("contextual_test") for r in s["rows"]], key=lambda r: r["id"])
    contextual = [r for s in shards if s.get("contextual_test") for r in s["rows"]]
    assert len(set(r["id"] for r in rows)) == len(rows)
    counts = Counter(r["split"] for r in rows)
    assert counts == {"twowiki-fit": 768, "twowiki-test": 420, "musique-test": 300}
    if contextual:
        contextual_ids = {(r["id"], r["split"]) for r in contextual}
        assert len(contextual_ids) == len(contextual)
        assert contextual_ids == {(r["id"], r["split"]) for r in rows if r["split"].endswith("-test")}
        lookup = {r["id"]: r["variants"] for r in contextual}
        rows = [r | {"variants": r["variants"] | lookup.get(r["id"], {})} for r in rows]
    for row in rows:
        for item in row["variants"].values():
            count = len(item["features"])
            assert 16 <= count <= 32
            assert -1 <= item["gold"] < count
            assert len(item["winners"]) == 2 and all(0 <= i < count for i in item["winners"])
            assert all(len(x) == 16 for x in item["features"])
    train = [r for r in rows if r["split"] == "twowiki-fit"]
    fitted = {name: fit(train, dim, scenarios) for name, (dim, scenarios) in MODELS.items()}
    report, challenge, by_scenario = {}, {}, {}
    for split in ("twowiki-test", "musique-test"):
        heldout = [r for r in rows if r["split"] == split]
        challenge[split] = linked_distractor_run(heldout, fitted)
        for scenario in SCENARIOS + (CONTEXTUAL if contextual else ()):
            items = [(r["id"], r["variants"][scenario]) for r in heldout if scenario in r["variants"]]
            responses = {"full": [], "without": [], "oracle_shortlist": [], "oracle_heads": []}
            responses.update({name: [] for name in fitted})
            for _, item in items:
                gold = item["gold"]
                responses["full"].append(item["winners"][0] == gold)
                responses["without"].append(item["winners"][1] == gold)
                responses["oracle_shortlist"].append(gold >= 0)
                responses["oracle_heads"].append(gold in item["winners"])
                for name, params in fitted.items():
                    responses[name].append(select(item, params) == gold)
            frontier = shortlist_frontier(items, fitted["semantic_sentence"])
            assert frontier["16"]["correct"] == sum(responses["semantic_sentence"])
            report[f"{split}:{scenario}"] = {"n": len(items), "correct": {k: sum(v) for k, v in responses.items()},
                                            "paired_vs_full": {k: paired(responses[k], responses["full"]) for k in fitted},
                                            "semantic_vs_lexical": paired(responses["semantic_sentence"], responses["lexical_sentence"]),
                                            "semantic_vs_base": paired(responses["semantic_sentence"], responses["base_reranker"]),
                                            "fixed_weight_shortlist_frontier": frontier}
            by_scenario[split, scenario] = {k: dict(zip((id_ for id_, _ in items), values, strict=True))
                                           for k, values in responses.items()}
    drops, interactions = {}, {}
    for split in ("twowiki-test", "musique-test"):
        for sham, link in (("sham", "linked"), CONTEXTUAL):
            if (split, link) not in by_scenario:
                continue
            before, after = (by_scenario[split, v] for v in (sham, link))
            keys = list(before["full"])
            assert set(keys) == set(after["full"])
            drops[f"{split}:{link}"] = {k: paired([after[k][q] for q in keys], [before[k][q] for q in keys])
                                         for k in before}
            semantic = [int(after["semantic_sentence"][q]) - int(before["semantic_sentence"][q]) for q in keys]
            full = [int(after["full"][q]) - int(before["full"][q]) for q in keys]
            interactions[f"{split}:{link}"] = paired(semantic, full)
    return {**{k: shards[0][k] for k in metadata}, "counts": dict(sorted(counts.items())),
            "hardware": sorted({s["device"] for s in shards}),
            "fitted": {name: {"eligible_variants": params[-1], "coefficients": np.round(params[2], 8).tolist()}
                       for name, params in fitted.items()}, "tests": report,
            "clean_linked_shortlist_challenge": challenge, "linked_minus_sham": drops,
            "semantic_versus_frozen_link_interaction": interactions,
            "feature_group_ablation": feature_group_ablation(rows, shards[0]["columns"], fitted["semantic_sentence"])}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--write-replay", type=Path)
    p.add_argument("--contextual-input", type=Path)
    p.add_argument("--write-policy", type=Path)
    a = p.parse_args()
    if a.input.is_dir():
        shards = [json.loads(p.read_text()) for p in sorted(a.input.glob("*-*.json"))]
    else:
        shards = json.loads(gzip.decompress(a.input.read_bytes()))
    if a.contextual_input:
        shards.extend(json.loads(p.read_text()) for p in sorted(a.contextual_input.glob("*-*.json")))
    a.output.write_text(json.dumps(summarize(shards), indent=2, allow_nan=False) + "\n")
    if a.write_policy:
        training = sorted([r for s in shards if not s.get("contextual_test") for r in s["rows"]
                           if r["split"] == "twowiki-fit"], key=lambda r: r["id"])
        mean, scale, coefficients, support = fit(training, *MODELS["semantic_sentence"])
        policy = {"fixture_sha256": shards[0]["fixture_sha256"], "split_sha256": shards[0]["split_sha256"],
                  "features": shards[0]["columns"], "fit_variants": support, "mean": mean.tolist(),
                  "scale": scale.tolist(), "coefficients": coefficients.tolist()}
        a.write_policy.write_text(json.dumps(policy, indent=2, allow_nan=False) + "\n")
    if a.write_replay:
        small = [{k: v for k, v in s.items() if k != "seconds"} for s in shards]
        a.write_replay.write_bytes(gzip.compress(json.dumps(small, separators=(",", ":")).encode(), mtime=0))


if __name__ == "__main__":
    main()
