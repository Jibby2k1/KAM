#!/usr/bin/env python3
"""Lock numerical final-evidence selectors before scoring the frozen test lists."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from analyze_kam_shared_serving import unpack

L2 = 0.04
MODEL_COUNT = 3


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def features(paths, manifest):
    chunks = sorted(map(unpack, paths), key=lambda r: r["shard"])
    cols = chunks[0]["columns"]
    assert len(cols) == 12 and len(set(cols)) == 12
    assert [p["shard"] for p in chunks] == list(range(len(chunks)))
    keys = ("weights_sha256", "columns", "fixture_sha256", "corpus_sha256", "model_revisions", "source_sha256")
    metadata = {key: chunks[0][key] for key in keys}
    assert manifest["fixture_sha256"] == metadata["fixture_sha256"]
    assert manifest["corpus_sha256"] == metadata["corpus_sha256"]
    assert all(p["device"] == "NVIDIA L4" and all(p[k] == v for k, v in metadata.items()) for p in chunks)
    for c in chunks:
        assert c["shards"] == len(chunks)
        assert [r["id"] for r in c["rows"]] == [r["id"] for r in manifest["rows"][c["shard"]::len(chunks)]]
    rows = {q["id"]: q["seeds"] for c in chunks for q in c["rows"]}
    return rows, metadata


def loss(w, x, target):
    raw = np.einsum("ijk,k->ij", x, w, optimize=True)
    log = logsumexp(raw, axis=1, keepdims=True)
    prob = np.exp(raw - log)
    ll = float(np.mean(np.sum(-target * (raw - log), axis=1)) + L2 * (w @ w) * .5)
    delta = np.einsum("ij,ijk->k", prob - target, x, optimize=True) / len(x) + L2 * w
    return ll, delta


def fit(panel, training):
    vectors, labels = [], []
    for q in panel:
        pool = training[q["id"]]["pool"]
        truth = set(q["gold"])
        gold = np.array([p in truth for p in pool], dtype=np.float64)
        if not training[q["id"]]["fallback"] and gold.any():
            labels.append(gold / gold.sum())
            vectors.append(training[q["id"]]["features"])
    x = np.asarray(vectors, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    assert x.shape == (len(y), 128, 12)
    center = x.mean(axis=(0, 1))
    scale = x.std(axis=(0, 1)).clip(min=1e-6)
    z = (x - center) / scale
    model = minimize(loss, np.zeros(x.shape[-1]), jac=True, args=(z, y), method="L-BFGS-B")
    assert model.success and np.isfinite(model.x).all(), model.message
    return {"coef": model.x.tolist(), "mean": center.tolist(), "scale": scale.tolist(), "fit_questions": len(y),
            "objective": float(model.fun)}


def fit_all(a):
    source = unpack(a.manifest)
    test = unpack(a.test)
    assert source["base_fixture_sha256"] == test["fixture_sha256"]
    assert source["corpus_sha256"] == test["corpus_sha256"]
    assert len(source["rows"]) == 208 and len(test["rows"]) == 576
    assert {q["id"] for q in source["rows"]}.isdisjoint(q["id"] for q in test["rows"])
    assert {x for q in source["rows"] for x in q["gold"]}.isdisjoint(x for q in test["rows"] for x in q["gold"])
    source_rank, ds = features(a.features, source)
    result = [fit(source["rows"], {id_: ranks[i] for id_, ranks in source_rank.items()})
              for i in range(MODEL_COUNT)]
    return {"fit_provenance": ds, "test_fixture_sha256": test["fixture_sha256"],
            "fit_features_sha256": list(map(digest, a.features)), "penalty": L2, "models": result,
            "split_support_title_counts": source["support_title_count"]}


def score(a):
    test = unpack(a.manifest)
    policy = json.loads(a.policy.read_text())
    assert test["fixture_sha256"] == policy["test_fixture_sha256"]
    rank, data = features(a.features, test)
    assert all(data[k] == policy["fit_provenance"][k] for k in
               ("weights_sha256", "columns", "corpus_sha256", "model_revisions", "source_sha256"))
    rows = []
    for q in test["rows"]:
        selection = []
        for scored, model in zip(rank[q["id"]], policy["models"], strict=True):
            matrix = np.asarray(scored["features"], dtype=np.float64)
            pool = scored["pool"]
            assert matrix.shape == (128, 12) and len(pool) == len(set(pool)) == 128
            if scored["fallback"]:
                picks = pool[:4]
            else:
                v = ((matrix - model["mean"]) / model["scale"]) @ model["coef"]
                order = np.argsort(-v, kind="stable")[:4]
                picks = [pool[i] for i in order]
            selection.append(picks)
        rows.append({"id": q["id"], "selections": selection})
    return {"fixture_sha256": test["fixture_sha256"], "policy_sha256": digest(a.policy),
            "test_features_sha256": list(map(digest, a.features)), "provenance": data, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "score"))
    p.add_argument("--features", nargs="+", type=Path, required=True)
    for key in ("manifest", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--test", type=Path)
    p.add_argument("--policy", type=Path)
    a = p.parse_args()
    result = (fit_all if a.mode == "fit" else score)(a)
    a.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
