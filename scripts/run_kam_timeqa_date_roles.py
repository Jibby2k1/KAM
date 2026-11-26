#!/usr/bin/env python3
"""Test whether query-relevant sentence dates resist temporal metadata noise."""

import argparse
from pathlib import Path
import re
from time import perf_counter

import torch
from torch import nn
from torch.nn.functional import log_softmax

from audit_kam_natural_data import digest
from run_kam_hotpot_independent import read, save
from run_kam_natural_feasibility import Encoder
from run_kam_timeqa_contrasts import YEAR, data_matrix, padded, with_date_collisions
from run_kam_timeqa_human import make_matrix_entries

ROLES = ("all_years", "best_sentence", "weighted_sentences")


@torch.no_grad()
def make_features(rows, encoder):
    original, contexts = data_matrix(encoder, rows)
    spans, positions = [], []
    for ctx in contexts:
        boundaries = []
        for _, text in ctx["context"]:
            sentences = re.split(r"(?<=[.!?])\s+", " ".join(text))
            offset = len(spans)
            for sentence in sentences:
                years = set(map(int, YEAR.findall(sentence)))
                if years:
                    spans.append((YEAR.sub("[date]", sentence), years))
            boundaries.append((offset, len(spans)))
        positions.append(boundaries)
    sentence_vectors = encoder([x for x, _ in spans], "passage")
    query_vectors = encoder([q["query_without_date"] for r in rows for q in r["questions"]], "query")
    results = {v: [] for v in ROLES}
    for i, (row, candidates) in enumerate(zip(rows, positions, strict=True)):
        for qid, question in enumerate(row["questions"]):
            flat = 2 * i + qid
            q = query_vectors[flat]
            lo, hi = question["lower"] // 12, question["upper"] // 12
            variants = {v: [] for v in ROLES[1:]}
            one, last = candidates[0][0], candidates[-1][1]
            relevance = (sentence_vectors[one:last] @ q).tolist()
            for begin, end in candidates:
                if begin == end:
                    for state in variants.values():
                        state.append((0, 0, 0, 0))
                    continue
                scores = relevance[begin-one:end-one]
                best = max(range(len(scores)), key=scores.__getitem__)
                years = [x[1] for x in spans[begin:end]]
                metrics = []
                weights = []
                for score, bucket in zip(scores, years, strict=True):
                    count = sum(lo <= y <= hi for y in bucket)
                    distance = min(max(0, lo - y, y - hi) for y in bucket)
                    metrics.append((1, bool(count), max(0, 1 - distance / 30), count / len(bucket)))
                    # Centering changes scale only. The weight is derived from
                    # date-masked sentence and query embeddings.
                    weights.append(max(0, min(1, (score - 0.55) / 0.35)))
                variants["best_sentence"].append(metrics[best])
                variants["weighted_sentences"].append(tuple(max(w * values[j] for w, values in zip(weights, metrics, strict=True))
                                                              for j in range(4)))
            results["all_years"].append(original[flat])
            for key, extra in variants.items():
                assert len(extra) == len(original[flat])
                results[key].append(torch.cat((original[flat][:, :4], torch.tensor(extra, device="cuda")), -1))
    return results


def fit(dataset, features):
    heads = {}
    for role, matrices in features.items():
        x, allowed, gold = padded(dataset, matrices)
        torch.manual_seed(852761)
        head = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1)).cuda()
        optimizer = torch.optim.AdamW(head.parameters(), lr=0.004, fused=True)
        rng = torch.Generator(device="cuda").manual_seed(61529)
        for _ in range(30):
            for block in torch.randperm(len(x), generator=rng, device="cuda").split(64):
                adjusted = head(x[block]).squeeze(-1)
                scores = log_softmax(adjusted.masked_fill(~allowed[block], -torch.inf), -1)
                loss = -scores.masked_fill(~gold[block], -torch.inf).logsumexp(-1).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        assert torch.isfinite(loss)
        heads[role] = {"weights": head.cpu().state_dict(), "loss": float(loss.detach())}
    return heads


@torch.no_grad()
def select(group, qlabels, features, fitted):
    models = {}
    for role, entry in fitted.items():
        model = nn.Sequential(nn.Linear(8, 16), nn.SiLU(), nn.Linear(16, 1)).cuda().eval()
        model.load_state_dict(entry["weights"])
        models[role] = model
    output = []
    for i, (row, qdata) in enumerate(zip(group, qlabels, strict=True)):
        qs = []
        for j, q in enumerate(qdata):
            flat = 2 * i + j
            gold = set(q)
            n = len(row["context"])
            modes = {}
            for role, datasets in features.items():
                x = datasets[flat]
                assert len(x) >= n
                scores = models[role](x).flatten()
                # Shallow semantic retrieval remains a fixed reference.
                if role == "all_years":
                    all_scores = {"dense": x[:, 0], role: scores}
                else:
                    all_scores = {role: scores}
                for name, values in all_scores.items():
                    first, full = int(values[:n].argmax()), int(values.argmax())
                    modes[name] = {"clean": first in gold, "with_noise": full in gold,
                                   "clean_slot": first, "noise_slot": full, "noise_pick": full >= n}
            qs.append(modes)
        output.append(qs)
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fit", "test", "human"))
    for name in ("assets", "training-fixture", "weights", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--variant", choices=("donor", "sham", "date_collision"), default="donor")
    p.add_argument("--human-fixture", type=Path)
    p.add_argument("--reference-weights", type=Path)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    start = perf_counter()
    torch.set_float32_matmul_precision("highest")
    fixture = read(a.training_fixture)
    encoder = Encoder(a.assets / "encoder", 64)
    binding = {"training_sha256": digest(a.training_fixture), "models": read(a.assets / "revisions.json")}
    if a.mode == "fit":
        rows = fixture["rows"]["dev"]
        state = fit(rows, make_features(rows, encoder))
        old = torch.load(a.reference_weights, weights_only=True, map_location="cpu")
        assert old["fixture_sha256"] == binding["training_sha256"]
        param_gap = max(float((v - old["heads"]["learned_dates"][k]).abs().max())
                        for k, v in state["all_years"]["weights"].items())
        torch.save({**binding, "heads": state}, a.weights)
        details = {"train_entities": len(rows), "raw_refit_max_param_delta": param_gap,
                   "loss": {k: v["loss"] for k, v in state.items()}}
    else:
        state = torch.load(a.weights, weights_only=True, map_location="cpu")
        assert all(state[k] == v for k, v in binding.items())
        if a.mode == "human":
            fixture_ = read(a.human_fixture)
            rows = fixture_["rows"][a.shard::a.shards]
            group, tags = make_matrix_entries(rows)
            qlabels = [[r["gold"], r["gold"]] for r in rows]
            identity = [{"id": r["id"], "entity": r["entity"], "eligible": True,
                         "tags": t} for r, t in zip(rows, tags, strict=True)]
        else:
            rows = fixture["rows"]["test"][a.shard::a.shards]
            group = rows if a.variant == "donor" else with_date_collisions(rows, a.variant)
            qlabels = [[q["gold"] for q in r["questions"]] for r in rows]
            identity = [{"id": r["id"], "entity": r["entity"], "eligible": r.get("date_collision_eligible", True)}
                        for r in group]
        selected = select(group, qlabels, make_features(group, encoder), state["heads"])
        if a.mode == "human":
            for result, entry in zip(selected, identity, strict=True):
                for i, mode in enumerate(("easy", "hard")):
                    if entry["tags"][mode] == "unparsed":
                        for role in ROLES:
                            result[i][role] = result[i]["dense"].copy()
        details = {"rows": [{**identity, "retrieval": values} for identity, values in zip(identity, selected, strict=True)]}
    save(a.output, {**binding, "mode": a.mode, "variant": a.variant, "weights_sha256": digest(a.weights),
                    "gpu": torch.cuda.get_device_name(), "elapsed_seconds": perf_counter() - start, **details})


if __name__ == "__main__":
    main()
