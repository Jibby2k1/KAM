#!/usr/bin/env python3
"""Select which heads keep dense attention using frozen residual errors."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from audit_kam_omitted_attention import (
    BLOCK, BOOKS, CONTEXT, REVISION, WEIGHTS, dataset, load_model, local_outputs, observe_dense, sha, write_json,
)

PROTOCOL = "kam-head-fallback-v1"
HEADS = 9
GQA_GROUP = 3
LAYERS = 30
BUDGET = HEADS * LAYERS // 2
HEAD_MASKS = ((np.arange(2**HEADS)[:, None] >> np.arange(HEADS)) & 1).astype(bool)
FULL = 2**HEADS - 1


def cross_terms(module, dense, protected):
    difference = protected - dense
    weights = module.o_proj.weight.unflatten(1, (HEADS, -1))
    projected = torch.einsum("hqd,ohd->hqo", difference, weights)
    together = module.o_proj(difference.transpose(0, 1).flatten(1))
    assert torch.allclose(projected.sum(dim=0), together, atol=3e-4, rtol=3e-4)
    per_head = projected.flatten(1).double()
    terms = (per_head @ per_head.T).numpy()
    assert np.isclose(terms.sum(), float(together.double().square().sum()), rtol=2e-5, atol=1e-6)
    ground = module.o_proj(dense.transpose(0, 1).flatten(1))
    energy = float(ground.double().square().sum())
    return terms, energy


def audit_book(model, tokens, positions):
    result = {}

    def assess(module, query, keys, values, _dense, scaling):
        reference, outputs, _, _ = local_outputs(query, keys, values, positions, scaling)
        result[module.layer_idx] = cross_terms(module, reference, outputs["protected"])

    observe_dense(model, tokens, assess, tag=PROTOCOL)
    assert sorted(result) == list(range(LAYERS))
    raw, energy = zip(*(result[layer] for layer in range(LAYERS)), strict=True)
    return {"pairwise_projected_error": np.stack(raw).tolist(), "dense_output_energy_by_layer": energy}


def schedule_errors(covariance, schedule):
    sparse = (~HEAD_MASKS[np.asarray(schedule)]).astype(float)
    return np.einsum("li,blij,lj->b", sparse, covariance, sparse)


def optimize(covariance, *, allowance, group_width):
    dense_heads = HEAD_MASKS.sum(axis=1)
    mixed = (dense_heads > 0) & (dense_heads < HEADS)
    group_shape = HEAD_MASKS.reshape(-1, HEADS // group_width, group_width)
    valid = (group_shape.all(axis=-1) == group_shape.any(axis=-1)).all(axis=-1)
    sparse = (~HEAD_MASKS).astype(float)
    error = np.einsum("mi,lij,mj->lm", sparse, covariance, sparse)
    states = {(0, 0): (0.0, ())}
    for layer, costs in enumerate(error):
        choices = []
        for count in range(0, HEADS + 1, group_width):
            feasible = np.flatnonzero(valid & (dense_heads == count))
            best = int(feasible[np.argmin(costs[feasible])])
            choices.append((best, count, int(mixed[best])))
        next_states = {}
        for (used, split), (cost, mask) in states.items():
            for choice, count, local_split in choices:
                pair = (used + count, split + local_split)
                if pair[0] > BUDGET or pair[0] + (LAYERS - layer - 1) * HEADS < BUDGET or pair[1] > allowance:
                    continue
                old = next_states.get(pair)
                value = cost + costs[choice]
                if old is None or value < old[0]:
                    next_states[pair] = (value, mask + (choice,))
        states = next_states
    _, (_, result) = min(states.items(), key=lambda pair: (pair[1][0], pair[0][1], pair[1][1]))
    assert sum(dense_heads[m] for m in result) == BUDGET and sum(mixed[m] for m in result) <= allowance
    return result


def compare(books):
    articles = sorted([book["id"] for book in books], key=lambda book: sha(f"{PROTOCOL}/{book}".encode()))
    fit_ids = set(articles[:BOOKS // 2])
    assert len(fit_ids) == BOOKS // 2 and len(set(articles) - fit_ids) == BOOKS // 2
    fit = np.array([book["id"] in fit_ids for book in books])
    denom = np.array([sum(book["dense_output_energy_by_layer"]) for book in books])
    covariance = np.array([book["pairwise_projected_error"] for book in books]) / denom[:, None, None, None]
    average = covariance[fit].mean(axis=0)
    half_depth = tuple(FULL if layer % 2 == 0 else 0 for layer in range(LAYERS))
    diagonal = average * np.eye(HEADS)
    schedules = {"alternating": half_depth,
                 "whole_layers": optimize(average, allowance=0, group_width=HEADS),
                 "groups_up_to_4": optimize(average, allowance=4, group_width=GQA_GROUP),
                 "groups_up_to_8": optimize(average, allowance=8, group_width=GQA_GROUP),
                 "groups_all_layers": optimize(average, allowance=LAYERS, group_width=GQA_GROUP),
                 "heads_all_layers": optimize(average, allowance=LAYERS, group_width=1),
                 "heads_diagonal": optimize(diagonal, allowance=LAYERS, group_width=1)}
    test = covariance[~fit]
    reference = schedule_errors(test, schedules["whole_layers"])
    draw = np.random.default_rng(98413).integers(len(reference), size=(20_000, len(reference)))
    results = {}
    for name, mask in schedules.items():
        active = HEAD_MASKS[np.array(mask)]
        assert int(active.sum()) == BUDGET
        error = schedule_errors(test, mask)
        gain = reference - error
        draw_deltas = gain[draw].mean(axis=1) / reference[draw].mean(axis=1)
        results[name] = {"dense_heads": [np.flatnonzero(layer).tolist() for layer in active],
                         "dense_head_layer_count": BUDGET, "mixed_layers": int((active.any(1) & ~active.all(1)).sum()),
                         "held_book_error": error.tolist(), "mean_held_book_error": float(error.mean()),
                         "improvement_vs_whole_layer": float(gain.mean() / reference.mean()),
                         "held_books_better_than_whole_layer": int((gain > 0).sum()),
                         "paired_book_bootstrap95": np.quantile(draw_deltas, (.025, .975)).tolist()}
    return {"fit_books": [book["id"] for book in books if book["id"] in fit_ids],
            "test_books": [book["id"] for book in books if book["id"] not in fit_ids], "schedules": results}


@torch.inference_mode()
def collect(args):
    torch.set_num_threads(args.threads)
    model, tokenizer = load_model(args.model)
    assert model.config.num_attention_heads == HEADS and model.config.num_hidden_layers == LAYERS
    positions = (torch.arange(544, CONTEXT, BLOCK)[:, None] + torch.tensor([0, 31])).flatten()
    rows = []
    for record, tokens in dataset(tokenizer, args.sources, args.books, limit=args.limit):
        rows.append({**record, **audit_book(model, tokens, positions)})
        print("verified head projection book", record["id"], flush=True)
    write_json(args.output, {"protocol": PROTOCOL, "revision": REVISION, "model_sha256": WEIGHTS,
                            "source_list_sha256": sha(args.sources.read_bytes()), "positions": positions.tolist(),
                            "torch": torch.__version__, "books": rows})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    cli = parser.add_subparsers(dest="action", required=True)
    pre = cli.add_parser("collect")
    for name in ("model", "sources", "books", "output"):
        pre.add_argument(name, type=Path)
    pre.add_argument("--limit", type=int, default=BOOKS, choices=(1, BOOKS))
    pre.add_argument("--threads", type=int, default=4)
    post = cli.add_parser("select")
    post.add_argument("projections", type=Path)
    post.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.action == "collect":
        collect(args)
    else:
        data = args.projections.read_bytes()
        panel = json.loads(data)
        assert (panel["protocol"], panel["model_sha256"], len(panel["books"])) == (PROTOCOL, WEIGHTS, BOOKS)
        write_json(args.output, {"protocol": PROTOCOL, "projection_sha256": sha(data), **compare(panel["books"])})
