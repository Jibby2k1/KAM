#!/usr/bin/env python3
"""Compare frozen KAM retrieval policies on the same IIRC link candidate pools."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from run_kam_hotpot_pair_scoring import links
from run_kam_hotpot_pool_scaling import pair_selection
from run_kam_musique_transfer import restore_checkpoint
from run_kam_natural_feasibility import Encoder, candidate_texts
from run_kam_natural_router import score

ARMS = ("single_shot", "sequential", "pair", "pair_masked")


@torch.inference_mode()
def evaluate(rows, encoder, adapter, pair):
    banks = [candidate_texts({"context": [(t, [body]) for t, body in row["candidates"]]}) for row in rows]
    flats = list(dict.fromkeys(text for bank in banks for text in bank))
    vectors = dict(zip(flats, encoder(flats, "passage"), strict=True))
    by_width = defaultdict(list)
    for i, row in enumerate(rows):
        by_width[len(row["candidates"])].append(i)
    graphs = {}
    for width, indices in by_width.items():
        cases = [{"context": [(t, [body]) for t, body in rows[i]["candidates"]]} for i in indices]
        graphs.update(zip(indices, links(cases).cuda(), strict=True))
    pair_indices = {n: torch.triu_indices(n, n, 1, device="cuda") for n in by_width}
    keys = [torch.stack([vectors[x] for x in bank]).unsqueeze(0) for bank in banks]

    def request(i, arm):
        row, bank, graph = rows[i], keys[i], graphs[i].unsqueeze(0)
        input_text = f'{row["question"]} Source paragraph: {row["source_excerpt"]}'
        torch.cuda.synchronize()
        started = perf_counter()
        query = adapter(encoder([input_text], "query", max_length=256))
        if arm in ("single_shot", "sequential"):
            affinity = score(query, bank)
            chosen = affinity.argsort(descending=True, stable=True)[0, :2].tolist()
            if arm == "sequential":
                first = chosen[0]
                excerpt = " ".join(banks[i][first].split()[:48])
                conditioned = f"{input_text} First retrieved document: {excerpt}"
                q2 = adapter(encoder([conditioned], "query", max_length=256))
                blended = (affinity + score(q2, bank)) * .5
                blended[0, first] = -torch.inf
                chosen = [first, blended.argmax(-1).item()]
        else:
            graph = graph if arm == "pair" else None
            chosen = pair_selection(query, bank, graph, pair, pair_indices[bank.shape[1]])[0].tolist()
        torch.cuda.synchronize()
        return {"selected": chosen, "route_ms": (perf_counter() - started) * 1000}

    for arm in ARMS:
        request(0, arm)
    output = []
    for i, row in enumerate(rows):
        order = Random(row["id"]).sample(ARMS, k=len(ARMS))
        gold = row["gold_indices"]
        graph = graphs[i]
        linked = bool(graph[gold[0], gold[1]] | graph[gold[1], gold[0]])
        pool = sha256(json.dumps(row["candidates"], ensure_ascii=False).encode()).hexdigest()
        output.append({"id": row["id"], "source_id": row["source_id"], "pool_sha256": pool,
                       "candidates": len(row["candidates"]), "gold": gold, "gold_title_link": linked,
                       "arms": {arm: request(i, arm) for arm in order}})
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "assets", "config", "baseline", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline-sha256", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--shards", type=int, required=True)
    a = p.parse_args()
    assert 0 <= a.shard < a.shards
    torch.set_float32_matmul_precision("highest")
    f = json.loads(a.fixture.read_text())
    rows = f["rows"][a.shard::a.shards]
    config = json.loads(a.config.read_text())
    adapter, pair = restore_checkpoint(a.baseline, a.config, config, None, None, weights_sha256=a.baseline_sha256)
    encoder = Encoder(a.assets / "encoder", 128)
    results = evaluate(rows, encoder, adapter, pair)
    payload = {"fixture_sha256": digest(a.fixture), "source_sha256": f["source_sha256"],
               "config_sha256": digest(a.config), "baseline_sha256": a.baseline_sha256,
               "assets": json.loads((a.assets / "revisions.json").read_text()), "device": torch.cuda.get_device_name(),
               "torch": torch.__version__, "shard": a.shard, "shards": a.shards, "rows": results}
    a.output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
