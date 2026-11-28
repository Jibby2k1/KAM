#!/usr/bin/env python3
"""Profile physical cache compaction with deterministic output checks."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

from kam_adapter_preservation import digest, key
from kam_math_kl_protocol import test_runs, write_json
from kam_tabmwp_archive_noise import BUDGET, encode_case, load_panel

PROTOCOL = "kam-tabmwp-retirement-v1"
WIDTHS = (8, 32)


def retire(model, stack, tokenizer, rows, compact):
    import torch
    from eval_kam_pretrained import next_logits, pad

    with torch.inference_mode():
        x, mask = pad(rows, tokenizer.pad_token_id)
        stack.reset()
        logits, past = next_logits(model, stack, x, mask, use_cache=True)
        stop_ids = model.generation_config.eos_token_id
        stop_ids = {stop_ids} if isinstance(stop_ids, int) else set(stop_ids)
        owners = list(range(len(rows)))
        text = [[] for _ in rows]
        ended = [False for _ in rows]
        slots, active, retired = 0, 0, 0
        for step in range(BUDGET):
            slots += len(owners)
            active += sum(not ended[i] for i in owners)
            next_ids = logits.argmax(-1)
            for i, token in zip(owners, next_ids.tolist(), strict=True):
                if not ended[i]:
                    if token in stop_ids:
                        ended[i] = True
                    else:
                        text[i].append(token)
            keep = [j for j, i in enumerate(owners) if not ended[i]]
            if not keep or step + 1 == BUDGET:
                break
            if compact and (step + 1) % 16 == 0 and 4 * len(keep) <= 3 * len(owners):
                indices = torch.tensor(keep, device=x.device)
                # Transformers and the residual dot adapter own separate caches.
                past.batch_select_indices(indices)
                for layer in stack.modules:
                    if layer.cache is not None:
                        layer.cache = tuple(t.index_select(0, indices) for t in layer.cache)
                mask = mask.index_select(0, indices)
                next_ids = next_ids.index_select(0, indices)
                owners = [owners[j] for j in keep]
                retired += 1
            next_ids = next_ids.unsqueeze(1)
            mask = torch.cat((mask, torch.ones_like(next_ids, dtype=torch.bool)), dim=1)
            logits, past = next_logits(model, stack, next_ids, mask, past, use_cache=True)
        stack.reset()
    return (text, ended), {"token_slots": slots, "active_steps": active, "compactions": retired}


def evaluate(root, work, index):
    import torch
    from eval_kam_adapter_preservation import load_checkpoint

    rows, source = load_panel(work)
    selection = work / "selection.json"
    selected = json.loads(selection.read_text())
    assert selected["selected_recipe"] == "lower_lr_kl05" and selected["selected_kl_weight"] == .5
    run = test_runs(selected["selected_recipe"])[index]
    model, tokenizer, stack, metadata = load_checkpoint(root, *run)
    gpu = torch.cuda.get_device_name()
    assert gpu == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    rows = sorted(rows, key=lambda row: hashlib.sha256((PROTOCOL + ":" + row["id"]).encode()).digest())[:32]
    cohort = [(row["id"], encode_case(tokenizer, row)["current_only"]) for row in rows]
    cohort.sort(key=lambda item: len(item[1]))
    metrics = {}
    for width in WIDTHS:
        times = {"fixed": [], "physical_retirement": []}
        outcomes, counters = {}, {}
        retire(model, stack, tokenizer, [x[1] for x in cohort[:width]], False)
        for mode in ("fixed", "physical_retirement", "physical_retirement", "fixed"):
            observed = []
            counts = {"compactions": 0, "token_slots": 0, "active_steps": 0}
            torch.cuda.synchronize()
            started = perf_counter()
            for offset in range(0, len(cohort), width):
                requests = [x[1] for x in cohort[offset:offset+width]]
                (tokens, eos), total = retire(model, stack, tokenizer, requests, mode != "fixed")
                observed.extend(zip(tokens, eos, strict=True))
                for field in counts:
                    counts[field] += total[field]
            torch.cuda.synchronize()
            times[mode].append(perf_counter() - started)
            assert outcomes.get(mode, observed) == observed and counters.get(mode, counts) == counts
            outcomes[mode], counters[mode] = observed, counts
        left, right = outcomes["fixed"], outcomes["physical_retirement"]
        metrics[width] = {"seconds": times, "counters": counters, "records": [
            {"id": ident, "exact": original == candidate, "fixed_tokens": len(original[0]),
             "retired_tokens": len(candidate[0]), "fixed_eos": original[1], "retired_eos": candidate[1]}
            for (ident, _), original, candidate in zip(cohort, left, right, strict=True)
        ]}
    write_json(work / f"retire-{key(*run)}.json", {"metadata": {
        **metadata, "gpu": gpu, "protocol": PROTOCOL, "manifest_sha256": source,
        "selection_sha256": digest(selection)}, "widths": metrics})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("evaluate",))
    parser.add_argument("root", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("index", type=int)
    args = parser.parse_args()
    evaluate(args.root, args.work, args.index)
