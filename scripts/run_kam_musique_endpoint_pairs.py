#!/usr/bin/env python3
"""Measure endpoint selection with a fixed, paired MuSiQue candidate bank."""

import argparse
import gzip
from hashlib import sha256
import json
from pathlib import Path
from random import Random
from time import perf_counter

import torch

from audit_kam_natural_data import digest
from prepare_kam_musique_endpoint_pairs import literal
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_reader_composition import read_packed, tokenized_question, unpack
from run_kam_musique_reader_windows import READERS
from run_kam_natural_feasibility import Encoder, measure_answer

MODES = ("question_only", "endpoints", "true_prefix", "cross_prefix", "bound_endpoints")


def exact(answer, aliases):
    return any(measure_answer(answer, ref)["em"] for ref in aliases)


def build_prompts(fixture, shard, count, tokenizer):
    def body(doc, cap):
        return tokenizer(doc["title"] + ". " + doc["body"], add_special_tokens=False,
                         max_length=cap, truncation=True).input_ids

    prefix_separator = tokenizer("Previous evidence:", add_special_tokens=False).input_ids
    endpoint_separator = tokenizer("Candidate passages:", add_special_tokens=False).input_ids
    sep = tokenizer(" ; ", add_special_tokens=False).input_ids

    def join(parts):
        return [token for i, row in enumerate(parts) for token in ((sep if i else []) + row)]

    rejected, items, parts, prepared = [], [], [], {name: [] for name in MODES}
    for pair in fixture["pairs"][shard::count]:
        a, b = pair["questions"]
        previous = [[body(doc, 128) for doc in r["evidence"][:-1]] for r in (a, b)]
        widths = [min(len(x), len(y)) for x, y in zip(*previous, strict=True)]
        earlier = [join([section[:width] for section, width in zip(p, widths, strict=True)]) for p in previous]
        last = [body(r["evidence"][-1], 192) for r in (a, b)]
        base_docs = tokenizer.batch_decode(last)
        early_text = tokenizer.batch_decode(earlier)
        for i, q in enumerate((a, b)):
            present = literal(q["anchor"], early_text[i]) and literal(q["anchor"], base_docs[i])
            answer_present = any(literal(x, base_docs[i]) for x in q["answers"])
            if not (present and answer_present):
                rejected.append(pair["id"])
                break
        else:
            for q_index, (q, other) in enumerate(((a, b), (b, a))):
                for reverse in (False, True):
                    evidence = last[::-1] if reverse else last
                    ends = endpoint_separator + join(evidence)
                    true, sham = earlier[q_index], earlier[1-q_index]
                    prefix = tokenized_question(tokenizer, q["question"])
                    bound = tokenized_question(tokenizer, q["bound_final"])
                    case = {"question_only": prefix, "endpoints": prefix + ends,
                            "true_prefix": prefix + prefix_separator + true + ends,
                            "cross_prefix": prefix + prefix_separator + sham + ends, "bound_endpoints": bound + ends}
                    assert len(case["true_prefix"]) == len(case["cross_prefix"])
                    row = {"pair": pair["id"], "family": pair["family"], "hops": pair["hops"],
                           "question": q["id"], "reverse": reverse,
                           "tokens": {m: len(ids)+1 for m, ids in case.items()}}
                    items.append((row, q["answers"], other["answers"],
                                  {"endpoints": q["question"], "bound_endpoints": q["bound_final"],
                                   "true_prefix": q["question"] + " Previous evidence: " + early_text[q_index],
                                   "cross_prefix": q["question"] + " Previous evidence: " + early_text[1-q_index]},
                                  base_docs[::-1] if reverse else base_docs, q_index ^ int(reverse)))
                    parts.append((prefix, prefix + prefix_separator + true, endpoint_separator, evidence, sep))
                    for arm, ids in case.items():
                        assert len(ids) < 1024
                        tokens = (*ids, tokenizer.eos_token_id)
                        prepared[arm].append((tokens, (1,)*len(tokens)))
    return items, parts, prepared, rejected


def route_prompts(parts, chosen, tok, *, include_prefix=True, retain_shape=False):
    output = []
    for (question, with_prefix, intro, pair, separator), found in zip(parts, chosen, strict=True):
        begin = (with_prefix if include_prefix else question) + intro
        fully = begin + pair[found] + [tok.eos_token_id]
        if retain_shape:
            mask = [1]*len(begin) + [int(found == 0)]*len(pair[0]) + [1]*len(separator) + [int(found == 1)]*len(pair[1]) + [1]
            sequence = begin + pair[0] + separator + pair[1] + [tok.eos_token_id]
            fully = [v if visible else tok.pad_token_id for v, visible in zip(sequence, mask, strict=True)]
        else:
            mask = [1]*len(fully)
        output.append((tuple(fully), tuple(mask)))
    return output


def compare_latency(items, parts, questions, full_prompts, candidate_bank, embed, readers, seed):
    """Warm preloaded readers, then time the same question batches through generation."""
    rng = Random(seed)
    indices = [i for i, (row, *_) in enumerate(items) if row["hops"] == 2 and not row["reverse"]]
    rng.shuffle(indices)
    results = []
    for width in (1, 4, 8):
        selected = [indices[i:i+width] for i in range(0, min(len(indices), 12*width)-width+1, width)]
        for size, reader in readers.items():
            tok = reader.tokenizer
            for batch_no, span in enumerate(selected):
                t = [questions[i] for i in span]
                options = candidate_bank[span]
                phase = {}
                for label in rng.sample(("both_with_prefix", "route_then_answer"), 2):
                    torch.cuda.synchronize()
                    start = perf_counter()
                    if label == "both_with_prefix":
                        prompts = [full_prompts[i] for i in span]
                    else:
                        scores = (embed(t, "query")[:, None, :] * options).sum(-1)
                        chosen = scores.argmax(-1).tolist()
                        prompts = route_prompts([parts[i] for i in span], chosen, tok, include_prefix=False)
                    answers = read_packed(reader, prompts)
                    torch.cuda.synchronize()
                    phase[label] = {"wall_ms": (perf_counter()-start)*1000,
                                    "exact": [exact(reply, items[i][1]) for reply, i in zip(answers, span, strict=True)]}
                results.append({"reader": size, "width": width, "warmup": batch_no == 0,
                                "cases": [items[i][0]["question"] for i in span], "phases": phase})
    return results


@torch.inference_mode()
def run(a):
    fixture = unpack(a.fixture)
    device = torch.cuda.get_device_name()
    readers = {}
    for name, (revision, weight) in READERS.items():
        path = a.assets / f"reader_{name}_{revision}"
        assert digest(path/"model.safetensors") == weight
        readers[name] = Reader(path)
    tok = readers["base"].tokenizer
    assert tok.get_vocab() == readers["large"].tokenizer.get_vocab()
    items, parts, tests, rejected = build_prompts(fixture, a.shard, a.shards, tok)
    rows = [row | {"readers": {}, "dense_endpoints": {}} for row, *_ in items]
    for name, reader in readers.items():
        for mode, cases in tests.items():
            answers = read_packed(reader, cases)
            for row, ans, (_, correct, other, *_), (token, _) in zip(rows, answers, items, cases, strict=True):
                assert len(token) == row["tokens"][mode]
                before = row["readers"].setdefault(name, {})
                before[mode] = {"exact": exact(ans, correct), "foil_answer": exact(ans, other),
                                "prompt_sha256": sha256(json.dumps(token, separators=(",", ":")).encode()).hexdigest()}
    embed = Encoder(a.assets/"encoder", 256)
    candidates = [p for _, _, _, _, p, _ in items]
    docs = list(dict.fromkeys(p for names in candidates for p in names))
    order = {p: i for i, p in enumerate(docs)}
    bank = embed(docs, "passage").clone()
    slots = torch.tensor([[order[p] for p in names] for names in candidates], device="cuda")
    blocks = bank[slots]
    paths = {}
    for mode in ("endpoints", "true_prefix", "cross_prefix", "bound_endpoints"):
        lookups = [r[3][mode] for r in items]
        query = embed(lookups, "query")
        score = (query[:, None, :] * blocks).sum(-1)
        choices = score.argmax(-1).tolist()
        paths[mode] = choices
        for row, item, got in zip(rows, items, choices, strict=True):
            row["dense_endpoints"][mode] = got == item[-1]
    timing = []
    if a.reference:
        original = unpack(a.reference)
        assert original["fixture_sha256"] == digest(a.fixture) and original["shard"] == a.shard
        assert original["rows"] == rows and original["rejected"] == rejected
        controls = {
            "query_route": (paths["endpoints"], True, False),
            "true_prefix_route": (paths["true_prefix"], True, False),
            "cross_prefix_route": (paths["cross_prefix"], True, False),
            "true_prefix_route_masked": (paths["true_prefix"], True, True),
            "true_prefix_route_endpoint_only": (paths["true_prefix"], False, False),
            "gold_route": ([item[-1] for item in items], True, False),
            "gold_route_endpoint_only": ([item[-1] for item in items], False, False),
        }
        for mode, (indices, earlier, masked) in controls.items():
            prompts = route_prompts(parts, indices, tok, include_prefix=earlier, retain_shape=masked)
            for name, reader in readers.items():
                predictions = read_packed(reader, prompts)
                for row, item, answer, found, (tokens, mask) in zip(rows, items, predictions, indices, prompts, strict=True):
                    if masked:
                        assert len(tokens) == row["tokens"]["true_prefix"]
                    result = row.setdefault("routed", {}).setdefault(name, {})
                    result[mode] = {"support_exact": found == item[-1], "exact": exact(answer, item[1]),
                                    "foil_answer": exact(answer, item[2]), "physical_tokens": len(tokens), "visible_tokens": sum(mask)}
        if a.profile:
            timing = compare_latency(items, parts, [r[3]["true_prefix"] for r in items], tests["true_prefix"], blocks,
                                     embed, readers, 51371 + a.shard)
            checks = {(row["question"], size): row for row in rows if not row["reverse"] for size in readers}
            for case in timing:
                name = case["reader"]
                matched = [checks[q, name] for q in case["cases"]]
                old = {"both_with_prefix": [r["readers"][name]["true_prefix"]["exact"] for r in matched],
                       "route_then_answer": [r["routed"][name]["true_prefix_route_endpoint_only"]["exact"] for r in matched]}
                for mode, flags in old.items():
                    case["phases"][mode]["prior_matches"] = sum(x == y for x, y in zip(flags, case["phases"][mode]["exact"], strict=True))
    return {"fixture_sha256": digest(a.fixture), "source_sha256": fixture["source_sha256"], "models": READERS,
            "model_revisions": unpack(a.assets/"revisions.json"), "device": device, "shard": a.shard,
            "shards": a.shards, "rejected": rejected, "rows": rows,
            "timing": timing,
            "reference_sha256": digest(a.reference) if a.reference else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("fixture", "assets", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("shard", "shards"):
        p.add_argument("--" + key, type=int, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--profile", action="store_true")
    a = p.parse_args()
    torch.set_float32_matmul_precision("highest")
    result = run(a)
    a.output.write_bytes(gzip.compress(json.dumps(result, separators=(",", ":"), allow_nan=False).encode(), mtime=0))


if __name__ == "__main__":
    main()
