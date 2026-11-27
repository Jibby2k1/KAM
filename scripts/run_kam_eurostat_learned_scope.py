#!/usr/bin/env python3
"""Learn which of two clauses is the requested lookup and audit held-out wording."""

import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import re

import torch
from torch import nn

from analyze_kam_eurostat_text_routing import archive_bank, asof
from audit_kam_natural_data import digest
from benchmark_kam_eurostat_scope import matched_queries
from run_kam_eurostat_text_routing import read, write
from prepare_kam_eurostat_text_routing import parse_dates
from run_kam_natural_feasibility import Encoder, batches
from run_kam_hotpot_pool_scaling import Reader
from run_kam_musique_bindings import BASE_REVISION, BASE_WEIGHT

CUES = {
    "fit": (
        ("Please answer this now. {}", "This earlier question is included for comparison. {}"),
        ("This question is my current request. {}", "This is background only. {}"),
        ("For the reply, look up the following. {}", "The following is an example, rather than a request. {}"),
        ("{} Please provide that result.", "{} That was an earlier question and no response is needed."),
    ),
    "calibration": (
        ("Which value should I include? Please resolve this. {}", "The old draft listed this unrelated question. {}"),
        ("{} I need that result for the present reply.", "Use this as a side comparison only. {}"),
    ),
    "test": (
        ("The deliverable should cover this query. {}", "I am not requesting an answer to this one. {}"),
        ("{} That is the figure I need in the response.", "{} This is for context and can be left unanswered."),
        ("I would like to change my request to this. {}", "This used to be the question before I revised it. {}"),
        ("The note I need filled in now uses this lookup. {}", "Keep this as an example from my notes. {}"),
        ("Please include the result from this query. {}", "{} It would be a mistake to answer this old question."),
        ("What I actually need is an answer here. {}", "I mention this for comparison with the active lookup. {}"),
    ),
}
ROLE_CARDS = ("The user wants this question answered in the response.", "The user gives this only as an example or previous question.")


def cases(fixture, fold, mixed_ambiguous_cues=False):
    rows = []
    split = "test" if fold == "test" else "development"
    patterns = CUES[fold]
    for i, (group, (axis, (a, b))) in enumerate(matched_queries(fixture, split).items()):
        family = i % len(patterns)
        active, background = patterns[family]
        assert active.count("{}") == background.count("{}") == 1
        for reverse, role in product((False, True), ((1, 0), (0, 1), (0, 0), (1, 1))):
            if reverse and sum(role) != 1:
                continue
            chosen = (b, a) if reverse else (a, b)
            if mixed_ambiguous_cues and sum(role) != 1:
                next_family = (family + 1 + (i // len(patterns)) % (len(patterns) - 1)) % len(patterns)
                first, second = role
                templates = ((active if first else background), patterns[next_family][0 if second else 1])
            else:
                templates = tuple(active if flag else background for flag in role)
            parts = [template.format(q["text"]) for q, template in zip(chosen, templates, strict=True)]
            target = role.index(1) if sum(role) == 1 else 2
            kind = "single" if sum(role) == 1 else ("neither" if sum(role) == 0 else "both")
            assert all(parse_dates(text) == (q["observation"], q["cutoff"]) for text, q in zip(parts, chosen, strict=True))
            rows.append({"group": group, "axis": axis, "kind": kind, "family": family, "messages": parts,
                         "roles": role, "target": target, "base_queries": tuple(q["id"] for q in chosen),
                         "target_release": chosen[target]["gold"] if target < 2 else None})
    return rows


def cue_only(text):
    """Replace the one sentence with both an explicit source month and cutoff."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    agenda = []
    for i, part in enumerate(parts):
        try:
            parse_dates(part)
        except AssertionError:
            continue
        agenda.append(i)
    assert len(agenda) == 1
    parts[agenda[0]] = "The statistical lookup appears here."
    return " ".join(parts)


def learn(x, train):
    features = x.double()
    gold = torch.tensor(train, dtype=torch.float64)
    initial = nn.Linear(features.shape[1], 1, dtype=torch.float64)
    with torch.no_grad():
        initial.weight.zero_()
        initial.bias.zero_()
    optimiser = torch.optim.LBFGS(initial.parameters(), max_iter=90, tolerance_grad=1e-11, line_search_fn="strong_wolfe")

    def task():
        optimiser.zero_grad()
        score = initial(features).squeeze(-1)
        cost = nn.functional.binary_cross_entropy_with_logits(score, gold) + .02 * initial.weight.square().sum()
        cost.backward()
        return cost

    optimiser.step(task)
    return initial


def one(p, q, threshold):
    side = p >= threshold, q >= threshold
    return side.index(True) if sum(side) == 1 else 2


def relative(p, q, threshold):
    return int(q > p) if abs(q - p) > threshold else 2


def calibrate(rows, values, policy=one):
    if policy is relative:
        possible = sorted({0.0, *(abs(a - b) for a, b in values)})
    else:
        possible = sorted({v for pair in values for v in pair})
    penalties = []
    types = Counter(row["kind"] for row in rows)
    bounds = possible if policy is relative else [possible[0] - 1e-9, *possible, possible[-1] + 1e-9]
    for threshold in bounds:
        score = Counter()
        for r, (a, b) in zip(rows, values, strict=True):
            score[r["kind"]] += policy(a, b, threshold) == r["target"]
        balanced = sum(score[k] / n for k, n in types.items()) / len(types)
        penalties.append((balanced, score["neither"] + score["both"], threshold))
    return max(penalties)


@torch.inference_mode()
def t5_scores(reader, rows):
    prompts = [("Decide which question the user currently wants answered. Respond A only if it is the first, "
                "B only if it is the second, and C if the messages contain either no active question or two active questions. "
                f"First message: {a} Second message: {b} Answer:") for a, b in (r["messages"] for r in rows)]
    labels = [reader.tokenizer(c, add_special_tokens=False).input_ids for c in "ABC"]
    assert all(len(n) == 1 for n in labels)
    options = torch.tensor([n[0] for n in labels], device="cuda")
    output = []
    for group in batches(prompts, 48):
        feed = reader.tokenizer(group, padding=True, truncation=True, max_length=384, return_tensors="pt").to("cuda")
        start = torch.full((len(group), 1), reader.model.config.decoder_start_token_id, dtype=torch.long, device="cuda")
        logits = reader.model(**feed, decoder_input_ids=start, use_cache=False).logits[:, 0, options]
        output.extend(logits.argmax(-1).tolist())
    return output


def measure(rows, policies, reference):
    summary = {}
    samples = []
    for i, r in enumerate(rows):
        result = {}
        for name, ids in policies.items():
            chosen = ids[i]
            found = reference[r["base_queries"][chosen]] if chosen < 2 else None
            result[name] = {"choice": chosen, "release": found}
        samples.append({k: v for k, v in r.items() if k not in ("messages", "roles")} | {"observed": result})
    profiles = {name: [r for r in samples if r["kind"] == name] for name in ("single", "neither", "both")}
    profiles["address"] = [r for r in profiles["single"] if r["axis"] == "address"]
    profiles["cutoff"] = [r for r in profiles["single"] if r["axis"] == "cutoff"]
    for name, subset in profiles.items():
        groups = defaultdict(list)
        for r in subset:
            groups[r["group"]].append(r)
        summary[name] = {"n": len(subset), "groups": len(groups), "policies": {}}
        for policy in policies:
            exact = sum(r["target"] == r["observed"][policy]["choice"] for r in subset)
            release = sum(r["target_release"] == r["observed"][policy]["release"] for r in subset)
            group_count = sum(all(r["target_release"] == r["observed"][policy]["release"] for r in rows) for rows in groups.values())
            summary[name]["policies"][policy] = {"correct_roles": exact, "exact_release_or_refusal": release,
                                               "complete_groups": group_count,
                                               "declined": sum(r["observed"][policy]["choice"] == 2 for r in subset)}
    return summary, samples


def run(a):
    torch.set_float32_matmul_precision("highest")
    fixture = read(a.fixture)
    measured = read(a.ranked)
    fx = digest(a.fixture)
    assert measured["fixture_sha256"] == fx and measured["indexed_records"] == len(fixture["corpus"])
    assert measured["model_revisions"] == read(a.assets/"revisions.json")
    groups = {key: cases(fixture, key, a.mixed_ambiguous_cues and key != "fit") for key in CUES}
    if a.cue_only:
        for split in groups.values():
            for row in split:
                row["messages"] = [cue_only(part) for part in row["messages"]]
    rendered = list(dict.fromkeys(part for r in groups.values() for slot in r for part in slot["messages"]))
    by_string = {v: i for i, v in enumerate(rendered)}
    encoder = Encoder(a.assets/"encoder", 128)
    embeddings = encoder(rendered, "query").cpu().clone()
    definitions = encoder(list(ROLE_CARDS), "passage").cpu().clone()
    del encoder
    features = {name: embeddings[torch.tensor([[by_string[t] for t in r["messages"]] for r in rows])]
                for name, rows in groups.items()}
    training = [role for row in groups["fit"] for role in row["roles"]]
    learned = learn(features["fit"].flatten(0, 1), training)
    scored = {
        "frozen_prototype": {k: (m @ (definitions[0]-definitions[1])).tolist() for k, m in features.items()},
        "trained_clause_score": {k: learned(m.double()).squeeze(-1).detach().tolist() for k, m in features.items()},
    }
    threshold = {name: calibrate(groups["calibration"], scores["calibration"]) for name, scores in scored.items()}
    reference = {}
    store = archive_bank(fixture["corpus"])
    for split, rows in fixture["questions"].items():
        for q, row in zip(rows, measured["sections"][split]["rows"], strict=True):
            assert q["id"] == row["id"]
            address = fixture["corpus"][row["dense_docs"][0]]
            obs, date = parse_dates(q["text"])
            reference[q["id"]] = asof(store, address["indicator"], address["country"], obs, date)
    seen = groups["test"]
    policies = {name: [one(a, b, threshold[name][2]) for a, b in scores["test"]] for name, scores in scored.items()}
    if a.force_diagnostics:
        for name, values in scored.items():
            policies[name + "_forced"] = [int(b > a) for a, b in values["test"]]
    if a.relative_abstention:
        for name, values in scored.items():
            key = name + "_relative"
            threshold[key] = calibrate(groups["calibration"], values["calibration"], relative)
            policies[key] = [relative(a, b, threshold[key][2]) for a, b in values["test"]]
    policies.update(first_clause=[0]*len(seen), last_clause=[1]*len(seen), exact_scope=[r["target"] for r in seen])
    path = a.assets/f"reader_base_{BASE_REVISION}"
    assert digest(path/"model.safetensors") == BASE_WEIGHT
    policies["flan_t5_choice"] = t5_scores(Reader(path), seen)
    summaries, numerical = measure(seen, policies, reference)
    return {"fixture_sha256": fx, "ranked_sha256": digest(a.ranked), "cue_families": {k: len(v) for k, v in CUES.items()},
            "split_sizes": {k: len(v) for k, v in groups.items()}, "device": torch.cuda.get_device_name(),
            "cue_only": a.cue_only, "mixed_ambiguous_cues": a.mixed_ambiguous_cues,
            "calibration": threshold, "reader_revision": BASE_REVISION, "reader_sha256": BASE_WEIGHT, "outcomes": summaries,
            "rows": numerical}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("fixture", "ranked", "assets", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--cue-only", action="store_true")
    p.add_argument("--force-diagnostics", action="store_true")
    p.add_argument("--mixed-ambiguous-cues", action="store_true")
    p.add_argument("--relative-abstention", action="store_true")
    a = p.parse_args()
    write(a.output, run(a))


if __name__ == "__main__":
    main()
