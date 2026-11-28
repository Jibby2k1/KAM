#!/usr/bin/env python3
"""Measure causal dependence on distant passage text with length-matched controls."""
import argparse
import json
from pathlib import Path

from audit_kam_natural_data import stable
from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from prepare_kam_long_sparse import display, literal, load_locked_tokenizer, retrieval_segments

ARMS = ("flash", "protected", "tail-256")
EDITS = ("earlier", "background")
RUNS = tuple((s, a) for s in SEEDS for a in ARMS)


def read(path):
    return json.loads(path.read_text())


def replace(ids, at, replacement):
    return ids[:at] + replacement + ids[at+len(replacement):]


def prepare(root, reference, evidence, work):
    spans_file = evidence / "spans.json"
    spans = read(spans_file)
    manifest_file = reference / "manifest.json"
    assert spans["holdout_manifest_sha256"] == digest(manifest_file)
    inputs_file = reference / "inputs.json"
    assert read(manifest_file)["inputs_sha256"] == digest(inputs_file)
    tokenizer, _ = load_locked_tokenizer(root)
    data = read(inputs_file)["retrieval"]
    known = {row["id"]: row for row in read(reference / "retrieval.json")["cases"]}
    chosen = []
    panels = {edit: [] for edit in EDITS}
    for i in range(0, len(data), 2):
        left, right = data[i:i+2]
        annotations = spans["holdout"][i:i+2]
        assert (left["expected"], right["expected"]) == (0, 1) and left["id"] == right["id"]
        raw = known[left["id"]]
        r = annotations[0]
        assert r["earlier"] == annotations[1]["earlier"] and r["bridge"] == annotations[1]["bridge"]
        earlier, gold, foil = [tokenizer.decode(tokenizer.encode(display(raw[k]),
            add_special_tokens=False, truncation=True, max_length=256)) for k in ("earlier", "gold", "foil")]
        prefix, _ = retrieval_segments(tokenizer, raw, earlier, (gold, foil))
        background_start = len(tokenizer.encode(prefix, add_special_tokens=False))
        start, stop = max(r["earlier"][0], 64), r["earlier"][1]
        length = stop-start
        behind = 1536 + int(stable(left["id"])[:8], 16) % 768
        source = left["ids"][behind:behind+length]
        assert length > 0 and source == right["ids"][behind:behind+length]
        assert stop <= background_start and background_start+length < behind
        assert behind+length < min(min(x["gold"][0], x["foil"][0]) for x in annotations)
        modified = replace(left["ids"], start, source)
        bridge_removed = not literal(raw["anchor"], tokenizer.decode(modified[r["earlier"][0]:stop]))
        if r["bridge"][0] < 64 or not bridge_removed:
            continue
        chosen.append({"id": left["id"], "family": left["family"], "earlier": [start, stop],
                       "background": [background_start, background_start+length], "donor": [behind, behind+length]})
        for edit, position in (("earlier", start), ("background", background_start)):
            for prompt, ann in zip((left, right), annotations, strict=True):
                ids = replace(prompt["ids"], position, source)
                end = min(ann["gold"][0], ann["foil"][0])
                assert len(ids) == len(prompt["ids"]) == 4096 and ids[:64] == prompt["ids"][:64]
                assert ids[end:] == prompt["ids"][end:]
                assert ids[:position] == prompt["ids"][:position] and ids[position+length:] == prompt["ids"][position+length:]
                if edit == "background":
                    assert ids[start:stop] == prompt["ids"][start:stop]
                panels[edit].append(prompt | {"ids": ids})
    assert len(chosen) >= 240
    inputs_path = work / "ablation-inputs.json"
    write_json(inputs_path, panels)
    write_json(work / "ablation-manifest.json", {"inputs_sha256": digest(inputs_path),
        "spans_sha256": digest(spans_file), "original_inputs_sha256": digest(inputs_file),
        "holdout_manifest_sha256": digest(manifest_file), "questions": chosen})


def evaluate(root, source, work, index):
    from eval_kam_backbone_shortlist import load
    from eval_kam_long_sparse import retrieval_score
    from kam_live_token_factorial import GPU
    from kam_query_window_flash import WindowAttention

    seed, arm = RUNS[index]
    model, stack, projection, _, checkpoint, attention = load(root, source, work, seed, WindowAttention)
    fixture, inputs = (work / f"ablation-{kind}.json" for kind in ("manifest", "inputs"))
    metadata = read(fixture)
    assert metadata["inputs_sha256"] == digest(inputs) and metadata["holdout_manifest_sha256"] == digest(work / "manifest.json")
    panels = read(inputs)
    attention.mode = arm
    with attention.installed():
        outcomes = {edit: retrieval_score(model, stack, projection, {"retrieval": panels[edit]}) for edit in EDITS}
    write_json(work / f"ablation-{seed}-{arm}.json", {"seed": seed, "arm": arm, "gpu": GPU,
        "checkpoint_sha256": checkpoint, "fixture_sha256": digest(fixture), "outcomes": outcomes})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    prep = commands.add_parser("prepare")
    for name in ("root", "reference", "evidence", "work"):
        prep.add_argument(name, type=Path)
    run = commands.add_parser("evaluate")
    for name in ("root", "source", "work"):
        run.add_argument(name, type=Path)
    run.add_argument("index", type=int, choices=range(len(RUNS)))
    a = parser.parse_args()
    if a.mode == "prepare":
        prepare(a.root, a.reference, a.evidence, a.work)
    else:
        evaluate(a.root, a.source, a.work, a.index)
