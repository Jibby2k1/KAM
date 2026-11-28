#!/usr/bin/env python3
"""Evaluate which of two public publication clocks a model follows."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from audit_kam_natural_data import stable
from kam_adapter_preservation import digest, key
from kam_math_kl_protocol import test_runs, write_json

PROTOCOL = "kam-bisource-release-clock-v1"
PROJECTS = {"numpy": "numpy/numpy", "pandas": "pandas-dev/pandas", "matplotlib": "matplotlib/matplotlib",
            "pydantic": "pydantic/pydantic", "pytest": "pytest-dev/pytest", "ruff": "astral-sh/ruff"}
CLOCKS = ("github", "pypi")
LETTERS = "ABCD"
YEAR_START = 2019
YEAR_STOP = 2026
NONE = "no_eligible_release"


def timestamp(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def format_date(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def observed_history(source, project, repo):
    github = source / f"{project}-github.json"
    pypi = source / f"{project}-pypi.json"
    index = json.loads(pypi.read_text())
    assert index["info"]["name"].lower() == project
    versions = index["releases"]
    entries = {}
    for release in json.loads(github.read_text()):
        tag = release["tag_name"].removeprefix("v")
        url_prefix = f"https://github.com/{repo}/releases/tag/"
        assert release["html_url"].lower().startswith(url_prefix.lower())
        if release["draft"] or release["prerelease"] or not re.fullmatch(r"\d+\.\d+\.\d+", tag):
            continue
        files = [f for f in versions.get(tag, []) if not f["yanked"]]
        if not files or not release["published_at"]:
            continue
        git = timestamp(release["published_at"])
        registry = min(timestamp(f["upload_time_iso_8601"]) for f in files)
        if not all(YEAR_START <= datetime.fromtimestamp(t, timezone.utc).year < YEAR_STOP for t in (git, registry)):
            continue
        assert tag not in entries
        entries[tag] = {"version": tag, "github": git, "pypi": registry, "release_url": release["html_url"]}
    return list(entries.values()), {"github_sha256": digest(github), "pypi_sha256": digest(pypi), "repo": repo}


def latest(rows, clock, cutoff):
    eligible = [row for row in rows if row[clock] <= cutoff]
    ordered = sorted(eligible, key=lambda row: row[clock], reverse=True)
    if len(ordered) > 1 and ordered[0][clock] == ordered[1][clock]:
        return None
    return ordered[0]["version"] if ordered else NONE


def cases_for(project, history):
    candidates = {"flip": {}, "agree": {}}
    for row in history:
        low, high = sorted(row[clock] for clock in CLOCKS)
        for category, cutoff in (("flip", (low + high) // 2), ("agree", high + 1)):
            if category == "flip" and high - low < 3600:
                continue
            gold = {clock: latest(history, clock, cutoff) for clock in CLOCKS}
            if None in gold.values() or NONE in gold.values():
                continue
            if (gold["github"] != gold["pypi"]) != (category == "flip"):
                continue
            if row["version"] not in gold.values():
                continue
            population = tuple(sorted(set(gold.values())))
            # Compare fixed labels of two or three versions close to the cutoff.
            selected = [r for r in history if r["version"] in population]
            adjacent = sorted((r for r in history if r["version"] not in population),
                              key=lambda r: (min(abs(r[c] - cutoff) for c in CLOCKS), r["version"]))
            selected.extend(adjacent[:3 - len(selected)])
            assert len(selected) == 3
            if not any(r[c] > cutoff for r in selected for c in CLOCKS):
                continue
            ident = stable(f"{PROTOCOL}:{project}:{category}:{cutoff}")
            cases = {"id": ident, "project": project, "category": category, "cutoff": cutoff,
                     "records": sorted(selected, key=lambda r: stable(ident + r["version"])), "gold": gold}
            assert all(latest(cases["records"], c, cutoff) == gold[c] for c in CLOCKS)
            candidates[category][ident] = cases
    cases = []
    for variants in candidates.values():
        cases.extend(variants[k] for k in sorted(variants)[:12])
    before = min(r[c] for r in history for c in CLOCKS) - 3600
    outset = sorted(history, key=lambda r: min(r[c] for c in CLOCKS))[:3]
    cases.append({"id": stable(f"{PROTOCOL}:{project}:unavailable"), "project": project,
                  "category": "unavailable", "cutoff": before, "records": outset,
                  "gold": {c: NONE for c in CLOCKS}})
    return cases


def prepare(source, output):
    groups, hashes, panel = {}, {}, []
    for project, repo in PROJECTS.items():
        history, recorded = observed_history(source, project, repo)
        assert len(history) >= 3, project
        cases = cases_for(project, history)
        groups[project] = {"history": len(history), "cases": Counter(row["category"] for row in cases)}
        hashes[project] = recorded
        panel.extend(cases)
    output.mkdir(parents=True, exist_ok=True)
    file = output / "panel.json"
    write_json(file, panel)
    write_json(output / "manifest.json", {"protocol": PROTOCOL, "panel_sha256": digest(file),
               "source_hashes": hashes, "groups": groups, "clocks": CLOCKS,
               "cases": [{k: r[k] for k in ("id", "category", "project")} for r in panel]})


def prompts(tokenizer, panel):
    for case in panel:
        cutoff = format_date(case["cutoff"])
        records = [r["version"] for r in case["records"]] + [NONE]
        details = {r["version"]: f"Version {r['version']} | GitHub {format_date(r['github'])} | "
                   f"PyPI {format_date(r['pypi'])}" for r in case["records"]}
        details[NONE] = "No eligible release"
        for clock in CLOCKS:
            clock_name = {"github": "GitHub", "pypi": "PyPI"}[clock]
            allowed = [r["version"] for r in case["records"] if r[clock] <= case["cutoff"]] or [NONE]
            assert latest(case["records"], clock, case["cutoff"]) == case["gold"][clock]
            for offset in range(len(records)):
                order = records[offset:] + records[:offset]
                choices = "\n".join(f"{label}. {details[v]}" for label, v in zip(LETTERS, order, strict=True))
                message = (f"Read this snapshot of matched release records for {case['project']}. "
                           f"Which listed version has the latest {clock_name} timestamp at or before {cutoff}? "
                           f"Use the {clock_name} column. Reply with A, B, C, or D.\n\n" + choices + "\n\nChoice:")
                text = tokenizer.apply_chat_template([{"role": "user", "content": message}],
                                                     tokenize=False, add_generation_prompt=True)
                row = {"id": case["id"], "project": case["project"], "category": case["category"],
                       "clock": clock, "offset": offset, "gold": order.index(case["gold"][clock]),
                       "eligible": [i for i, v in enumerate(order) if v in allowed]}
                yield row, tokenizer.encode(text, add_special_tokens=False)


def evaluate(root, work, index):
    import torch
    from eval_kam_adapter_preservation import load_checkpoint
    from eval_kam_pretrained import CONTEXT, next_logits, pad

    selection = work / "selection.json"
    settings = json.loads(selection.read_text())
    assert settings["selected_recipe"] == "lower_lr_kl05" and settings["selected_kl_weight"] == .5
    run = test_runs(settings["selected_recipe"])[index]
    manifest = work / "manifest.json"
    fixture = work / "panel.json"
    descriptor = json.loads(manifest.read_text())
    assert descriptor["protocol"] == PROTOCOL and descriptor["panel_sha256"] == digest(fixture)
    model, tokenizer, stack, metadata = load_checkpoint(root, *run)
    gpu = torch.cuda.get_device_name()
    assert gpu == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
    docs = list(prompts(tokenizer, json.loads(fixture.read_text())))
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    assert all(len(v) == 1 for v in ids) and len(set(v[0] for v in ids)) == len(LETTERS)
    assert max(len(v) for _, v in docs) <= CONTEXT
    vocab = [v[0] for v in ids]
    records = []
    for start in range(0, len(docs), 16):
        batch = docs[start:start+16]
        x, mask = pad([inp for _, inp in batch], tokenizer.pad_token_id)
        logits, _ = next_logits(model, stack, x, mask)
        for (record, inp), scores in zip(batch, logits[:, vocab].tolist(), strict=True):
            direct = max(range(len(LETTERS)), key=scores.__getitem__)
            eligible = max(record["eligible"], key=scores.__getitem__)
            records.append({**record, "tokens": len(inp), "direct": direct, "restricted": eligible,
                            "logits": scores})
    write_json(work / f"clock-{key(*run)}.json", {"metadata": {**metadata, "gpu": gpu,
               "selection_sha256": digest(selection), "protocol": PROTOCOL,
               "manifest_sha256": digest(manifest)}, "records": records})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("source", type=Path)
    prep.add_argument("output", type=Path)
    gpu = commands.add_parser("evaluate")
    gpu.add_argument("root", type=Path)
    gpu.add_argument("work", type=Path)
    gpu.add_argument("index", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.output)
    else:
        evaluate(args.root, args.work, args.index)
