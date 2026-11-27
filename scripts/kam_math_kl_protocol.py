"""Locked fixtures and numerical scoring for the development-selected KL comparison."""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from kam_adapter_preservation import ARMS, LOW_RATE_KL_RECIPE, RUNS, SEEDS, digest

PROTOCOL = "kam-math-kl-selection-v1"
RECIPES = ("lower_lr", LOW_RATE_KL_RECIPE, "lower_lr_kl")
DEV_RUNS = tuple((seed, recipe, arm) for recipe in RECIPES for seed in SEEDS for arm in ARMS)
MARKERS = {"hash": "####", "answer": "Answer:"}
BUDGET = 384
BATCH_SIZE = 64
SOURCES = {
    "gsm8k-train.jsonl": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465",
    "gsm8k-test.jsonl": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    "math500-test.jsonl": "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
    "SVAMP.json": "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e",
}
UINT_OR_GROUPED = r"(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)"
NUMBER = rf"-?{UINT_OR_GROUPED}(?:\.[0-9]+)?"


def patterns(marker):
    line = rf"^[ \t]*{re.escape(marker)}[ \t]+({NUMBER})[ \t]*[.!]?[ \t]*"
    return (
        re.compile(line + r"(?:\n|\Z)", re.IGNORECASE | re.MULTILINE),
        re.compile(line + r"\n", re.IGNORECASE | re.MULTILINE),
    )


def score(response, gold, parser):
    matches = parser.findall(response)
    if not matches:
        return False, False
    return True, Decimal(matches[-1].replace(",", "")) == gold


def prompt(problem, marker):
    return f"{problem}\nSolve the problem. End your answer on a new line using '{marker} ' followed by the final number."


def test_runs(recipe):
    if recipe not in RECIPES:
        raise ValueError(f"Unexpected development-selected recipe {recipe}")
    return (*((seed, recipe, arm) for seed in SEEDS for arm in ARMS), RUNS[-1])


def normalized(text):
    return tuple(re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold()))


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)


def fixture_rows(root):
    for name, sha in SOURCES.items():
        if digest(root / name) != sha:
            raise ValueError(f"Unexpected digest for {name}")
    train, old_test, math = (
        [json.loads(line) for line in (root / name).read_text().splitlines()]
        for name in ("gsm8k-train.jsonl", "gsm8k-test.jsonl", "math500-test.jsonl")
    )
    svamp = json.loads((root / "SVAMP.json").read_text())
    assert (len(train), len(old_test), len(math), len(svamp)) == (7473, 1319, 500, 1000)
    prior = {normalized(row["question"]) for row in old_test}
    prior.update(normalized(row["Body"] + " " + row["Question"]) for row in svamp)
    train_prompts = [normalized(row["question"]) for row in train]
    counts = Counter(train_prompts)
    elig = [i for i, words in enumerate(train_prompts) if counts[words] == 1 and words not in prior]
    order = sorted(elig, key=lambda i: hashlib.sha256(f"{PROTOCOL}:dev:{i}".encode()).digest())
    dev = []
    for i in sorted(order[:512]):
        row = train[i]
        gold = row["answer"].rpartition("#### ")[-1].strip().replace(",", "")
        assert re.fullmatch(r"-?[0-9]+", gold)
        dev.append({"id": f"train:{i}", "problem": row["question"], "answer": int(gold)})
    blocked = prior | set(train_prompts)
    eligible_test = [r for r in math if re.fullmatch(r"-?[0-9]+", r["answer"])]
    test_prompts = Counter(normalized(r["problem"]) for r in eligible_test)
    test = [
        {"id": row["unique_id"], "problem": row["problem"], "answer": int(row["answer"]),
         "subject": row["subject"], "level": row["level"]}
        for row in eligible_test
        if (words := normalized(row["problem"])) not in blocked and test_prompts[words] == 1
    ]
    assert len(dev) == 512 and len(test) == len({r["id"] for r in test})
    return {"dev": dev, "test": test}, len(eligible_test)


def prepare(source, output):
    fixtures, eligible_test = fixture_rows(source)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"protocol": PROTOCOL, "sources": SOURCES, "math500_integer_eligible": eligible_test,
                "fixtures": {}, "markers": MARKERS, "budget": BUDGET, "batch_size": BATCH_SIZE,
                "recipes_in_tie_break_order": RECIPES,
                "selection_rule": "Maximize exact answers under both markers, pooled over both arms and all three seeds"}
    for split, rows in fixtures.items():
        path = output / f"{split}.json"
        write_json(path, rows)
        manifest["fixtures"][split] = {"sha256": digest(path), "n": len(rows),
                                       "ids": [r["id"] for r in rows]}
    write_json(output / "manifest.json", manifest)


def audit(root, fixtures, output):
    manifest = json.loads((root / "corpus" / "manifest.json").read_text())
    prefixes = defaultdict(list)
    for split in ("dev", "test"):
        for row in json.loads((fixtures / f"{split}.json").read_text()):
            words = normalized(row["problem"])
            width = min(8, len(words))
            assert width > 0
            prefixes[width, words[:width]].append((split, row["id"], words))
    widths = sorted({width for width, _ in prefixes})
    results = {}
    for domain in ("wiki", "code", "math"):
        filename = f"{domain}-train"
        path = root / "corpus" / f"{filename}.bin"
        sha = digest(path)
        assert sha == manifest["corpus"][filename]["sha256"]
        matches = {"dev": set(), "test": set()}
        for document in path.read_bytes().split(b"\n\x00\n"):
            words = normalized(document.decode("utf-8"))
            for width in widths:
                for index in range(len(words) - width + 1):
                    for split, ident, needle in prefixes.get((width, words[index:index+width]), ()):
                        if words[index:index+len(needle)] == needle:
                            matches[split].add(ident)
        results[domain] = {"sha256": sha, "matches": {split: sorted(ids) for split, ids in matches.items()}}
    write_json(output, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_cli = subparsers.add_parser("prepare")
    prepare_cli.add_argument("source", type=Path)
    prepare_cli.add_argument("output", type=Path)
    audit_cli = subparsers.add_parser("audit")
    audit_cli.add_argument("root", type=Path)
    audit_cli.add_argument("fixtures", type=Path)
    audit_cli.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.output)
    else:
        audit(args.root, args.fixtures, args.output)
