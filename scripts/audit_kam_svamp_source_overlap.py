#!/usr/bin/env python3
"""Audit exact normalized SVAMP prompts in the frozen adapter source streams."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from kam_adapter_preservation import digest

SOURCE_SHA256 = "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e"
PREFIX_WORDS = 8


def normalized(text):
    return re.findall(r"\w+", text.casefold())


def examine(root):
    source = root / "SVAMP.json"
    assert digest(source) == SOURCE_SHA256
    svamp = json.loads(source.read_text())
    blocked = defaultdict(list)
    short = []
    for row in svamp:
        words = normalized(row["Body"] + " " + row["Question"])
        if len(words) < PREFIX_WORDS:
            short.append(row["ID"])
        else:
            blocked[tuple(words[:PREFIX_WORDS])].append((row["ID"], words))
    assert len(svamp) == 1000
    manifest = json.loads((root / "corpus" / "manifest.json").read_text())
    findings = {}
    for domain in ("wiki", "code", "math"):
        key = f"{domain}-train"
        path = root / "corpus" / f"{key}.bin"
        assert digest(path) == manifest["corpus"][key]["sha256"]
        matches = set()
        for document in path.read_bytes().split(b"\n\x00\n"):
            words = normalized(document.decode("utf-8"))
            for start in range(len(words) - PREFIX_WORDS + 1):
                for ident, prompt in blocked.get(tuple(words[start:start+PREFIX_WORDS]), ()):
                    if words[start:start+len(prompt)] == prompt:
                        matches.add(ident)
        findings[domain] = {"matching_svamp_ids": sorted(matches), "source_sha256": digest(path)}
    out = root / "svamp-source-overlap.json"
    out.write_text(json.dumps({"dataset_sha256": SOURCE_SHA256, "n": len(svamp),
                               "min_prompt_words": PREFIX_WORDS, "short_ids": short,
                               "training_source": findings}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    examine(parser.parse_args().root)
