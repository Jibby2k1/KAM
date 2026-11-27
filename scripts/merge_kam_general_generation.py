#!/usr/bin/env python3
"""Merge the three generation outputs after the KAM GPU shard array."""
import argparse
import json
from pathlib import Path

SHARDS = {"humaneval": (3, 164), "mbpp": (4, 500), "gsm8k": (9, 1319)}


def merge(root):
    for task, (shards, expected) in SHARDS.items():
        parts = [
            json.loads((root / f"{task}-T-KAM-F-shard{shard}.json").read_text())
            for shard in range(shards)
        ]
        metadata = parts[0]["metadata"]
        assert all(part["metadata"] == metadata for part in parts)
        rows = sorted((row for part in parts for row in part["records"]), key=lambda row: row["index"])
        assert [row["index"] for row in rows] == list(range(expected))
        final = root / f"{task}-T-KAM-F.json"
        temporary = final.with_suffix(".tmp")
        temporary.write_text(json.dumps({"metadata": metadata, "records": rows}))
        temporary.replace(final)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    merge(parser.parse_args().root)
