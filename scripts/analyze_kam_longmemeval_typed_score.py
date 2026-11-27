#!/usr/bin/env python3
"""Check semantic duration equivalence on exactly replayed LongMemEval prompts."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from analyze_kam_shared_serving import unpack


def audit(reference, repeated):
    checked, counts, formats = set(), defaultdict(Counter), Counter()
    for i, (base, run) in enumerate(zip(reference, repeated, strict=True)):
        assert run["shard"] == base["shard"] == i and run["device"] == base["device"] == "NVIDIA L4"
        assert all(run[k] == base[k] for k in ("fixture_sha256", "reference_sha256", "labels_sha256", "weights", "arms"))
        rows = {r["id"]: r["arms"] for r in base["rows"]}
        for case in run["rows"]:
            id_ = case["id"]
            assert id_ not in checked
            checked.add(id_)
            labels = set()
            for arm, observed in case["arms"].items():
                previous = rows[id_][arm]
                assert {k: v for k, v in observed.items() if k != "readers"} == {
                    k: v for k, v in previous.items() if k != "readers"}
                for size, item in observed["readers"].items():
                    assert {k: v for k, v in item.items() if k not in ("typed_kind", "typed_exact")} == previous["readers"][size]
                    labels.add(item["typed_kind"])
                    lexical, typed = item["em"], item["typed_exact"]
                    evidence = observed["marked_support"]
                    counts[f"{arm}/{size}"].update(
                        lexical=lexical, typed=typed, exact_under_either=lexical or typed,
                        typed_only=typed and not lexical, lexical_only=lexical and not typed,
                        joint_lexical=lexical and evidence, joint_under_either=(typed or lexical) and evidence,
                    )
            kind, = labels
            formats[kind] += 1
    assert len(checked) == 25 and formats == {"hours": 16, "month": 9}
    return {"fixture_sha256": reference[0]["fixture_sha256"], "questions": len(checked), "formats": formats,
            "arms": counts}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("measurement", "reference"):
        p.add_argument("--" + name, type=Path, nargs=4, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    archive, observed = [sorted(map(unpack, files), key=lambda item: item["shard"])
                         for files in (a.reference, a.measurement)]
    result = audit(archive, observed)
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
