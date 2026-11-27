#!/usr/bin/env python3
"""Verify shared inference fixtures and summarize paired kernel measurements."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import median


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def audit(records):
    cross = defaultdict(list)
    devices = defaultdict(list)
    for r in records:
        cross[r["seed"]].append(r)
        devices[r["device"]].append(r)
    assert len(cross) == len(devices) == 3 and all(len(items) == 3 for items in cross.values())
    common_reference_matches = 0
    for rows in cross.values():
        assert len({r["weights_sha256"] for r in rows}) == len({r["reference_sha256"] for r in rows}) == 1
        for geo, element in rows[0]["geometries"].items():
            for key, panel in element["panels"].items():
                trials = [r["geometries"][geo]["panels"][key] for r in rows]
                assert {t["fixture_sha256"] for t in trials} == {panel["fixture_sha256"]}
                before = panel["arms"]["eager/full"]["decisions"]
                for test in trials[1:]:
                    common_reference_matches += sum(x == y for x, y in zip(test["arms"]["eager/full"]["decisions"], before, strict=True))

    result = {}
    for device, rows in devices.items():
        assert len(rows) == 3 and len({r["torch"] for r in rows}) == 1
        controls = {}
        for geometry, control in rows[0]["geometries"].items():
            groups = [r["geometries"][geometry] for r in rows]
            assert not any(p["unsupported"] for p in groups)
            output = {}
            for name, panel in control["panels"].items():
                traces = [r["panels"][name]["arms"] for r in groups]
                for row in traces:
                    assert row.keys() == panel["arms"].keys()
                    baseline = row["eager/full"]["decisions"]
                    for values in row.values():
                        assert len(values["decisions"]) == len(baseline) == 64
                        assert values["matches_eager"] == sum(x == y for x, y in zip(values["decisions"], baseline, strict=True))
                alternatives = {}
                for tag in panel["arms"]:
                    vals = [r[tag] for r in traces]
                    alternatives[tag] = {"tested": 64*len(vals), "paired_exact_decisions": sum(v["matches_eager"] for v in vals),
                                         "worst_absolute_logit_error": max(v["max_logit_error"] for v in vals),
                                         "wall_ms": median(v["timing"]["wall_ms"] for v in vals),
                                         "stream_ms": median(v["timing"]["stream_ms"] for v in vals),
                                         "extra_peak_bytes": median(v["timing"]["extra_peak_bytes"] for v in vals)}
                output[name] = {"batch": panel["batch"], "arms": alternatives}
            controls[geometry] = output
        result[device] = controls
    return {"identical_eager_across_device_pairs": common_reference_matches, "device_results": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, nargs=9, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(list(map(load, a.inputs)))
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
