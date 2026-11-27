#!/usr/bin/env python3
"""Audit unit-invariant clock features and the cost of discarding absolute scale."""

import argparse
import json
from pathlib import Path
from statistics import mean, median

from analyze_kam_archive_clock_encoding import split_inputs
from analyze_kam_external_clock_transfer import indexed
from analyze_kam_native_musique import compare
from analyze_kam_oecd_revisions import paired_quarter_seed

GEOMETRIES = ("dot", "radial")
REFERENCES = ("linear", "signed_log")
REPERTOIRE = ("episode_scale", "ordinal")
RELATIVE_LOGS = ("log_episode_max", "log_episode_mean")
EVAL = ("16", "48", "128", "512")


def measure(runs, arm, modes):
    procedural, indices = {}, {}
    reference = "signed_log"
    for encoding in modes:
        key = f"{arm}/{encoding}"
        procedure = {}
        for kind in ("original", "corrections"):
            values = {}
            for d in EVAL:
                per_seed = {}
                for seed, record in runs.items():
                    a = record["results"][f"{arm}/{reference}"]["procedural"][kind][d]
                    b = record["results"][key]["procedural"][kind][d]
                    assert a["fixture_sha256"] == b["fixture_sha256"]
                    old, new = [list(map(all, r["outcomes"])) for r in (a, b)]
                    per_seed[str(seed)] = {"n": len(new), "correct": sum(new),
                                           "vs_signed_log": compare(new, old, [[i] for i in range(len(new))])}
                values[d] = {"mean_joint": mean(r["correct"]/r["n"] for r in per_seed.values()), "seeds": per_seed}
            procedure[kind] = values
        procedural[encoding] = procedure | {"median_fit_seconds": median(
            r["results"][key]["fit"]["seconds"] for r in runs.values())}
        for era in ("earlier", "later", "external", "external_clock_times_3"):
            for depth in (16, 48):
                indices[encoding, era, depth] = indexed(runs, arm, encoding, era, depth)
    result = {}
    for (encoding, name, depth), rows in indices.items():
        other = indices[reference, name, depth]
        assert rows.keys() == other.keys()
        record = {"correct": sum(r["joint"] for r in rows.values()), "n": len(rows),
                  "vs_signed_log": paired_quarter_seed({i: int(rows[i]["joint"])-int(other[i]["joint"]) for i in rows})}
        if name == "external_clock_times_3":
            plain = indices[encoding, "external", depth]
            assert rows.keys() == plain.keys()
            record["exact_scope_replays"] = sum(x == plain[i] for i, x in rows.items())
            record["unit_scale_effect"] = paired_quarter_seed({i: int(rows[i]["joint"])-int(plain[i]["joint"]) for i in rows})
        result[f"{encoding}/{name}/{depth}"] = record
    return {"procedures": procedural, "archives": result}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for domain in ("corrections", "archive-shaped", "mixed"):
        for variant in ("", "reference-"):
            p.add_argument("--" + variant + domain, nargs=3, type=Path)
    p.add_argument("--relative-log", action="store_true")
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    modes = REFERENCES + (RELATIVE_LOGS if a.relative_log else REPERTOIRE)
    data, matched = {}, 0
    for domain in ("corrections", "archive_shaped", "mixed"):
        paths = getattr(a, domain)
        if not paths:
            continue
        past = getattr(a, "reference_"+domain)
        assert past is not None
        raw, baseline = [split_inputs(m, domain) for m in (paths, past)]
        for seed, value in raw.items():
            assert value["archive_sha256"] == baseline[seed]["archive_sha256"]
            for arm in GEOMETRIES:
                for key in REFERENCES:
                    name = f"{arm}/{key}"
                    x, y = [r["results"][name] for r in (value, baseline[seed])]
                    assert all(x[k] == y[k] for k in ("initial_sha256", "procedural", "archive"))
                    matched += 1
        encoded = {m: max(r["external_feature_max_delta"][m] for r in raw.values()) for m in modes}
        if not a.relative_log:
            assert encoded["ordinal"] == 0
        data[domain] = {"external_feature_max_delta": encoded,
                        "scopes": {arm: measure(raw, arm, modes) for arm in GEOMETRIES}}
    result = {"conditions": data, "exact_reference_fits": matched, "encodings": modes}
    a.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
