#!/usr/bin/env python3
"""Measure decision parity, support-address skill, and timing of terminal attention."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median

from analyze_kam_native_musique import compare
from analyze_kam_oecd_revisions import paired_quarter_seed

MODES = ("full_fp32", "readout_fp32", "top1", "top4", "top16", "top64", "full_flash", "readout_flash")


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def observations(run):
    assert len(run["decisions"]) == len(run["correct"])
    return {"alias": [a for a, _ in run["correct"]], "record": [b for _, b in run["correct"]],
            "joint": [a and b for a, b in run["correct"]], "frames": run["decisions"]}


def reference(mode):
    return "full_flash" if mode.endswith("flash") else "full_fp32"


def procedural(runs):
    results = {}
    for panel in runs[0]["procedural"]:
        tested = [r["procedural"][panel] for r in runs]
        assert len({t["batch"] for t in tested}) == 1 and all(set(t["results"]) == set(MODES) for t in tested)
        overview = {}
        for mode in MODES:
            samples = [run["results"][mode] for run in tested]
            anchor = [run["results"][reference(mode)] for run in tested]
            probe = {}
            for record, check, raw in zip(samples, anchor, runs, strict=True):
                scores, control = map(observations, (record, check))
                pairs = list(zip(scores["frames"], control["frames"], strict=True))
                n = len(pairs)
                probe[str(raw["seed"])] = {"n": n, "counts": {k: sum(scores[k]) for k in ("alias", "record", "joint")},
                                           "decision_parity": sum(a == b for a, b in pairs),
                                           "joint_vs_reference": compare(scores["joint"], control["joint"], [[i] for i in range(n)]),
                                           "largest_logit_difference": record["largest_logit_difference"]}
            info = {"seeds": probe}
            if "timing" in samples[0]:
                fields = ("wall_ms", "stream_ms", "extra_peak_bytes")
                info["median_timing"] = {k: median(s["timing"][k] for s in samples) for k in fields}
                info["paired_median_speedup"] = median(o["timing"]["wall_ms"] / s["timing"]["wall_ms"]
                                                       for s, o in zip(samples, anchor, strict=True))
            if mode.startswith("top"):
                info["retained_attention_mass_by_seed"] = [s["per_head_retained"] for s in samples]
            overview[mode] = info
        results[panel] = {"batch": tested[0]["batch"], "fixture_hashes": [t["fixture_sha256"] for t in tested],
                          "modes": overview}
    return results


def archive(runs, domain):
    findings = {}
    for depth in (16, 48):
        by_mode = {}
        for mode in MODES:
            rows = {}
            for run in runs:
                for panel in run["archives"][domain]:
                    if panel["depth"] != depth:
                        continue
                    score = observations(panel["results"][mode])
                    for i, case in enumerate(panel["rows"]):
                        id_ = case["id"]
                        if domain == "eurostat":
                            id_ += "-" + str(int(id_.rsplit("-", 1)[1]) // 3)
                        key = run["seed"], id_, case["period"], panel["recode"]
                        assert key not in rows
                        rows[key] = {k: v[i] for k, v in score.items()}
            assert len(rows) == (3504 if domain == "dallas" else 3456)
            by_mode[mode] = rows
        found = {}
        for mode, rows in by_mode.items():
            old = by_mode[reference(mode)]
            assert rows.keys() == old.keys()
            found[mode] = {"records": len(rows), "joint_vs_reference": paired_quarter_seed({
                k: int(r["joint"]) - int(old[k]["joint"]) for k, r in rows.items()}),
                "decision_parity": sum(old[k]["frames"] == r["frames"] for k, r in rows.items()),
                "counts": {k: sum(r[k] for r in rows.values()) for k in ("alias", "record", "joint")}}
        findings[str(depth)] = found
    return findings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measurement", type=Path, nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    runs = sorted(map(read, a.measurement), key=lambda r: r["seed"])
    assert len({r["seed"] for r in runs}) == 3
    assert all(r["clock"] == runs[0]["clock"] and r["device"] == runs[0]["device"] and
               r["archives_sha256"] == runs[0]["archives_sha256"] for r in runs)
    data = {"clock": runs[0]["clock"], "device": runs[0]["device"], "archives_sha256": runs[0]["archives_sha256"],
            "models": {str(r["seed"]): {"reference_sha256": r["reference_sha256"], "weights_sha256": r["weights_sha256"]}
                       for r in runs}, "procedural": procedural(runs),
            "archives": {name: archive(runs, name) for name in ("dallas", "eurostat")}}
    a.output.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
