#!/usr/bin/env python3
"""Select two equal-budget layer schedules on frozen development data."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json

PROTOCOL = "kam-layer-budget-v1"


def read(path):
    return json.loads(path.read_text())


def select(path):
    hashes, outputs = {}, []
    for seed in SEEDS:
        source = path / f"layer-calibration-{seed}.json"
        report = read(source)
        assert report["protocol"] == PROTOCOL and report["seed"] == seed
        outputs.append(report)
        hashes[source.stem] = digest(source)
    development = {}
    for dataset, count in (("books", 13), ("musique", 48)):
        units = [a["id"] for a in outputs[0]["results"][dataset]]
        assert len(set(units)) == count
        same = ("gpu", "spans_sha256", "pg_manifest_sha256")
        assert len({tuple(r[k] for k in same) for r in outputs}) == 1
        assert all([r["id"] for r in x["results"][dataset]] == units for x in outputs)
        points = [r for out in outputs for r in out["results"][dataset]]
        n = np.array([r["n_queries"] for r in points])[:, None]
        energy = np.array([r["values"]["output_energy"] for r in points])
        dense_evidence = np.array([r["values"]["dense_evidence"] for r in points])
        assert (energy > 0).all()
        values = {}
        for b in ("4", "8", "12"):
            sse, mass, evidence = (np.array([r["values"][key][b] for r in points]) for key in
                                   ("output_sse", "retained_mass", "retained_evidence"))
            assert sse.shape == mass.shape == evidence.shape == (len(points), 32)
            assert all(np.isfinite(a).all() and a.min() >= 0 for a in (sse, mass, evidence))
            assert (mass/n).max() < 1+1e-5 and (evidence <= dense_evidence+1e-7).all()
            values[b] = {"relative_output_sse": (sse/energy).mean(0), "retained_mass": (mass/n).mean(0),
                         "absolute_retained_evidence": (evidence/n).mean(0)}
        development[dataset] = values

    book_4, book_12 = (development["books"][a]["relative_output_sse"] for a in ("4", "12"))
    qa_4, qa_12 = (development["musique"][a]["absolute_retained_evidence"] for a in ("4", "12"))
    gains = {"output-selected": book_4-book_12, "evidence-selected": qa_12-qa_4}
    schedule = {}
    for arm, gain in gains.items():
        assert np.isfinite(gain).all()
        best = set(np.argsort(-gain, kind="stable")[:16])
        schedule[arm] = [12 if idx in best else 4 for idx in range(32)]
    schedule["alternating"] = [4, 12] * 16
    assert all(r.count(4) == r.count(12) == 16 for r in schedule.values())
    return {"protocol": PROTOCOL, "calibration_sha256": hashes, "spans_sha256": outputs[0]["spans_sha256"],
            "pg_manifest_sha256": outputs[0]["pg_manifest_sha256"],
            "checkpoints": {r["seed"]: r["checkpoint_sha256"] for r in outputs},
            "gpu": outputs[0]["gpu"], "schedule": schedule,
            "gains": {k: v.tolist() for k, v in gains.items()},
            "development": {k: {b: {f: a.tolist() for f, a in vals.items()} for b, vals in row.items()}
                            for k, row in development.items()}}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("work", type=Path)
    args = cli.parse_args()
    write_json(args.work/"layer-budget-selection.json", select(args.work))
