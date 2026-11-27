#!/usr/bin/env python3
"""Compare curriculum effects between radial and dot-product attention."""

import argparse
import json
from pathlib import Path
from statistics import mean

from analyze_kam_radial_curriculum_order import ARMS, cases, run
from analyze_kam_sentence_verifier import paired


def inspect(radial, dot):
    assert all(radial[key] == dot[key] for key in
               ("seed", "device", "config", "banks", "initial_shared_parameter_sha256", "steps"))
    assert radial["execution"] == dot["execution"] == "eager"
    assert radial["geometry"] == "radial" and dot["geometry"] == "dot"
    selected = {p["geometry"]: run(p) for p in (radial, dot)}
    steps = selected["radial"].keys()
    for step in steps:
        assert all(selected["radial"][step][k] == selected["dot"][step][k] for k in ("tokens", "depth_updates"))
    cross = {}
    for step in steps:
        data = {}
        for arm in ARMS:
            left = radial["policies"][arm]["held_out"][step]["outcomes"]
            right = dot["policies"][arm]["held_out"][step]["outcomes"]
            assert left.keys() == right.keys()
            for d, lhs in left.items():
                assert lhs["fixture_sha256"] == right[d]["fixture_sha256"]
                y, x = cases(lhs)["joint"], cases(right[d])["joint"]
                data.setdefault(d, {})[arm] = {"radial_joint": mean(y), "dot_joint": mean(x),
                                               "paired_radial_minus_dot": paired(y, x)}
        cross[step] = data
    return {"seed": radial["seed"], "by_geometry": selected, "paired_cross_geometry": cross}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dot", "radial"):
        p.add_argument("--" + name, type=Path, nargs=3, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    analyses = [sorted([json.loads(path.read_text()) for path in paths], key=lambda m: m["seed"])
                for paths in (a.radial, a.dot)]
    rows = [inspect(r, d) for r, d in zip(*analyses, strict=True)]
    result = {"seeds": {r["seed"]: r for r in rows}, "device": analyses[0][0]["device"],
              "updates": analyses[0][0]["steps"], "config": analyses[0][0]["config"]}
    a.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
