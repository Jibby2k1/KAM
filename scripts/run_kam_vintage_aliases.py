#!/usr/bin/env python3
"""Test compositional alias resolution with separate unknown and collision gates."""

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch

from audit_kam_natural_data import digest, stable, words
from prepare_kam_vintage_revisions import SERIES
from run_kam_hotpot_independent import read, save
from run_kam_natural_feasibility import Encoder

BASES = {
    "fit": (("in real terms", "in constant dollars", "adjusted for inflation", "using chained dollars", "in volume terms"),
            ("in nominal terms", "in current dollars", "without an inflation adjustment", "in market prices", "in face-value dollars")),
    "calibration": (("on a deflated basis", "on a chain-weighted basis", "with prices held constant"),
                    ("using nominal valuations", "using current prices", "before deflation")),
    "test": (("at fixed purchasing power", "using base-year prices", "after removing price changes", "at inflation-neutral valuations"),
             ("at the prices prevailing then", "in the dollars of that period", "without removing price changes", "at period prices")),
}
MEASURES = {
    "fit": (("gross domestic product", "GDP", "domestic production output", "national output"),
            ("personal consumption expenditures", "PCE", "household consumption", "consumption spending")),
    "calibration": (("domestic product", "economy-wide production", "total output of domestic goods and services"),
                    ("consumer spending", "personal outlays", "private consumer expenditure")),
    "test": (("domestic economic output", "aggregate value of domestic production", "overall domestic output", "aggregate domestic production"),
             ("consumer purchases of goods and services", "the value of household purchases", "consumption by households", "households' final consumption")),
}
TEMPLATES = {
    "fit": ("{measure} {basis}", "the figure for {measure}, reported {basis}"),
    "calibration": ("the archived {measure} figure, stated {basis}",),
    "test": ("the value listed for {measure} {basis}", "the recorded level, {basis}, of {measure}"),
}
UNKNOWN = {
    "calibration": ("the unemployment rate", "the count of new housing starts", "industrial production",
                    "corporate tax collections", "business equipment investment", "government wages",
                    "private inventories", "central bank currency reserves", "nonfarm payrolls",
                    "consumer price inflation", "public spending", "commercial credit"),
    "test": ("labor force participation", "disposable personal income", "import price inflation",
             "manufacturing capacity utilization", "real business investment", "the national median wage",
             "total residential construction", "household saving rather than spending", "corporate profits",
             "real federal defense expenditures", "interest income of households", "filed unemployment claims"),
}
CONTRADICTIONS = {
    "calibration": ("nominal and real {measure}", "{measure} given in current and constant dollars"),
    "test": ("both deflated and unadjusted {measure}", "{measure} in fixed-price and period-price units"),
}
CODES = {(0, 0): "ROUTPUT", (1, 0): "NOUTPUT", (0, 1): "RCON", (1, 1): "NCON"}


def nkey(text):
    return " ".join(words(text))


def make_cases(part):
    rows = []
    for price in range(2):
        for concept in range(2):
            if part == "fit" and (price, concept) == (1, 1):
                continue
            for basis in BASES[part][price]:
                for name in MEASURES[part][concept]:
                    for template in TEMPLATES[part]:
                        text = template.format(measure=name, basis=basis)
                        rows.append({"text": text, "status": "known", "price": price, "concept": concept,
                                     "code": CODES[price, concept], "basis_family": basis, "measure_family": name})
    if part == "fit":
        return rows
    for entity in UNKNOWN[part]:
        for template in ("{measure}", "the archived value for {measure}"):
            rows.append({"text": template.format(measure=entity), "status": "unknown"})
    for family in MEASURES[part]:
        for name in family:
            rows.append({"text": f"the numerical reading for {name}", "status": "unspecified_basis"})
            for template in CONTRADICTIONS[part]:
                rows.append({"text": template.format(measure=name), "status": "conflicting_basis"})
    for price in range(2):
        for basis in BASES[part][price]:
            rows.append({"text": f"both {MEASURES[part][0][0]} and {MEASURES[part][1][0]} {basis}",
                         "status": "conflicting_measure"})
    assert len({nkey(r["text"]) for r in rows}) == len(rows)
    return rows


def fit_factor(embeddings, labels, reg=0.035):
    x = embeddings.double()
    center = x.mean(0, keepdim=True)
    target = labels.double() * 2 - 1
    centered = x - center
    weights = torch.ones_like(target)
    for index in (0, 1):
        sel = labels.eq(index)
        weights[sel] = len(labels) / (2 * sel.sum().to(x.dtype))
    scaled = centered * weights.sqrt().unsqueeze(1)
    design = scaled @ scaled.T
    design.diagonal().add_(reg)
    coefficient = scaled.T @ torch.linalg.solve(design, target * weights.sqrt())
    return center, coefficient


def factor_outputs(vectors, axes):
    scores = torch.stack([((vectors.double() - center) @ coef).float() for center, coef in axes], -1)
    return scores.gt(0).long(), scores.abs().amin(-1)


def choose(scores, rows, positive, negative, cap):
    assert scores.isfinite().all()
    values = sorted(set(float(t) for t in scores))
    thresholds = [values[0] - 1e-5, *((x + y) / 2 for x, y in zip(values[:-1], values[1:], strict=True)),
                  values[-1] + 1e-5]
    accepted = lambda subset, t: sum(float(scores[i]) >= t for i in subset)
    feasible = [(accepted(positive, t), -accepted(negative, t), -t) for t in thresholds
                if accepted(negative, t) <= cap * len(negative)]
    assert feasible
    return -max(feasible)[2]


def calibrate(rows, proximity, factor_margin):
    known = [i for i, r in enumerate(rows) if r["status"] == "known"]
    unknown = [i for i, r in enumerate(rows) if r["status"] == "unknown"]
    ambiguous = [i for i, r in enumerate(rows) if r["status"] not in ("known", "unknown")]
    u = choose(proximity, rows, known, unknown, 0.10)
    entries = [i for i in known if proximity[i] >= u]
    collisions = [i for i in ambiguous if proximity[i] >= u]
    if collisions and entries:
        a = choose(factor_margin, rows, entries, collisions, 0.10)
    else:
        a = float(factor_margin.max()) + 1e-5 if collisions else -1.0
    return u, a


def assess(rows, predicted, gate, vocabulary, outputs):
    for idx, rec in enumerate(rows):
        resolved = vocabulary[predicted[idx]] if gate[idx] else None
        outputs[idx]["correct"] = resolved == rec.get("code")
        outputs[idx]["resolved"] = resolved


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--vintage-fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    started = perf_counter()
    torch.set_float32_matmul_precision("highest")
    cases = {part: make_cases(part) for part in ("fit", "calibration", "test")}
    aliases = {part: {nkey(r["text"]) for r in rows} for part, rows in cases.items()}
    assert not aliases["fit"] & aliases["test"] and not aliases["calibration"] & aliases["test"]
    assert all(r["code"] != "NCON" for r in cases["fit"])
    enc = Encoder(a.assets / "encoder", 128)
    vectors = {p: enc([r["text"] for r in rows], "query") for p, rows in cases.items()}
    canonical = enc([alias for items in SERIES.values() for alias in items[:2]], "passage")
    vocabulary = list(SERIES)
    axes = [fit_factor(vectors["fit"], torch.tensor([r[key] for r in cases["fit"]], device="cuda"))
            for key in ("price", "concept")]

    controls = {}
    for part in ("calibration", "test"):
        known_similarity = (vectors[part] @ vectors["fit"].T).amax(-1).cpu()
        factor_choice, factor_margin = factor_outputs(vectors[part], axes)
        direct = (vectors[part] @ canonical.T).reshape(len(vectors[part]), 4, 2).amax(-1)
        direct_choice = direct.argmax(-1).cpu()
        factored = torch.tensor([vocabulary.index(CODES[tuple(x)]) for x in factor_choice.cpu().tolist()])
        controls[part] = {"proximity": known_similarity, "margin": factor_margin.cpu(),
                          "dense": direct_choice, "factored": factored, "factors": factor_choice.cpu().tolist()}
    calib = controls["calibration"]
    u, c = calibrate(cases["calibration"], calib["proximity"], calib["margin"])
    fixture = read(a.vintage_fixture)
    addresses = {name: [r for r in fixture["rows"]["test"] if r["series"] == name] for name in SERIES}
    all_records = {}
    for part in ("calibration", "test"):
        snapshots = cases[part]
        results = [{**r, "arms": {}} for r in snapshots]
        estimates = controls[part]
        available = estimates["proximity"] >= u
        stable_factor = estimates["margin"] >= c
        options = {
            "dense": (estimates["dense"], torch.ones_like(available)),
            "factor": (estimates["factored"], torch.ones_like(available)),
            "factor_unknown_gate": (estimates["factored"], available),
            "factor_two_gates": (estimates["factored"], available & stable_factor),
        }
        for key, (choice, gate) in options.items():
            assess(snapshots, choice.tolist(), gate.tolist(), vocabulary, [r["arms"].setdefault(key, {}) for r in results])
        for i, row in enumerate(results):
            if row["status"] == "known":
                group = addresses[row["code"]][int(stable(row["text"])[:8], 16) % len(addresses[row["code"]])]
                earlier = next(q for q in group["questions"] if q["style"] == "canonical" and q["period"] == "earlier")
                row["grounded_fixture_group"] = group["id"]
                row["grounded_cutoff"] = earlier["cutoff"]
                row["record_id"] = earlier["gold"]
            row["factor_prediction"] = estimates["factors"][i]
            row["fit_proximity"] = float(estimates["proximity"][i])
            row["factor_margin"] = float(estimates["margin"][i])
        all_records[part] = results
    save(a.output, {"fixture_sha256": digest(a.vintage_fixture), "fit_counts": dict(Counter(r["code"] for r in cases["fit"])),
                    "admission_threshold": u, "ambiguity_threshold": c, "rows": all_records,
                    "device": torch.cuda.get_device_name(), "elapsed_seconds": perf_counter() - started})


if __name__ == "__main__":
    main()
