#!/usr/bin/env python3
"""Audit exact as-of outcomes and separately calibrate open-set and collision gates."""

import argparse
from bisect import bisect_right
from collections import defaultdict
from hashlib import sha256
import gzip
import json
from pathlib import Path

import numpy as np

from analyze_kam_bright_transfer import group_queries, limits, resample
from prepare_kam_eurostat_text_routing import COUNTRIES, NAMES, parse_dates, registry_country, registry_indicator


def read(path):
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes())


def archive_bank(corpus):
    history = defaultdict(list)
    for i, r in enumerate(corpus):
        history[r["indicator"], r["country"], r["observation"]].append((r["publication"], i))
    for rows in history.values():
        rows.sort()
    return history


def asof(history, indicator, country, obs, cutoff):
    find = history.get((indicator, country, obs), [])
    i = bisect_right(find, (cutoff, float("inf"))) - 1
    return find[i][1] if i >= 0 else None


def feature_rows(dataset, measurement, history, geo_cards, metric_cards):
    evidence = []
    for q, r in zip(dataset, measurement, strict=True):
        assert q["id"] == r["id"]
        country = registry_country(q["text"])
        canonical_metric = registry_indicator(q["text"])
        assert (obs := parse_dates(q["text"])) == (q["observation"], q["cutoff"])
        metric_rank = sorted(range(len(metric_cards)), key=lambda i: (-r["metric_scores"][i], i))
        top, next_ = [r["metric_scores"][i] for i in metric_rank[:2]]
        suggested = metric_cards[metric_rank[0]]
        card = max(range(len(geo_cards)), key=r["card_scores"].__getitem__)
        geo_metric, geo_country = geo_cards[card]
        truth = q["gold"]
        if truth is not None:
            assert country in COUNTRIES and q["kind"] == "answerable"
        values = q | {"country": country, "canonical_metric": canonical_metric,
                      "suggested": suggested, "top_score": top, "margin": top-next_,
                      "literal_registry": asof(history, canonical_metric, country, *obs),
                      "full_card_then_asof": asof(history, geo_metric, geo_country, *obs),
                      "semantic_then_asof": asof(history, canonical_metric or suggested, country, *obs),
                      "retrieval": {k: r[k] for k in ("dense_docs", "dense_docs_dated", "bm25_docs", "bm25_docs_dated")}}
        evidence.append(values)
    return evidence


def calibration(evidence, feature, negative):
    filtered = [r for r in evidence if r["country"] and r["canonical_metric"] is None]
    positive = [r[feature] for r in filtered if r["kind"] == "answerable"]
    other = [r[feature] for r in filtered if r["kind"] == negative]
    assert positive and other
    possible = sorted(set(positive + other))
    possible = [possible[0] - 1e-8] + possible + [possible[-1] + 1e-8]
    actions = [(sum(s >= t for s in positive) / len(positive), sum(s < t for s in other) / len(other), t)
               for t in possible]
    retain, reject, cut = max(actions, key=lambda r: ((r[0]+r[1])/2, r[1], -r[2]))
    return {"feature": feature, "calibration_contract": negative, "threshold": cut,
            "retained_positives": retain, "rejected_negatives": reject, "positives": len(positive), "negatives": len(other)}


def choices(evidence, corpus, bank, outside, collision):
    timeline = defaultdict(list)
    for record in corpus:
        timeline[record["observation"]].append(record["publication"])
    for days in timeline.values():
        days.sort()
    all_rows = []
    for r in evidence:
        country, known = r["country"], r["canonical_metric"]
        raw = r["retrieval"]
        result = {
            k: r[k] for k in ("literal_registry", "full_card_then_asof", "semantic_then_asof")
        }
        allow_unknown = r["top_score"] >= outside["threshold"]
        allow_collision = r["margin"] >= collision["threshold"]
        for label, permitted in (("unknown_gate", allow_unknown), ("collision_gate", allow_collision),
                                 ("separate_gates", allow_unknown and allow_collision)):
            result[label] = r["semantic_then_asof"] if country and (known or permitted) else None
        for column, ranking in raw.items():
            dated = column.endswith("dated")
            eligible_total = bisect_right(timeline[r["observation"]], r["cutoff"]) if dated else len(corpus)
            assert len(ranking) == len(set(ranking)) == min(64, eligible_total)
            assert min(ranking) >= 0 and max(ranking) < len(corpus)
            result[column] = i = ranking[0]
            record = corpus[i]
            if dated:
                assert all(corpus[j]["observation"] == r["observation"] and corpus[j]["publication"] <= r["cutoff"]
                           for j in ranking)
            result[column + "_then_asof"] = asof(bank, record["indicator"], record["country"], r["observation"], r["cutoff"])
        if r["gold"] is not None:
            best = corpus[r["gold"]]
            assert asof(bank, best["indicator"], best["country"], r["observation"], r["cutoff"]) == r["gold"]
            if r["literal_registry"] is not None:
                assert r["literal_registry"] == r["gold"]
        outputs = r | {"answers": result}
        all_rows.append(outputs)
    return all_rows


def quality(data, corpus):
    policies = list(data[0]["answers"])
    parts = {"all_answerable": [r for r in data if r["kind"] == "answerable"]}
    parts.update({v: [r for r in data if r["style"] == v] for v in sorted({r["style"] for r in data})})
    result = {}
    for scope, items in parts.items():
        unique = group_queries([r | {"gold_docs": [r["group"]]} for r in items])
        sample = resample(unique)
        output = {}
        for arm in policies:
            output_values = [r["answers"][arm] for r in items]
            strict = np.asarray([q["gold"] == m for q, m in zip(items, output_values, strict=True)], dtype=int)
            sham = np.asarray([q["gold"] == q["answers"]["dense_docs"] for q in items], dtype=int)
            numeric = sum(r["gold"] is not None and p is not None and corpus[r["gold"]]["value"] == corpus[p]["value"]
                          for r, p in zip(items, output_values, strict=True))
            allowed = sum(p is not None and corpus[p]["publication"] <= r["cutoff"]
                          for r, p in zip(items, output_values, strict=True))
            address = sum(r["gold"] is not None and p is not None
                          and all(corpus[r["gold"]][k] == corpus[p][k] for k in ("indicator", "country", "observation"))
                          for r, p in zip(items, output_values, strict=True))
            delta = strict-sham
            output[arm] = {"correct": int(sum(strict)), "numeric_exact": numeric, "correct_typed_address": address,
                           "answered": sum(p is not None for p in output_values), "within_cutoff": allowed,
                           "versus_dense": {"points": float(100*delta.mean()), "cluster_interval95_pp": [100 * x for x in limits(delta, unique, sample)]}}
        result[scope] = {"rows": len(items), "case_clusters": len(unique), "outcomes": output,
                         "top64_gold": {k: sum(r["gold"] is not None and r["gold"] in r["retrieval"][k] for r in items)
                                        for k in data[0]["retrieval"]}}
    return result


def run(fixture_path, measurement):
    fixture = read(fixture_path)
    assert sha256(fixture_path.read_bytes()).hexdigest() == measurement["fixture_sha256"]
    corpus = fixture["corpus"]
    assert len(corpus) == measurement["indexed_records"]
    bank = archive_bank(corpus)
    card_labels = measurement["card_labels"]
    metric_labels = measurement["metric_labels"]
    assert {tuple(x) for x in card_labels} == {(metric, country) for metric in NAMES for country in COUNTRIES}
    assert set(metric_labels) == set(NAMES)
    prepared = {s: feature_rows(r, measurement["sections"][s]["rows"], bank, card_labels, metric_labels)
                for s, r in fixture["questions"].items()}
    dev = prepared["development"]
    thresholds = (calibration(dev, "top_score", "unknown"), calibration(dev, "margin", "ambiguous"))
    result = {s: quality(choices(rows, corpus, bank, *thresholds), corpus) for s, rows in prepared.items()}
    return {k: v for k, v in measurement.items() if k != "sections"} | {
        "sources": fixture["sources"], "gate_calibration": dict(zip(("unknown", "collision"), thresholds, strict=True)),
        "timing": {k: {a: b for a, b in v.items() if a != "rows"} for k, v in measurement["sections"].items()},
        "outcomes": result,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("fixture", "measurement", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a.fixture, read(a.measurement)), sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    main()
