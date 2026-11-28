#!/usr/bin/env python3
"""Replay matched layer-budget quality, calibration, and latency records."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from select_kam_layer_budget import PROTOCOL, select
from summarize_kam_block_shortlist import quality, rates, score_matrix
from summarize_kam_distant_ablation import margin_panel
from summarize_kam_evidence_pins import WIDTHS, archive, retrieval_summary, summarize_timing
from summarize_kam_live_token_factorial import estimate

PINS = ("output-selected", "evidence-selected", "alternating")
ARMS = ("flash", "protected", *PINS)
COMPARISONS = tuple((mode, base) for mode in PINS for base in ("protected", "flash")) + (
    ("output-selected","alternating"), ("evidence-selected","alternating"))


def read(path):
    return json.loads(path.read_text())


def summarize(work, selection_path, m_fixture, h_fixture, m_prior, h_prior):
    plan, m, h = map(read, (selection_path,m_fixture,h_fixture))
    assert plan["protocol"]==PROTOCOL and set(plan["schedule"])==set(PINS)
    assert json.loads(json.dumps(select(work))) == plan
    plan_hash, m_hash, h_hash = map(digest, (selection_path,m_fixture,h_fixture))
    gpu = plan["gpu"]
    base, hot = map(archive,(m_prior,h_prior))
    checks = {s:json.loads(base[f"transfer-{s}-flash"])["checkpoint_sha256"] for s in SEEDS}
    assert {str(k):v for k,v in checks.items()}==plan["checkpoints"]
    hashes = dict(plan["calibration_sha256"])
    profile_path = work/"layer-budget-profile.json"
    profile = read(profile_path)
    hashes[profile_path.stem]=digest(profile_path)
    base_key = ("gpu","seed","checkpoint_sha256","plan_sha256","musique_manifest_sha256","hotpot_manifest_sha256")
    assert tuple(profile[k] for k in base_key) == (gpu,SEEDS[0],checks[SEEDS[0]],plan_hash,m_hash,h_hash)
    assert profile["arm"]=="benchmark"
    records = {arm:[] for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            if arm in ("flash","protected"):
                a, b = [json.loads(src[f"{name}-{seed}-{arm}"]) for src,name in ((base,"transfer"),(hot,"hotpot"))]
                assert (a["gpu"],a["seed"],a["checkpoint_sha256"],a["transfer_manifest_sha256"]) == (
                    gpu,seed,checks[seed],m_hash)
                assert (b["gpu"],b["seed"],b["checkpoint_sha256"],b["hotpot_manifest_sha256"]) == (gpu,seed,checks[seed],h_hash)
                records[arm].append({w: a["outcomes"][w] | {"hotpot":b["outcomes"][w]} for w in WIDTHS})
            else:
                path = work/f"layer-budget-{seed}-{arm}.json"
                r = read(path)
                assert tuple(r[k] for k in base_key)==(gpu,seed,checks[seed],plan_hash,m_hash,h_hash) and r["arm"]==arm
                hashes[path.stem]=digest(path)
                records[arm].append(r["outcomes"])

    outputs = {}
    books = [r["id"] for r in m["pg19"]]
    for width in WIDTHS:
        m_cases,h_cases = [f["fixture"][width]["questions"] for f in (m,h)]
        labels = [(r["id"],r["family"],order) for r in m_cases for order in (0,1)]
        hot_labels = [(r["id"],r["family"],order) for r in h_cases for order in (0,1)]
        pg = {arm:quality([score_matrix(o[width],books,labels,windows=1) for o in values])
              for arm,values in records.items()}
        measures = {a:{arm:np.stack([margin_panel(o[width][field], lab, exact=True) for o in values])
                       for arm,values in records.items()}
                    for a,field,lab in (("musique","retrieval",labels),("hotpot","hotpot",hot_labels))}
        outputs[width] = {"perplexity": {arm:rates(row)["perplexity"] for arm,row in pg.items()},
            "paired_nll": {f"{a}_minus_{b}": estimate(pg[a]["nll"]-pg[b]["nll"]) for a,b in COMPARISONS},
            "retrieval": {a: retrieval_summary(v,cases,COMPARISONS)
                          for a,v,cases in (("musique",measures["musique"],m_cases),("hotpot",measures["hotpot"],h_cases))}}
    return {"protocol":PROTOCOL,"gpu":gpu,"selection_sha256":plan_hash,"musique_manifest_sha256":m_hash,
            "hotpot_manifest_sha256":h_hash,"musique_prior_sha256":digest(m_prior),"hotpot_prior_sha256":digest(h_prior),
            "raw_sha256":hashes,"quality":outputs,"timing":summarize_timing(profile,ARMS)}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("work","selection","musique","hotpot","musique_prior","hotpot_prior","output"):
        cli.add_argument(name,type=Path)
    a=cli.parse_args()
    write_json(a.output,summarize(a.work,a.selection,a.musique,a.hotpot,a.musique_prior,a.hotpot_prior))
