#!/usr/bin/env python3
"""Test confidence deferral with original order and a measured two-pass cost model."""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_kam_natural_data import stable
from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from summarize_kam_distant_ablation import continuous_interval
from summarize_kam_evidence_pins import archive
from summarize_kam_pretrained_endpoint_selection import family_difference

PROTOCOL="kam-quality-deferral-v1"
FIRST=("protected","tail","output-selected")


def read(path):
    return json.loads(path.read_text())


def weights(timing, policy, lengths):
    x=np.array([r["tokens"] for r in timing])
    assert (np.diff(x)>0).all() and min(lengths)>=x[0] and max(lengths)<=x[-1]
    y=np.array([r["policies"][policy]["wall_p50_ms"] for r in timing])
    assert np.isfinite(y).all() and min(y)>0
    return np.interp(lengths,x,y)


def records(content, rows, method):
    messages=[]
    for seed in SEEDS:
        r=json.loads(content[f"quality-{seed}-{method}"])
        assert [t["id"] for t in r["result"]]==[t["id"] for t in rows]
        logits=np.array([t["logits"][0] for t in r["result"]])
        assert logits.shape==(len(rows),4) and np.isfinite(logits).all()
        messages.append(logits)
    return np.stack(messages)


def correct(logits, answer):
    matched=answer[:,None]==np.arange(4)
    return np.where(matched,logits,-np.inf).max(-1)>np.where(matched,-np.inf,logits).max(-1)


def choose(score, fast_ok, full_ok, cost, end):
    cap=np.broadcast_to(end,score.shape)
    initial_cost=float(cost.mean())
    budget=float(cap.mean())
    if initial_cost>budget+1e-9:
        raise ValueError("The first pass already exceeds the calibrated Flash cost")
    # Sorting and accumulating by tied confidence group avoids order artifacts at the threshold.
    index=np.argsort(score.flatten(),kind="stable")
    margins=score.flatten()[index]
    exact=np.where(np.r_[margins[1:]!=margins[:-1],True])[0]
    bill=np.cumsum(cap.flatten()[index])[exact]/score.size + initial_cost
    base_delta=(full_ok.astype(np.int8)-fast_ok.astype(np.int8)).flatten()[index]
    performance=np.cumsum(base_delta)[exact]
    feasible=[(int(performance[i]),-float(bill[i]),float(margins[j])) for i,j in enumerate(exact) if bill[i]<=budget]
    win,cost,cut=max([(0,-initial_cost,None),*feasible],key=lambda x:x[:2])
    return cut, {"accuracy_gain_on_calibration":win/score.size,"mean_predicted_ms_on_calibration":-cost,
                  "calibration_flash_budget_ms":budget}


def oracle(fast_ok,full_ok,first_ms,flash_ms):
    budget=flash_ms.sum()-first_ms.sum()
    changes=full_ok & ~fast_ok
    picks=np.zeros_like(changes)
    counts,min_cost=[],[]
    for i,flags in enumerate(changes):
        score=np.flatnonzero(flags)
        candidates=score[np.argsort(flash_ms[score],kind="stable")]
        cumulative=np.cumsum(flash_ms[candidates])
        affordable=int(np.searchsorted(cumulative,budget,side="right"))
        picks[i,candidates[:affordable]]=True
        counts.append(affordable)
        needed=max(int(full_ok[i].sum()-fast_ok[i].sum()),0)
        assert needed<=len(candidates)
        paid=0 if needed==0 else cumulative[needed-1]
        min_cost.append(float((first_ms.sum()+paid)/len(first_ms)))
    return picks,min_cost


def summary(manifest_path,score_summary,archive_path):
    fixture, score=map(read,(manifest_path,score_summary))
    assert score["manifest_sha256"]==digest(manifest_path)
    rows=fixture["selected"]
    ids=np.array([r["article_id"] for r in rows])
    article_ids=sorted(set(ids),key=lambda x:stable(PROTOCOL+str(x)))
    calibration=set(article_ids[::2])
    fit=np.array([a in calibration for a in ids])
    test=~fit
    assert fit.sum()>=190 and test.sum()>=190
    lengths=np.array([r["lengths"][0] for r in rows])
    answers=np.array([r["answer"] for r in rows])
    content=archive(archive_path)
    flash=correct(records(content,rows,"flash"),answers)
    cells={}
    for first in FIRST:
        logits=records(content,rows,first)
        rank=np.sort(logits,axis=-1)
        margin=rank[:,:,-1]-rank[:,:,-2]
        first_ok=correct(logits,answers)
        profile=score["budget_timing"] if first=="output-selected" else score["timing"]
        fast_cost,end_cost=(weights(profile,m,lengths) for m in (first,"flash"))
        tune, details=choose(margin[:,fit],first_ok[:,fit],flash[:,fit],fast_cost[fit],end_cost[fit])
        defer=np.zeros_like(first_ok[:,test]) if tune is None else margin[:,test]<=tune
        base, slow=fast_cost[test],end_cost[test]
        got,truth=first_ok[:,test],flash[:,test]
        accepted=np.where(defer,truth,got)
        spent=base+defer*slow
        picks,min_cost=oracle(got,truth,base,slow)
        optimistic=np.where(picks,truth,got)
        policy=family_difference(accepted,truth,ids[test])
        cost=continuous_interval(spent-slow,ids[test])
        cells[first]={"threshold":tune,"calibration":details,"mean_latency_ms":{"first":float(base.mean()),
            "defer":float(spent.mean()),"flash":float(slow.mean()),"oracle_match_flash":float(np.mean(min_cost))},
            "accuracy":{"first":float(got.mean()),"defer":float(accepted.mean()),"flash":float(truth.mean()),
                        "oracle":float(optimistic.mean())},
            "defer_rate":float(defer.mean()),"oracle_defer_rate":float(picks.mean()),
            "time_fraction_maximum_deferral":float(1-base.sum()/slow.sum()),
            "paired_accuracy_vs_flash":policy,"paired_latency_vs_flash":cost}
    return {"protocol":PROTOCOL,"quality_manifest_sha256":digest(manifest_path),"quality_summary_sha256":digest(score_summary),
            "quality_outcomes_sha256":digest(archive_path),"calibration_articles":sorted(calibration),
            "evaluation_articles":sorted(set(ids[test])),"calibration_questions":int(fit.sum()),
            "evaluation_questions":int(test.sum()),"methods":cells}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("manifest","summary","archive","output"):
        cli.add_argument(name,type=Path)
    a=cli.parse_args()
    write_json(a.output,summary(a.manifest,a.summary,a.archive))
