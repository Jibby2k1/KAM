#!/usr/bin/env python3
"""Explain matched policy disagreements with observed logit perturbations."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS,digest
from kam_math_kl_protocol import write_json
from summarize_kam_distant_ablation import continuous_interval,margin_panel
from summarize_kam_evidence_pins import archive
from summarize_kam_pretrained_endpoint_selection import family_difference

PROTOCOL="kam-mirrored-margin-stability-v1"
SPARSE=("protected","final-window","output-selected")
BINS=(0.5,1.,2.)


def read(path):
    return json.loads(path.read_text())


def winner(x):
    candidates=x==x.max(axis=-1,keepdims=True)
    count=candidates.sum(-1)
    labels=np.where(count==1,candidates.argmax(axis=-1),-1)
    ordered=np.sort(x,axis=-1)
    margin=ordered[..., -1]-ordered[..., -2]
    return labels,margin


def metrics(scores, answer, family):
    ref=scores["flash"]
    chosen, margin=winner(ref)
    truth=chosen==answer[None,:]
    marks=np.digitize(margin,BINS,right=True)
    result={}
    for policy in SPARSE:
        post=scores[policy]
        alternative,_=winner(post)
        observed=alternative==answer[None,:]
        change=chosen!=alternative
        restored=observed & ~truth
        lost=truth & ~observed

        # A common logit offset does not change any comparison in the decision rule.
        delta=post-ref
        radius=(delta.max(-1)-delta.min(-1))/2
        certified=margin>2*radius
        assert np.all(~certified | ~change)
        by_margin=[]
        for i in range(len(BINS)+1):
            use=marks==i
            n=int(use.sum())
            row={"n_seed_questions":n,"errors_recovered":int((restored&use).sum()),
                 "correct_lost":int((lost&use).sum()),"decisions_changed":int((change&use).sum())}
            if n:
                row["fraction_changed"]=row["decisions_changed"]/n
                row["fraction_certified_stable"]=float(certified[use].mean())
                row["p50_centered_delta"]=float(np.median(radius[use]))
            by_margin.append(row)
        result[policy]={"accuracy_flash":float(truth.mean()),"accuracy_policy":float(observed.mean()),
            "paired_accuracy":family_difference(observed,truth,family),
            "paired_centered_logit_delta":continuous_interval(radius,family),
            "p50_centered_logit_delta":float(np.median(radius)),
            "fraction_decisions_changed":float(change.mean()),"fraction_certified_stable":float(certified.mean()),
            "by_flash_margin":by_margin}
    return result


def binary(path, prior, mixed, dataset):
    manifest=read(path)
    output={}
    ref_prefix="transfer" if dataset=="musique" else "hotpot"
    ref_tail="selected" if dataset=="musique" else "tail-256"
    field="retrieval" if dataset=="musique" else "hotpot"
    fixture_key="musique_manifest_sha256" if dataset=="musique" else "hotpot_manifest_sha256"
    for width, s in manifest["fixture"].items():
        rows=s["questions"]
        labels=[(r["id"],r["family"],o) for r in rows for o in (0,1)]
        cohort=np.array([r["anchored_foil"] for r in rows])
        families=np.array([r["family"] for r in rows])
        values={}
        for policy in ("flash",*SPARSE):
            samples=[]
            for seed in SEEDS:
                if policy=="output-selected":
                    readout=json.loads(mixed[f"layer-budget-{seed}-output-selected"])
                    assert readout[fixture_key]==digest(path)
                    r=readout["outcomes"][width][field]
                else:
                    a=ref_tail if policy=="final-window" else policy
                    ref=json.loads(prior[f"{ref_prefix}-{seed}-{a}"])
                    key="transfer_manifest_sha256" if dataset=="musique" else "hotpot_manifest_sha256"
                    assert ref[key]==digest(path)
                    r=ref["outcomes"][width]
                    if dataset=="musique":
                        r=r["retrieval"]
                # Align two options and average the two presentations. Their scores differ by M/2.
                y=margin_panel(r,labels,exact=True)/4
                samples.append(np.stack((-y,y),axis=-1))
            values[policy]=np.stack(samples)
        output[width]={}
        for name,keep in (("all",np.ones_like(cohort)),("linked",cohort),("unlinked",~cohort)):
            subset={a:v[:,keep] for a,v in values.items()}
            output[width][name]={"questions":int(keep.sum()),"policies":metrics(subset,
                np.ones(int(keep.sum()),dtype=int),families[keep])}
    return output


def quality(path, prior):
    manifest=read(path)
    rows=manifest["selected"]
    orders=np.array([a["orders"] for a in rows])
    order=orders.argsort(axis=-1)
    gold=np.array([a["answer"] for a in rows])
    families=np.array([a["article_id"] for a in rows])
    lengths=np.array([a["lengths"][0] for a in rows])
    values={}
    for policy in ("flash",*SPARSE):
        a="tail" if policy=="final-window" else policy
        samples=[]
        for seed in SEEDS:
            record=json.loads(prior[f"quality-{seed}-{a}"])
            assert record["quality_manifest_sha256"]==digest(path)
            assert [r["id"] for r in record["result"]]==[r["id"] for r in rows]
            score=np.array([r["logits"] for r in record["result"]])
            samples.append(np.take_along_axis(score,order,axis=-1).mean(axis=1))
        values[policy]=np.stack(samples)

    panels={}
    hard=np.array([bool(a["hard"]) for a in rows])
    for name,keep in (("all",np.ones(len(rows),dtype=bool)),("hard",hard),("over-4k",lengths>4096)):
        subset={a:v[:,keep] for a,v in values.items()}
        panels[name]={"questions":int(keep.sum()),"policies":metrics(subset,gold[keep],families[keep])}
    return panels


def summarize(m_path,m_prior,h_path,h_prior,q_path,q_prior,allocation):
    origin={}
    for dtype,p in (("musique_fixture",m_path),("musique_predictions",m_prior),
      ("hotpot_fixture",h_path),("hotpot_predictions",h_prior),("quality_fixture",q_path),
      ("quality_predictions",q_prior),("allocation_predictions",allocation)):
        origin[dtype]=digest(p)
    mix=archive(allocation)
    return {"protocol":PROTOCOL,"source_hashes":origin,"flash_margin_bins":BINS,
        "musique":binary(m_path,archive(m_prior),mix,"musique"),
        "hotpot":binary(h_path,archive(h_prior),mix,"hotpot"),
        "quality":quality(q_path,archive(q_prior))}


if __name__=="__main__":
    c=argparse.ArgumentParser(description=__doc__)
    for field in ("musique","musique_prior","hotpot","hotpot_prior","quality","quality_prior","allocation","output"):
        c.add_argument(field,type=Path)
    a=c.parse_args()
    write_json(a.output,summarize(a.musique,a.musique_prior,a.hotpot,a.hotpot_prior,a.quality,a.quality_prior,a.allocation))
