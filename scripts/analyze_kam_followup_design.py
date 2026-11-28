#!/usr/bin/env python3
"""Size future matched experiments from frozen paired cluster dispersion."""
import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from kam_adapter_preservation import SEEDS,digest
from kam_math_kl_protocol import write_json
from summarize_kam_distant_ablation import margin_panel
from summarize_kam_evidence_pins import archive
from summarize_kam_quality_native import per_example

PROTOCOL="kam-attention-followup-design-v1"
POWER=.80
ALPHA=.05


def read(path):
    return json.loads(path.read_text())


def size(delta,family,targets,*,one_sided=False,seconds_per_example=None,higher_is_better=True):
    units,indices=np.unique(family,return_inverse=True)
    n=len(family)
    g=len(units)
    assert delta.shape==(len(SEEDS),n) and g>=10
    d=delta.mean(axis=0)
    center=float(d.mean())
    residual=np.bincount(indices,weights=d-center)
    v=float(g/(g-1)*np.sum(residual**2)/n**2)
    z=NormalDist()
    k=z.inv_cdf(1-(ALPHA if one_sided else ALPHA/2))+z.inv_cdf(POWER)
    experiment=[]
    for goal in targets:
        required=math.ceil((k*math.sqrt(g*v)/goal)**2)
        examples=math.ceil(required*n/g)
        q={"gap":goal,"required_independent_clusters":required,"approx_examples":examples}
        if seconds_per_example is not None:
            q["profile_equivalent_gpu_hours"]=float(examples*seconds_per_example/3600)
        if one_sided:
            effective = goal + (center if higher_is_better else -center)
            observed_size=math.ceil((k*math.sqrt(g*v)/effective)**2) if effective>0 else None
            q["clusters_if_observed_delta_is_true"]=observed_size
        experiment.append(q)
    return {"observed_delta":center,"by_seed":delta.mean(axis=1).tolist(),"paired_examples":n,"clusters":g,
       "estimated_standard_error":math.sqrt(v),"cluster_scale":math.sqrt(g*v),"two_sided":not one_sided,
       "seconds_per_example_at_profiled_cost":seconds_per_example,"sizes":experiment}


def retrieval(ref,width,prefix,arm,labels,field=None):
    identifiers=[(r["id"],r["family"],order) for r in labels for order in (0,1)]
    predictions=[]
    for seed in SEEDS:
        outcome=json.loads(ref[f"{prefix}-{seed}-{arm}"])["outcomes"][width]
        rows=outcome if field is None else outcome[field]
        predictions.append(margin_panel(rows,identifiers,exact=True)>0)
    return np.stack(predictions)


def summarize(repo):
    base=repo/"results"
    hot=base/"general_benchmarks_long_hotpot"
    mu=base/"general_benchmarks_attention_transfer"
    qu=base/"general_benchmarks_quality_native"
    allocation=base/"general_benchmarks_layer_budget"
    source={kind:{"manifest":digest(source/manifest),"archive":digest(source/"outcomes.json.gz")} for kind,source,manifest in (
        ("hotpot",hot,"hotpot-manifest.json"),("musique",mu,"transfer-manifest.json"),("quality",qu,"quality-manifest.json"),
        ("allocation",allocation,"layer-budget-selection.json"))}

    h=read(hot/"hotpot-manifest.json")["fixture"]["8192"]["questions"]
    h_prior=archive(hot/"outcomes.json.gz")
    layered=archive(allocation/"outcomes.json.gz")
    hs=read(allocation/"summary.json")
    h0=retrieval(h_prior,"8192","hotpot","flash",h)
    h1=retrieval(layered,"8192","layer-budget","output-selected",h,field="hotpot")
    ht=hs["timing"]["8192"]
    pace=2*len(SEEDS)*sum(ht[p]["wall_p50_ms"] for p in ("flash","output-selected"))/1000
    hot_case=size(h1.astype(np.int8)-h0,[r["family"] for r in h],(.05,.08,.10),
                  seconds_per_example=pace)

    m_prior=archive(mu/"outcomes.json.gz")
    manifest=read(mu/"transfer-manifest.json")["fixture"]
    control_m, selected_m = {},{}
    matched_ids=None
    for width,data in manifest.items():
        rows=data["questions"]
        current=[(r["id"],r["family"]) for r in rows]
        assert matched_ids is None or matched_ids==current
        matched_ids=current
        control_m[width]=retrieval(m_prior,width,"transfer","protected",rows,field="retrieval")
        selected_m[width]=retrieval(m_prior,width,"transfer","selected",rows,field="retrieval")
    contrast={w:selected_m[w].astype(np.int8)-control_m[w] for w in selected_m}
    old=read(mu/"summary.json")
    pace=2*len(SEEDS)*sum(r[p]["wall_p50_ms"] for r in old["timings"].values() for p in ("selected","protected"))/1000
    mu_case=size(contrast["8192"]-contrast["4096"],[f for _,f in matched_ids],(.06,.08,.10),
                 seconds_per_example=pace)

    q=read(qu/"quality-manifest.json")["selected"]
    q_prior=archive(qu/"outcomes.json.gz")
    rotations=np.array([r["orders"] for r in q])
    golds=np.array([r["answer"] for r in q])
    lengths=np.array([r["lengths"][0] for r in q])
    q_score={}
    for mode in ("flash","tail"):
        outputs=[json.loads(q_prior[f"quality-{s}-{mode}"])["result"] for s in SEEDS]
        assert all([r["id"] for r in series]==[r["id"] for r in q] for series in outputs)
        q_score[mode]=per_example(np.array([[a["logits"] for a in series] for series in outputs]),rotations,golds)["rotation_mean"]
    qt=read(qu/"summary.json")["timing"]
    axes=np.array([r["tokens"] for r in qt])
    pace=4*len(SEEDS)*sum(np.interp(lengths,axes,[r["policies"][p]["wall_p50_ms"] for r in qt]).mean()
                        for p in ("flash","tail"))/1000
    q_case=size(q_score["tail"].astype(np.int8)-q_score["flash"],[r["article_id"] for r in q],(.02,.03,.05),
                one_sided=True,seconds_per_example=pace)

    losses={}
    book_ids=[r["id"] for r in read(mu/"transfer-manifest.json")["pg19"]]
    for mode in ("flash","selected"):
        panels=[json.loads(m_prior[f"transfer-{s}-{mode}"])["outcomes"]["8192"]["pg19"] for s in SEEDS]
        assert all([r["id"] for r in p]==book_ids for p in panels)
        losses[mode]=np.array([[a["nll"][0] for a in p] for p in panels])
    pg_case=size(losses["selected"]-losses["flash"],book_ids,(.01,.02,.03),one_sided=True,higher_is_better=False)
    return {"protocol":PROTOCOL,"source_hashes":source,"planning_power":POWER,"alpha":ALPHA,"panels":{
        "hotpot_8k_output_selected_vs_flash":hot_case,"musique_length_interaction_final_vs_protected":mu_case,
        "quality_final_vs_flash_noninferiority_if_true_equal":q_case,
        "pg19_final_vs_flash_log_loss_noninferiority_if_true_equal":pg_case}}


if __name__=="__main__":
    c=argparse.ArgumentParser(description=__doc__)
    c.add_argument("repository",type=Path)
    c.add_argument("output",type=Path)
    a=c.parse_args()
    write_json(a.output,summarize(a.repository))
