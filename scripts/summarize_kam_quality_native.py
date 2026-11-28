#!/usr/bin/env python3
"""Replay original-order and offset-balanced long-document choice scores."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from prepare_kam_quality_native import PROTOCOL
from summarize_kam_distant_ablation import continuous_interval
from summarize_kam_evidence_pins import archive
from summarize_kam_pretrained_endpoint_selection import family_difference

ARMS = ("flash","protected","tail","question-only")
MEASURES = ("original_order","rotation_mean","all_orders")


def read(p):
    return json.loads(p.read_text())


def per_example(logits, order, answers):
    scores = np.take_along_axis(logits, order.argsort(axis=-1)[None], axis=-1)
    good = np.arange(4)==answers[:,None]
    gold = np.where(good[None,:,None,:],scores,-np.inf).max(-1)
    wrong = np.where(good[None,:,None,:],-np.inf,scores).max(-1)
    normalized = scores.mean(axis=2)
    main_margin = np.where(good[None,:,:],normalized,-np.inf).max(-1) - np.where(good[None,:,:],-np.inf,normalized).max(-1)
    shifted = normalized - normalized.max(axis=-1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    choice_nll = -np.where(good[None,:,:],log_prob,0).sum(axis=-1)
    brier = ((np.exp(log_prob)-good[None,:,:])**2).sum(axis=-1)
    return {"original_order":gold[:,:,0]>wrong[:,:,0],"rotation_mean":main_margin>0,
            "all_orders":(gold>wrong).all(-1),"margin":main_margin,"choice_nll":choice_nll,"choice_brier":brier}


def timing_panel(profile, names, lengths, policy_names):
    timing = []
    assert len(profile["result"])==8
    for record in profile["result"]:
        assert record["tokens"] == lengths[names.index(record["id"])]
        assert set(record["timings"])==set(policy_names)
        policies = {}
        for arm,r in record["timings"].items():
            wall,cuda=map(np.array,(r["wall_ms"],r["cuda_ms"]))
            assert len(wall)==len(cuda)==15 and all(np.isfinite(a).all() and min(a)>0 for a in (wall,cuda))
            policies[arm]={"wall_p50_ms":float(np.median(wall)),"wall_p95_ms":float(np.quantile(wall,.95)),
                           "cuda_p50_ms":float(np.median(cuda)),"peak_bytes":r["incremental_allocated_bytes"]}
        timing.append({"id":record["id"],"tokens":record["tokens"],"policies":policies})
    assert len({r["id"] for r in timing})==8
    return timing


def summarize(work, manifest_path, backbone, budget_selection=None, context_audit=None):
    manifest = read(manifest_path)
    assert manifest["protocol"]==PROTOCOL
    fixture = digest(manifest_path)
    previous = archive(backbone)
    rows = manifest["selected"]
    names = [r["id"] for r in rows]
    families = np.array([r["article_id"] for r in rows])
    lengths = np.array([r["lengths"][0] for r in rows])
    counts = Counter(families)
    assert len(set(names))==len(rows) and max(counts.values())<=manifest["article_cap"]
    profile_path = work/"quality-profile.json"
    profile = read(profile_path)
    hashes = {profile_path.stem:digest(profile_path)}
    gpu = profile["gpu"]
    checks = {s:json.loads(previous[f"backbone-{s}-0"])["checkpoint_sha256"] for s in SEEDS}
    assert (profile["protocol"],profile["arm"], profile["seed"],profile["checkpoint_sha256"],profile["quality_manifest_sha256"]) == (
        PROTOCOL, "profile",SEEDS[0],checks[SEEDS[0]],fixture)
    answers = np.array([r["answer"] for r in rows])
    hard = np.array([r["hard"] for r in rows],dtype=bool)
    order = np.array([r["orders"] for r in rows])
    assert set(answers) == {0,1,2,3} and np.all(np.sort(order,axis=1)==np.arange(4)[None,:,None])
    assert np.all(np.sort(order,axis=-1)==np.arange(4)) and np.all(order[:,0]==np.arange(4))
    arms=ARMS
    budget_panel, budget_hash=None,None
    comparisons=(("protected","flash"),("tail","flash"),("tail","protected"),("flash","question-only"))
    if budget_selection is not None:
        current=read(budget_selection)
        budget_hash=digest(budget_selection)
        assert current["protocol"]=="kam-layer-budget-v1"
        assert current["checkpoints"]=={str(s):sha for s,sha in checks.items()}
        assert current["schedule"]["output-selected"].count(4)==current["schedule"]["output-selected"].count(12)==16
        q=work/"quality-budget-profile.json"
        budget_panel=read(q)
        hashes[q.stem]=digest(q)
        common=("protocol","arm","seed","checkpoint_sha256","quality_manifest_sha256","gpu")
        assert all(budget_panel[k]==profile[k] for k in common) and budget_panel["budget_selection_sha256"]==budget_hash
        arms=(*ARMS,"output-selected")
        comparisons += tuple(("output-selected",r) for r in ("flash","protected","tail"))
    results = {}
    for arm in arms:
        values = []
        for seed in SEEDS:
            path = work/f"quality-{seed}-{arm}.json"
            r = read(path)
            hashes[path.stem]=digest(path)
            assert (r["protocol"],r["gpu"],r["arm"],r["seed"],r["checkpoint_sha256"],r["quality_manifest_sha256"]) == (
                PROTOCOL,gpu,arm,seed,checks[seed],fixture)
            if arm=="output-selected":
                assert r["budget_selection_sha256"]==budget_hash
            assert [p["id"] for p in r["result"]]==names
            scores = np.array([p["logits"] for p in r["result"]])
            assert scores.shape==(len(names),4,4) and np.isfinite(scores).all()
            values.append(scores)
        results[arm]=per_example(np.stack(values),order,answers)
    quality = {}
    cohorts=[("all", np.ones(len(rows),dtype=bool)),("hard",hard),("other",~hard),
             ("up-to-4k", lengths<=4096),("above-4k",lengths>4096)]
    human_context_hash=None
    if context_audit is not None:
        human_context_hash=digest(context_audit)
        rated=read(context_audit)
        assert (rated["quality_manifest_sha256"],rated["publisher_archive_sha256"])==(fixture,manifest["publisher_sha256"])
        assert [r["id"] for r in rated["context_ratings"]]==names
        votes=np.array([2*sum(x>=3 for x in r["values"])-len(r["values"]) for r in rated["context_ratings"]])
        cohorts += [("human-at-least-a-third",votes>0),("human-local",votes<0),("human-context-tied",votes==0)]
    for cohort, keep in cohorts:
        if not keep.any():
            continue
        cells={arm:{key:a[:,keep] for key,a in v.items()} for arm,v in results.items()}
        effects = {}
        for a,b in comparisons:
            effects[f"{a}_minus_{b}"] = {metric:family_difference(cells[a][metric],cells[b][metric],families[keep])
                                          for metric in MEASURES}
            for metric in ("margin","choice_nll","choice_brier"):
                effects[f"{a}_minus_{b}"][metric] = continuous_interval(cells[a][metric]-cells[b][metric],families[keep])
        quality[cohort]={"n":int(keep.sum()),"articles":len(set(families[keep])),
            "scores":{arm:{m:{"mean":float(v.mean()),"by_seed":v.mean(1).tolist()} for m,v in cell.items()}
                      for arm,cell in cells.items()},"paired":effects}
    timing=timing_panel(profile,names,lengths,ARMS[:3])
    budget_timing=None
    if budget_panel is not None:
        budget_timing=timing_panel(budget_panel,names,lengths,("flash","protected","output-selected"))
        assert [(r["id"],r["tokens"]) for r in budget_timing]==[(r["id"],r["tokens"]) for r in timing]
    return {"protocol":PROTOCOL,"gpu":gpu,"manifest_sha256":fixture,"raw_sha256":hashes,"quality":quality,
            "length_distribution":{"min":int(lengths.min()),"median":float(np.median(lengths)),"max":int(lengths.max())},
            "timing":timing,"budget_timing":budget_timing,"budget_selection_sha256":budget_hash,
            "human_context_sha256":human_context_hash}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for n in ("work","manifest","backbone","output"):
        cli.add_argument(n,type=Path)
    cli.add_argument("--budget-selection",type=Path)
    cli.add_argument("--human-context",type=Path)
    a=cli.parse_args()
    write_json(a.output,summarize(a.work,a.manifest,a.backbone,a.budget_selection,a.human_context))
