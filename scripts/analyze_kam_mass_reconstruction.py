#!/usr/bin/env python3
"""Analyze frozen value reconstruction and fixed-budget layer allocations."""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from summarize_kam_evidence_pins import archive

PROTOCOL="kam-mass-reconstruction-v1"
BUDGETS=(4,8,12)
WIDTHS=(4096,8192)
SLOTS=256
BLOCK=64
BUDGET_INDEX={b:i for i,b in enumerate(BUDGETS)}


def read(path):
    return json.loads(path.read_text())


def validate_counterexample():
    w=tuple(map(Fraction,("1/2","3/10","1/5")))
    v=tuple(map(Fraction,("0","2/3","-1")))
    dense=sum(a*b for a,b in zip(w,v,strict=True))
    assert dense==0
    output={}
    for keep in ((0,), (0,1), (1,2)):
        mass=sum(w[i] for i in keep)
        sparse=sum(w[i]*v[i] for i in keep)/mass
        output[",".join(map(str,keep))]={"mass":str(mass),"output":str(sparse),"error":str(abs(dense-sparse))}
    assert output["0,1"]["error"]=="1/4" and output["1,2"]["error"]=="0"
    return {"weight":list(map(str,w)),"value":list(map(str,v)),"dense":str(dense),"subsets":output}


def matched_four_twelve(cost):
    large=set(np.argsort(cost[2]-cost[0],kind="stable")[:16])
    return tuple(12 if i in large else 4 for i in range(32))


def optimize(cost):
    """Find the additive surrogate optimum under a fixed sum of block slots."""
    frontier={0:(0.,())}
    for column in cost.T:
        next_frontier={}
        for used,(objective,previous) in frontier.items():
            for b,c in zip(BUDGETS,column,strict=True):
                slots=used+b
                if slots>SLOTS:
                    continue
                candidate=(objective+float(c),(*previous,b))
                if slots not in next_frontier or candidate<next_frontier[slots]:
                    next_frontier[slots]=candidate
        frontier=next_frontier
    best=frontier[SLOTS][1]
    assert len(best)==32 and sum(best)==SLOTS
    return best


def arrays(archives,selection):
    out={}
    for field in ("books","musique"):
        by_seed={}
        for seed in SEEDS:
            key=f"layer-calibration-{seed}"
            blob=archives[key]
            assert hashlib.sha256(blob.encode()).hexdigest()==selection["calibration_sha256"][key]
            record=json.loads(blob)
            assert record["seed"]==seed and record["pg_manifest_sha256"]==selection["pg_manifest_sha256"]
            by_seed[seed]=record["results"][field]
        ids=[r["id"] for r in by_seed[SEEDS[0]]]
        assert all([r["id"] for r in rs]==ids for rs in by_seed.values())
        energy=np.array([[r["values"]["output_energy"] for r in rs] for rs in by_seed.values()])[:,:,None,:]
        count=np.array([[r["n_queries"] for r in rs] for rs in by_seed.values()])[:,:,None,None]
        raw=lambda item: np.array([[[r["values"][item][str(b)] for b in BUDGETS] for r in rs] for rs in by_seed.values()])
        observed={"output_error":raw("output_sse")/energy, "mass":raw("retained_mass")/count,
                  "present_evidence":raw("retained_evidence")/count}
        assert all(a.shape==(3,len(ids),3,32) and np.isfinite(a).all() for a in observed.values())
        out[field]=(ids,observed)
    return out


def schedule(book_output,book_mass,evidence_choice):
    return {
      "uniform-eight":(8,)*32,"alternating":(4,12)*16,"evidence-selected":tuple(evidence_choice),
      "output-four-twelve":matched_four_twelve(book_output),
      "mass-four-twelve":matched_four_twelve(-book_mass),
      "output-four-eight-twelve":optimize(book_output),
      "mass-four-eight-twelve":optimize(-book_mass),
    }


def mean_at(choice, observed):
    indices=np.array([BUDGET_INDEX[b] for b in choice])
    return {k:float(np.mean(z[...,indices,np.arange(32)])) for k,z in observed.items()}


def ranks(arr):
    _,inv,count=np.unique(arr,return_counts=True,return_inverse=True)
    return (np.cumsum(count)-(count+1)/2)[inv]


def paired_books(metrics, evidence_choice):
    eq=np.arange(metrics["output_error"].shape[1])
    held=[]
    for j in eq:
        support=eq!=j
        cost=metrics["output_error"][:,support].mean((0,1))
        mass=metrics["mass"][:,support].mean((0,1))
        allocations=schedule(cost,mass,evidence_choice)
        test={k:z[:,j:j+1] for k,z in metrics.items()}
        held.append({mode:{"scores":mean_at(s,test),"schedule":s} for mode,s in allocations.items()})
    expected=held[0].keys()
    stats={}
    for mode in expected:
        measures=held[0][mode]["scores"].keys()
        stats[mode]={"averages":{k:float(np.mean([r[mode]["scores"][k] for r in held])) for k in measures},
            "lower_error_than_alternating":sum(r[mode]["scores"]["output_error"]<r["alternating"]["scores"]["output_error"] for r in held),
            "lower_error_than_uniform":sum(r[mode]["scores"]["output_error"]<r["uniform-eight"]["scores"]["output_error"] for r in held)}
    return held,stats


def key_reads(width,choice):
    n=np.arange(width,dtype=np.int64)
    return int(sum(np.sum(BLOCK*np.minimum(b-1,n//BLOCK)+n%BLOCK+1) for b in choice))


def summarize(selection_path,archive_path):
    measured=read(selection_path)
    assert measured["protocol"]=="kam-layer-budget-v1"
    panels=arrays(archive(archive_path),measured)
    book_ids,books=panels["books"]
    ids,qa=panels["musique"]
    assert len(book_ids)==13 and len(ids)==48
    avg_out=books["output_error"].mean((0,1))
    avg_mass=books["mass"].mean((0,1))
    evidence=measured["schedule"]["evidence-selected"]
    choices=schedule(avg_out,avg_mass,evidence)
    assert tuple(measured["schedule"]["output-selected"])==choices["output-four-twelve"]
    output_gain=avg_out[0]-avg_out[2]
    mass_gain=avg_mass[2]-avg_mass[0]
    mu_evidence_gain=(qa["present_evidence"][:,:,2]-qa["present_evidence"][:,:,0]).mean((0,1))
    correlation = {"mass_output_pearson":float(np.corrcoef(mass_gain,output_gain)[0,1]),
        "mass_output_spearman":float(np.corrcoef(ranks(mass_gain),ranks(output_gain))[0,1]),
        "evidence_output_spearman":float(np.corrcoef(ranks(mu_evidence_gain),ranks(output_gain))[0,1])}
    result={}
    for name,choice in choices.items():
        assert sum(choice)==SLOTS
        result[name]={"schedule":choice,"histogram":Counter(choice),"books":mean_at(choice,books),"mu_dev":mean_at(choice,qa),
                      "sum_causal_token_reads_per_head":{n:key_reads(n,choice) for n in WIDTHS}}
    held, stats=paired_books(books,evidence)
    return {"protocol":PROTOCOL,"selected_plan_sha256":digest(selection_path),"archive_sha256":digest(archive_path),
            "exact_counterexample":validate_counterexample(),"all_calibration_schedules":result,
            "rank_correlation":correlation,"layer_marginals":{"output_error_drop":output_gain.tolist(),
                "retained_mass_gain":mass_gain.tolist(),"mu_evidence_gain":mu_evidence_gain.tolist()},
            "leave_one_book_out":stats,"per_book":dict(zip(book_ids,held,strict=True))}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for n in ("selection","archive","output"):
        cli.add_argument(n,type=Path)
    a=cli.parse_args()
    write_json(a.output,summarize(a.selection,a.archive))
