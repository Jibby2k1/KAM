#!/usr/bin/env python3
"""Check a tile-level retained-mass certificate on controlled key geometries."""
import argparse
from pathlib import Path

import numpy as np

from kam_math_kl_protocol import write_json

PROTOCOL="kam-shortlist-mass-certificate-v1"
DIM=128
BLOCKS=64
GROUPS=4
GROUP_SIZE=16
QUERY_TILE=32
FIXED=(0,BLOCKS-2,BLOCKS-1)
FLEX=5
THRESHOLDS=(.90,.95,.99)
TRIALS=12
REGIMES=("coherent","wide-residual","query-drift","compensated-outlier","independent")


def candidate(q, keys):
    means=keys.mean(axis=2)
    values=(means @ q / np.sqrt(DIM)).max(axis=1)
    values[list(FIXED)]=-np.inf
    order=np.argsort(-values,kind="stable")
    return np.array([*FIXED,*map(int,order[:FLEX])]),means


def construct(rng, regime):
    q0=rng.normal(size=DIM)
    q0*=np.sqrt(DIM)/np.linalg.norm(q0)
    perturb=.1 if regime=="query-drift" else .004
    queries=q0+rng.normal(scale=perturb,size=(QUERY_TILE,DIM))
    queries[0]=q0
    if regime=="independent":
        return queries,rng.normal(size=(BLOCKS,GROUPS,GROUP_SIZE,DIM))
    separation=rng.normal(scale=.35,size=(BLOCKS,GROUPS))
    pockets=rng.choice(np.arange(1,BLOCKS-2),size=3,replace=False)
    separation[pockets]+=np.arange(6.2,7.7,.5).reshape(3,1)
    centroid=rng.normal(scale=.35,size=(BLOCKS,GROUPS,DIM))
    centroid-=(centroid@q0)[:,:,None]*q0/DIM
    centroid+=separation[:,:,None]*q0/np.sqrt(DIM)
    residual=.2 if regime=="wide-residual" else .016
    keys=centroid[:,:,None,:]+rng.normal(scale=residual,size=(BLOCKS,GROUPS,GROUP_SIZE,DIM))
    if regime=="compensated-outlier":
        keep, _=candidate(q0,keys)
        outside=sorted(set(range(BLOCKS))-set(keep))
        block=outside[int(rng.integers(len(outside)))]
        direction=12.5*q0/np.sqrt(DIM)
        keys[block,0,0]+=direction
        keys[block,0,1:]-=direction/(GROUP_SIZE-1)
        new,_=candidate(q0,keys)
        assert np.array_equal(keep,new)
    return queries,keys


def exp_remainder(gap):
    return np.divide(np.expm1(gap)-gap,gap**2,out=.5*np.exp(gap),where=gap>1e-6)


def one(queries,keys):
    q=queries[0]
    root=np.sqrt(DIM)
    keep,mean=candidate(q,keys)
    inside=np.zeros(BLOCKS,dtype=bool)
    inside[keep]=True
    mu=mean@q/root
    residual=keys-mean[:,:,None,:]
    query_norm=np.linalg.norm(q)
    radius=np.linalg.norm(residual,axis=-1).max(axis=-1)*query_norm/root
    lo,hi=(a@np.maximum(q,0)+b@np.minimum(q,0) for a,b in
           ((keys.min(axis=2),keys.max(axis=2)),(keys.max(axis=2),keys.min(axis=2))))
    lo,hi=lo/root,hi/root
    # The narrower intersection of two valid score intervals also contains the mean.
    low=np.minimum(np.maximum(lo,mu-radius),mu)
    high=np.maximum(np.minimum(hi,mu+radius),mu)
    assert np.all(high>=low)
    width=high-low
    mix=np.divide(mu-low,width,out=np.zeros_like(width),where=width>0)
    upper_plain=GROUP_SIZE*np.exp(high)
    upper_chord=GROUP_SIZE*((1-mix)*np.exp(low)+mix*np.exp(high))
    gram=(residual@residual.swapaxes(-1,-2))/GROUP_SIZE
    variance_bound=np.abs(gram).sum(axis=-1).max(axis=-1)*query_norm**2/DIM
    upper_moment=GROUP_SIZE*np.exp(mu)*(1+exp_remainder(high-mu)*variance_bound)
    best_upper=np.minimum(upper_chord,upper_moment)
    compact_upper=GROUP_SIZE*np.exp(mu)*np.minimum(np.cosh(radius),1+exp_remainder(radius)*variance_bound)
    query_drift=np.linalg.norm(queries-q[None,:],axis=-1)
    key_cap=np.linalg.norm(keys,axis=-1).max(axis=2)/root
    inflation=np.exp(query_drift[:,None,None]*key_cap[None,:,:])
    bound={name: (inflation*value[None,:,:]).sum(axis=-1) for name,value in
           (("endpoint",upper_plain),("chord",upper_chord),("chord-or-moment",best_upper),
            ("compact-moment",compact_upper))}
    logits=(queries@keys.reshape(-1,DIM).T/root).reshape(QUERY_TILE,BLOCKS,GROUPS,GROUP_SIZE)
    exact=np.exp(logits).sum(axis=(2,3))
    assert all(np.all(z>=exact*(1-1e-12)) for z in bound.values())
    z_s=exact[:,keep].sum(axis=-1)
    true=z_s/exact.sum(axis=-1)
    lower={key:z_s/(z_s+value[:,~inside].sum(axis=-1)) for key,value in bound.items()}
    assert np.all(lower["endpoint"]<=lower["chord"]*(1+1e-12))
    assert np.all(lower["chord"]<=lower["chord-or-moment"]*(1+1e-12))
    assert np.all(lower["compact-moment"]<=lower["chord-or-moment"]*(1+1e-12))
    assert all(np.all(v<=true*(1+1e-12)) for v in lower.values())
    selected=np.array(keep[3:])
    exact_flex=exact.copy()
    exact_flex[:,list(FIXED)]=-np.inf
    exact_top=np.argsort(-exact_flex,axis=-1,kind="stable")[:,:FLEX]
    true_exact=np.all(np.sort(exact_top,axis=-1)==np.sort(selected),axis=-1)
    ceiling=bound["chord-or-moment"][:,~inside].max(axis=-1)
    certificate=exact[:,selected].min(axis=-1)>ceiling
    assert np.all(~certificate|true_exact)
    return {"actual":true,**lower,
            "actual_exact_route":true_exact,"certified_exact_route":certificate}


def summarize():
    reports={}
    for i, kind in enumerate(REGIMES):
        rng=np.random.default_rng(731913+i)
        traces=[one(*construct(rng,kind)) for _ in range(TRIALS)]
        joined={k:np.stack([t[k] for t in traces]) for k in traces[0]}
        arm={}
        for k in ("actual","endpoint","chord","chord-or-moment","compact-moment"):
            v=joined[k]
            arm[k]={"median":float(np.median(v)),"p10":float(np.quantile(v,.1)),
                    "decisions_above_threshold":{p:int((v>=p).sum()) for p in THRESHOLDS},
                    "all_queries_in_tile_above_threshold":{p:int((v>=p).all(-1).sum()) for p in THRESHOLDS}}
        reports[kind]={"retained_mass":arm,"queries":TRIALS*QUERY_TILE,"tiles":TRIALS,
            "actual_exact_route":int(joined["actual_exact_route"].sum()),
            "certified_exact_route":int(joined["certified_exact_route"].sum())}
    return {"protocol":PROTOCOL,"queries_per_tile":QUERY_TILE,"blocks":BLOCKS,"groups_per_block":GROUPS,
            "group_size":GROUP_SIZE,"head_dimension":DIM,"flexible_slots":FLEX,"reports":reports}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output",type=Path)
    a=cli.parse_args()
    write_json(a.output,summarize())
