#!/usr/bin/env python3
"""Tune sparse prefill launch geometry with parity as a hard gate."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from triton.compiler import CompilationError
from triton.runtime.errors import OutOfResources

from benchmark_kam_causal_proxy_tiles import SHAPES, samples
from benchmark_kam_sparse_live_read import profile
from eval_kam_backbone_shortlist import load
from eval_kam_pretrained import next_logits
from kam_adapter_preservation import SEEDS, digest
from kam_backbone_block_attention import block_attention
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json
from kam_query_window_flash import WindowAttention

PROTOCOL = "kam-sparse-kernel-tuning-v1"
BASELINE = "rectangular-default"
CONFIGS = {f"{kind}-{variant}": (kind == "single-launch", launch)
           for kind in ("rectangular", "single-launch")
           for variant, launch in [("default",None),*((f"{w}w-{s}s",(w,s)) for w in (2,4,8) for s in (1,2,3))]}
MIN_GAIN = .01


def read(path):
    return json.loads(path.read_text())


def apply(config, q, k, v, *, scaling=None, record_routes=False):
    dispatch, prefill = CONFIGS[config]
    return block_attention(q,k,v,scaling,keep_first=True,dispatch_proxy=dispatch,prefill_launch=prefill,
                           record_routes=record_routes)


def select(outcomes):
    medians = {name: float(np.median(arm["cuda_ms"])) for name, arm in outcomes.items()}
    best = min(medians, key=lambda name:(medians[name],name))
    speedup = 1-medians[best]/medians[BASELINE]
    return best if speedup >= MIN_GAIN else BASELINE


@torch.inference_mode()
def calibrate(work):
    torch.manual_seed(294175)
    records, schedule = {}, {}
    for batch, width in SHAPES:
        q,k,v = samples(batch,width)
        reference, routes = apply(BASELINE,q,k,v,record_routes=True)
        checks, options = {}, {}
        for name in CONFIGS:
            try:
                output, chosen = apply(name,q,k,v,record_routes=True)
                same_routes, same_values = torch.equal(routes,chosen),torch.equal(reference,output)
                checks[name] = {"same_routes":same_routes,"same_values":same_values,
                                "max_output_difference":float((output.float()-reference.float()).abs().max())}
                if same_routes and same_values:
                    options[name] = lambda n=name: apply(n,q,k,v)
            except (OutOfResources,CompilationError) as exc:
                checks[name] = {"error":type(exc).__name__,"message":str(exc).splitlines()[-1][:250]}
        assert BASELINE in options
        scores = profile(options,repeats=25)
        shape = f"{batch}x{width}"
        schedule[shape] = select(scores)
        records[shape] = {"checks":checks,"trials":scores}
    write_json(work/"sparse-kernel-calibration.json",{"protocol":PROTOCOL,"gpu":GPU,"records":records,
        "configs":CONFIGS,"minimum_gain":MIN_GAIN,"selected":schedule})


class TunedAttention(WindowAttention):
    def forward(self,module,q,k,v,mask,dropout=0.0,scaling=None,**kwargs):
        if self.mode != "tuned":
            return super().forward(module,q,k,v,mask,dropout,scaling,**kwargs)
        if mask is not None or dropout or not module.is_causal or q.size(-2)!=k.size(-2):
            raise ValueError("Tuned kernels require unpadded causal prefill")
        return apply(self.config,q,k,v,scaling=scaling),None


@torch.inference_mode()
def benchmark(root,source,work):
    seed=SEEDS[0]
    model,stack,projection,_,checkpoint,attention=load(root,source,work,seed,TunedAttention)
    schedule_path=work/"sparse-kernel-calibration.json"
    schedule=read(schedule_path)
    assert schedule["protocol"]==PROTOCOL and schedule["gpu"]==GPU
    assert schedule["selected"]=={shape:select(r["trials"]) for shape,r in schedule["records"].items()}
    fixture_path=work/"transfer-manifest.json"
    fixture=read(fixture_path)
    outcomes={}
    with attention.installed():
        for batch,width in SHAPES:
            name=f"{batch}x{width}"
            source_path=work/f"transfer-{width}.json"
            assert fixture["fixture"][str(width)]["sha256"]==digest(source_path)
            panel=read(source_path)
            attention.config=schedule["selected"][name]
            def action(mode,tokens,mask):
                attention.mode=mode
                return next_logits(model,stack,tokens,mask,projection=projection)[0]

            checked=0
            for start in range(0,8,batch):
                tokens=torch.tensor([p["ids"] for p in panel["retrieval"][start:start+batch]],device="cuda")
                mask=torch.ones_like(tokens,dtype=torch.bool)
                observed=action("tuned",tokens,mask)
                reference=action("protected",tokens,mask)
                assert torch.equal(observed,reference)
                checked+=observed.numel()
            tokens=torch.tensor([p["windows"][0][:-1] for p in panel["pg19"][:batch]],device="cuda")
            mask=torch.ones_like(tokens,dtype=torch.bool)
            methods=("flash","protected","tuned") if attention.config != BASELINE else ("flash","protected")
            outcomes[name]={"checked_logits":checked,"config":attention.config,
                            "timing":profile({a:lambda a=a:action(a,tokens,mask) for a in methods},repeats=15)}
    write_json(work/"sparse-kernel-holdout.json",{"protocol":PROTOCOL,"gpu":GPU,"seed":seed,
            "checkpoint_sha256":checkpoint,"manifest_sha256":digest(fixture_path),"selection_sha256":digest(schedule_path),
            "outcomes":outcomes})


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("root","source","work"):
        cli.add_argument(name,type=Path)
    cli.add_argument("mode",choices=("calibrate","benchmark"))
    a=cli.parse_args()
    assert torch.cuda.get_device_name()==GPU
    if a.mode=="calibrate":
        calibrate(a.work)
    else:
        benchmark(a.root,a.source,a.work)
