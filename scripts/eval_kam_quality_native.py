#!/usr/bin/env python3
"""Evaluate frozen Llama attention policies on intact QuALITY articles."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmark_kam_sparse_live_read import profile
from eval_kam_backbone_shortlist import load
from eval_kam_pretrained import next_logits
from kam_adapter_preservation import SEEDS, digest
from kam_backbone_block_attention import HEAD_DIM, QUERY_TILE, block_attention
from kam_live_token_factorial import GPU
from kam_math_kl_protocol import write_json
from kam_query_window_flash import WindowAttention
from prepare_kam_long_sparse import load_locked_tokenizer
from prepare_kam_quality_native import LABELS, PROTOCOL

MODES = ("flash", "protected", "tail", "question-only")
RUNS = tuple((s,a) for s in SEEDS for a in MODES)


def read(path):
    return json.loads(path.read_text())


def window(length):
    assert length > 256
    return ((length-256)//QUERY_TILE*QUERY_TILE, length)


class NativeAttention(WindowAttention):
    def forward(self, module, q, k, v, mask, dropout=0.0, scaling=None, **kwargs):
        if self.mode not in ("tail","output-selected"):
            return super().forward(module, q, k, v, mask, dropout, scaling, **kwargs)
        if mask is not None or dropout or not module.is_causal or q.size(-2) != k.size(-2):
            raise ValueError("Native QA scoring requires strictly causal unpadded prefill")
        if self.mode == "output-selected":
            return block_attention(q, k, v, scaling, keep_first=True, budget=self.schedule[module.layer_idx]), None
        return block_attention(q, k, v, scaling, keep_first=True, flash_queries=window(q.size(-2))), None


@torch.inference_mode()
def smoke(work):
    torch.manual_seed(921773)
    checks = {}
    for n in (1025, 4095, 6957):
        def sample(nheads):
            return torch.randn(1,n,nheads,HEAD_DIM,device="cuda",dtype=torch.bfloat16).transpose(1,2)
        q,k,v = sample(8),sample(2),sample(2)
        start, stop = window(n)
        prior = block_attention(q,k,v,keep_first=True)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            full = F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True).transpose(1,2)
            actual = block_attention(q,k,v,keep_first=True,flash_queries=(start,stop))
        assert torch.equal(prior[:,:start],actual[:,:start])
        torch.testing.assert_close(full[:,start:],actual[:,start:],atol=.012,rtol=.04)
        checks[str(n)] = {"prefix_bitwise": True, "tail_queries":stop-start,
                         "tail_max_abs":float((full[:,start:].float()-actual[:,start:].float()).abs().max())}
    write_json(work/"quality-smoke.json",{"gpu":GPU,"checks":checks})


def logits(model, stack, projection, prompts):
    batches = defaultdict(list)
    for i, ids in enumerate(prompts):
        batches[len(ids)].append((i,ids))
    output = [None]*len(prompts)
    for rows in batches.values():
        tokens = torch.tensor([a[1] for a in rows],device="cuda")
        values,_ = next_logits(model,stack,tokens,torch.ones_like(tokens,dtype=torch.bool),projection=projection)
        for (i,_),scores in zip(rows,values.tolist(),strict=True):
            output[i]=scores
    assert all(a is not None for a in output)
    return output


@torch.inference_mode()
def run(root, source, work, mode, index, selection_path=None):
    assert torch.cuda.get_device_name() == GPU
    if mode == "smoke":
        smoke(work)
        return
    if mode == "evaluate":
        seed, arm = RUNS[index] if selection_path is None else (SEEDS[index],"output-selected")
    else:
        seed, arm = SEEDS[0], "profile"
    model, stack, _, _, checkpoint, attention = load(root,source,work,seed,NativeAttention)
    selection_hash = None
    profile_arms = MODES[:3]
    if selection_path is not None:
        selection = read(selection_path)
        assert selection["protocol"]=="kam-layer-budget-v1" and selection["checkpoints"][str(seed)]==checkpoint
        attention.schedule = selection["schedule"]["output-selected"]
        assert attention.schedule.count(4)==attention.schedule.count(12)==16
        selection_hash = digest(selection_path)
        profile_arms = ("flash","protected","output-selected")
    tokenizer, tok_path = load_locked_tokenizer(root)
    projection_ids = [tokenizer.encode(a,add_special_tokens=False) for a in LABELS]
    assert all(len(a)==1 for a in projection_ids) and len(set(a[0] for a in projection_ids))==4
    projection = model.lm_head.weight.index_select(0,torch.tensor([a[0] for a in projection_ids],device="cuda"))
    manifest_path, panel_path = (work/f"quality-{name}.json" for name in ("manifest","inputs"))
    manifest, panel = read(manifest_path),read(panel_path)
    assert (manifest["protocol"],manifest["inputs_sha256"],manifest["tokenizer_sha256"]) == (
        PROTOCOL,digest(panel_path),digest(tok_path))
    rows = manifest["selected"]
    assert len(rows)==len(panel) and all(a["id"]==b["id"] for a,b in zip(rows,panel,strict=True))
    with attention.installed():
        if mode == "evaluate":
            attention.mode = "flash" if arm=="question-only" else arm
            source_kind = "question_only" if arm=="question-only" else "full"
            result = []
            for row, choices in zip(rows,panel,strict=True):
                prompts = choices[source_kind]
                if source_kind=="full":
                    assert [len(a) for a in prompts]==row["lengths"]
                result.append({"id":row["id"],"logits":logits(model,stack,projection,prompts)})
        else:
            order = sorted(range(len(panel)),key=lambda i:(rows[i]["lengths"][0],rows[i]["id"]))
            sample = [order[i*(len(order)-1)//7] for i in range(8)]
            assert len(set(sample))==8
            result = []
            for idx in sample:
                tokens = torch.tensor([panel[idx]["full"][0]],device="cuda")
                mask = torch.ones_like(tokens,dtype=torch.bool)

                def action(policy):
                    attention.mode = policy
                    return next_logits(model,stack,tokens,mask,projection=projection)[0]

                times = profile({a: lambda a=a: action(a) for a in profile_arms},repeats=15)
                result.append({"id":rows[idx]["id"],"tokens":tokens.size(-1),"timings":times})
    name = f"quality-{seed}-{arm}.json" if mode=="evaluate" else "quality-profile.json"
    if mode=="benchmark" and selection_hash:
        name = "quality-budget-profile.json"
    write_json(work/name,{"protocol":PROTOCOL,"quality_manifest_sha256":digest(manifest_path),"seed":seed,"gpu":GPU,
                         "checkpoint_sha256":checkpoint,"arm":arm,"result":result}
                         | ({} if selection_hash is None else {"budget_selection_sha256":selection_hash}))


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("root","source","work"):
        cli.add_argument(name,type=Path)
    cli.add_argument("mode",choices=("smoke","evaluate","benchmark"))
    cli.add_argument("index",type=int,nargs="?",choices=range(len(RUNS)))
    cli.add_argument("--budget-selection",type=Path)
    a=cli.parse_args()
    run(a.root,a.source,a.work,a.mode,a.index,a.budget_selection)
