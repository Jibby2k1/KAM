#!/usr/bin/env python3
"""Verify optimizer work, exposure and gradient checks without reading test scores."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from prepare_kam_sparse_adaptation import PROTOCOL, STEPS
from summarize_kam_evidence_pins import archive

ARMS=("flash","protected")
WIDTH=4096


def read(path):
    return json.loads(path.read_text())


def summarize(work, manifest_path, books_path, prior_path):
    manifest, books=map(read,(manifest_path,books_path))
    mh=digest(manifest_path)
    assert manifest["protocol"]==PROTOCOL and manifest["source_books_sha256"]==digest(books_path)
    assert books["source"]=="PG-19 train" and len(books["objects"])==len(manifest["books"])==96
    assert {Path(r["name"]).stem for r in books["objects"]}=={r["id"] for r in manifest["books"]}
    assert (manifest["width"],manifest["steps"])==(WIDTH,STEPS)
    assert manifest["available_windows"]==sum(len(b["starts"]) for b in manifest["books"])
    origin=archive(prior_path)
    hashes={}
    checks_p=work/"flex-gradient.json"
    smoke=read(checks_p)
    hashes[checks_p.stem]=digest(checks_p)
    assert set(smoke["checks"])=={"1024","4096"}
    assert all(r["max_abs_forward_delta"]<=.016 for r in smoke["checks"].values())
    assert all(v<.02 for v in smoke["checks"]["1024"]["relative_gradient_error"].values())
    gpu=smoke["gpu"]
    pairs={}
    averages={a:[] for a in ARMS}
    for seed in SEEDS:
        checkpoint=json.loads(origin[f"backbone-{seed}-0"])["checkpoint_sha256"]
        path=work/f"sparse-teacher-{seed}.json"
        teacher=read(path)
        hashes[path.stem]=digest(path)
        fields=("protocol","seed","gpu","source_checkpoint_sha256","training_manifest_sha256")
        common=(PROTOCOL,seed,gpu,checkpoint,mh)
        assert tuple(teacher[k] for k in fields)==common and teacher["shape"]==[128,16,4096]
        match=("exposure_sha256","initial_adapter_sha256","parameters","training_width","steps","tail_targets","kl_stride")
        outcomes={}
        for arm in ARMS:
            path=work/f"sparse-train-{seed}-{arm}.json"
            row=read(path)
            hashes[path.stem]=digest(path)
            assert tuple(row[k] for k in fields)==common and all(row[k]==teacher[k] for k in match)
            assert row["training_arm"]==arm and row["teacher_sha256"]==teacher["sha256"]
            assert (row["train_tokens"],row["peak_lr"],row["kl"])==(STEPS*WIDTH,8e-5,.5)
            steps=[r["step"] for r in row["history"]]
            assert steps==[1,8,16,32,64,96,128]
            seconds=np.array([r["seconds"] for r in row["history"]])
            assert np.isfinite(seconds).all() and (np.diff(seconds)>0).all() and seconds[0]>0
            tokens_sec=float(row["train_tokens"]/seconds[-1])
            outcomes[arm]={"checkpoint_sha256":row["adapter_sha256"],"train_tokens":row["train_tokens"],
                "peak_allocated_bytes":row["peak_allocated_bytes"],"seconds":float(seconds[-1]),"tokens_per_second":tokens_sec}
            averages[arm].append(tokens_sec)
            if seed==SEEDS[0]:
                pilot_path=work/f"sparse-train-{seed}-{arm}-pilot.json"
                pilot=read(pilot_path)
                hashes[pilot_path.stem]=digest(pilot_path)
                assert tuple(pilot[k] for k in fields)==common and all(pilot[k]==teacher[k] for k in match)
                assert pilot["train_tokens"]==3*WIDTH and [r["step"] for r in pilot["history"]]==[1,2,3]
                assert all(np.isfinite(r["preclip_norm"]) and r["preclip_norm"]>0 for r in pilot["history"])
        pairs[seed]={"match":{k:teacher[k] for k in match},"teacher_sha256":teacher["sha256"],
            "initial_checkpoint_sha256":checkpoint,"arms":outcomes}
    return {"protocol":PROTOCOL,"gpu":gpu,"train_manifest_sha256":mh,"source_books_sha256":digest(books_path),
            "raw_sha256":hashes,"train_loop_tokens_per_second":{a:float(np.median(v)) for a,v in averages.items()},
            "verified_training_runs":6,"gradient_checks":smoke["checks"],"per_seed":pairs}


if __name__=="__main__":
    c=argparse.ArgumentParser(description=__doc__)
    for x in ("work","manifest","books","prior","output"):
        c.add_argument(x,type=Path)
    a=c.parse_args()
    write_json(a.output,summarize(a.work,a.manifest,a.books,a.prior))
