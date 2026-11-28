#!/usr/bin/env python3
"""Verify frozen kernel choices and replay CUDA and full-reader timing."""
import argparse
import json
from pathlib import Path

import numpy as np

from kam_adapter_preservation import SEEDS, digest
from kam_math_kl_protocol import write_json
from summarize_kam_causal_proxy_tiles import timing
from summarize_kam_evidence_pins import archive

PROTOCOL = "kam-sparse-kernel-tuning-v1"
BASE = "rectangular-default"


def read(path):
    return json.loads(path.read_text())


def summarize(calibration, holdout, fixture, previous):
    c,h=map(read,(calibration,holdout))
    assert c["protocol"]==h["protocol"]==PROTOCOL and c["gpu"]==h["gpu"]
    assert h["manifest_sha256"]==digest(fixture) and h["selection_sha256"]==digest(calibration)
    assert h["seed"]==SEEDS[0] and c["minimum_gain"]==.01
    before=json.loads(archive(previous)[f"transfer-{SEEDS[0]}-flash"])
    assert h["checkpoint_sha256"]==before["checkpoint_sha256"]
    assert len(c["configs"])==20 and len(c["records"])==4
    records={}
    for shape in sorted(c["records"]):
        selected=c["selected"][shape]
        k=c["records"][shape]
        assert set(k["checks"])==set(c["configs"])
        assert set(k["trials"])=={arm for arm,r in k["checks"].items() if r.get("same_routes") and r.get("same_values")}
        kernel=timing(k["trials"],25)
        base=kernel[BASE]["cuda_p50_ms"]
        best=min(kernel,key=lambda name:(kernel[name]["cuda_p50_ms"],name))
        choose=best if 1-kernel[best]["cuda_p50_ms"]/base>=c["minimum_gain"] else BASE
        assert selected==choose
        row=h["outcomes"][shape]
        assert row["checked_logits"]==16 and row["config"]==selected
        runtime=timing(row["timing"],15)
        assert set(runtime)==({"flash","protected","tuned"} if selected!=BASE else {"flash","protected"})
        tuned=runtime.get("tuned",runtime["protected"])
        records[shape]={"selected":selected,"exact_kernel_options":len(kernel),
            "cuda_time_saved_kernel":1-kernel[selected]["cuda_p50_ms"]/base,
            "wall_time_saved_full":1-tuned["wall_p50_ms"]/runtime["protected"]["wall_p50_ms"],
            "wall_time_saved_vs_flash":1-tuned["wall_p50_ms"]/runtime["flash"]["wall_p50_ms"],
            "kernel_timing":kernel,"full_model_timing":runtime}
    return {"protocol":PROTOCOL,"gpu":c["gpu"],"fixture_sha256":h["manifest_sha256"],
            "calibration_sha256":digest(calibration),"holdout_sha256":digest(holdout),"shapes":records}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ("calibration","holdout","fixture","previous","output"):
        cli.add_argument(key,type=Path)
    a=cli.parse_args()
    write_json(a.output,summarize(a.calibration,a.holdout,a.fixture,a.previous))
