#!/usr/bin/env bash
set -euo pipefail

export LOCAL_OVERNIGHT_HOURS="${LOCAL_OVERNIGHT_HOURS:-10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MANIFEST="${1:-results/phase3/manifests/local_overnight.parquet}"
RUN_ROOT="${2:-results/phase3/local_overnight}"

python -m kam.phase3 preflight --mode local --manifest "$MANIFEST"
python -m kam.phase3 run-manifest \
  --manifest "$MANIFEST" \
  --run-root "$RUN_ROOT" \
  --max-parallel-gpu-jobs 1 \
  --walltime-hours "$LOCAL_OVERNIGHT_HOURS" \
  --resume
python -m kam.phase3 aggregate --run-root "$RUN_ROOT"
python -m kam.phase3 report --run-root "$RUN_ROOT"
