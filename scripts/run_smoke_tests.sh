#!/usr/bin/env bash
set -euo pipefail

python -m kam.train --task copy --model kam --steps 3 --eval-every 1 \
  --train-size 64 --val-size 32 --batch-size 8 --d-model 32 --num-heads 4 \
  --num-layers 1 --num-supports 16 --copy-length 6 --output outputs/smoke_copy

python -m kam.train --task regime --model kernel-self --steps 3 --eval-every 1 \
  --train-size 64 --val-size 32 --batch-size 8 --d-model 32 --num-heads 4 \
  --num-layers 1 --num-supports 16 --seq-len 16 --output outputs/smoke_regime

python -m kam.train --task mackey-glass --model kam --steps 3 --eval-every 1 \
  --series-length 1200 --batch-size 8 --d-model 32 --num-heads 4 \
  --num-layers 1 --num-supports 16 --seq-len 16 --output outputs/smoke_mg

python -m kam.benchmark --seq-lens 16 32 --batch-size 2 --d-model 32 \
  --num-heads 4 --num-layers 1 --num-supports 16 --iterations 2 --warmup 1 \
  --output outputs/smoke_timing
