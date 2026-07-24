#!/usr/bin/env bash
set -euo pipefail

# A reproducible first grid. Override STEPS, DEVICE, and OUT from the shell.
STEPS="${STEPS:-1500}"
DEVICE="${DEVICE:-auto}"
OUT="${OUT:-outputs/first_grid}"
SEEDS=(7 17 27)
MODELS=(kernel-self memory-only kam dot-transformer dot-hybrid)

for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    kam-train --task copy --model "$model" --seed "$seed" --steps "$STEPS" \
      --device "$DEVICE" --output "$OUT/copy/${model}/seed_${seed}"

    kam-train --task regime --model "$model" --seed "$seed" --steps "$STEPS" \
      --seq-len 64 --device "$DEVICE" --output "$OUT/regime/${model}/seed_${seed}"

    kam-train --task mackey-glass --model "$model" --seed "$seed" --steps "$STEPS" \
      --seq-len 32 --device "$DEVICE" --output "$OUT/mackey_glass/${model}/seed_${seed}"
  done
done

python scripts/aggregate_results.py "$OUT" --output "$OUT/summary.csv"
