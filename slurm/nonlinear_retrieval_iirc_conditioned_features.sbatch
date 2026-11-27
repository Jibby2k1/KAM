#!/usr/bin/env bash
#SBATCH --job-name=kam_iirc_conditioned
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:04:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
split="${2:?train or dev}"
work="$root/iirc_conditioned_features_v1"
index="$root/fullwiki_native_v1"
support="$index/support_link_v1"
extra=()
case "$split" in train) ;; dev) extra=(--first "$work/first.json");; *) exit 2;; esac
export PYTHONPATH="$work:$root/coverage_gate_v1:$index/python" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_iirc_conditioned_features.py" --assets "$root/assets" \
    --config "$support/hotpot_fullwiki_native_v1.json" --baseline "$support/baseline.pt" \
    --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833 \
    --fixture "$work/fit-$split.json" --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
    "${extra[@]}" --output "$work/features-$split-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json"
