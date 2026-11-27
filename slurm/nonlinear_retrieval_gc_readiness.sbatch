#!/usr/bin/env bash
#SBATCH --job-name=kam_gc_readiness
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:04:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/live_anchor_gc_readiness_v1"
index="$root/fullwiki_native_v1"
support="$index/support_link_v1"
export PYTHONPATH="$work:$root/coverage_gate_v1:$index/python" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
case "$SLURM_ARRAY_TASK_ID" in
  0) order=(normal collect freeze);;
  1) order=(collect freeze normal);;
  2) order=(freeze normal collect);;
  3) order=(freeze collect normal);;
  *) exit 2;;
esac
for mode in "${order[@]}"; do
    python -u "$work/run_kam_conditioned_union.py" --inputs "$index" --assets "$root/assets" \
        --config "$support/hotpot_fullwiki_native_v1.json" --baseline "$support/baseline.pt" \
        --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833 \
        --heads "$support/heads.pt" --ablations "$root/link_feature_ablation_v1/weights.pt" \
        --fixture "$work/fixture.json" --pools "$work/pools.json.gz" --fitted "$work/fitted.json" \
        --guard-split test --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
        --live-anchor-profile --sentence-reuse-profile --trace-gc --gc-readiness "$mode" \
        --output "$work/profile-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID-$mode.json.gz"
done
