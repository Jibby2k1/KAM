#!/usr/bin/env bash
#SBATCH --job-name=kam_serving_prewarm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:04:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
export work="$root/shared_prewarm_serving_v1"
export index="$root/fullwiki_native_v1"
export assets="$root/assets"
export support="$index/support_link_v1"
export ablations="$root/link_feature_ablation_v1/weights.pt"
export jobdir="$work/jobs/$SLURM_JOB_ID"
export PYTHONPATH="$work:$root/coverage_gate_v1:$index/python" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$jobdir"
(cd "$work" && sha256sum --quiet -c staged.sha256)
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c '
    set -euo pipefail
    python -u "$work/run_kam_conditioned_union.py" --inputs "$index" --assets "$assets" \
        --config "$support/hotpot_fullwiki_native_v1.json" --baseline "$support/baseline.pt" \
        --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833 \
        --heads "$support/heads.pt" --ablations "$ablations" \
        --fixture "$work/fixture.json" --pools "$work/pools.json.gz" --fitted "$work/fitted.json" \
        --guard-split test --shard 0 --shards 1 --index-prewarm-rank "$SLURM_PROCID" --index-prewarm-jobdir "$jobdir" \
        --output "$jobdir/serving-$SLURM_PROCID.json.gz"
'
