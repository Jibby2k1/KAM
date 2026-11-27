#!/usr/bin/env bash
#SBATCH --job-name=kam_fullwiki_coarse_live
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=26G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
partition_job="${2:?partition build job ID required}"
work="$root/fullwiki_coarse_live_v1"
index="$root/fullwiki_native_v1"
support="$index/support_link_v1"
old="$root/conditioned_union_v1/score"
export PYTHONPATH="$work:$old:$root/coverage_gate_v1:$index/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
shift 2
python -u "$work/run_kam_conditioned_union.py" --inputs "$index" --assets "$root/assets" \
    --config "$support/hotpot_fullwiki_native_v1.json" --fixture "$old/fixture.json" --pools "$old/pools.json.gz" \
    --fitted "$old/fitted.json" --baseline "$support/baseline.pt" \
    --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833 \
    --heads "$support/heads.pt" --ablations "$root/link_feature_ablation_v1/weights.pt" --guard-split test \
    --corpus-profile-index "$root/fullwiki_coarse_v1/partition-$partition_job.pt" \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
    --output "$work/live-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz" "$@"
