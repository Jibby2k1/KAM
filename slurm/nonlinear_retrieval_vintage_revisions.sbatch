#!/usr/bin/env bash
#SBATCH --job-name=kam_vintage_revisions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=18G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load pytorch/2.8.0
if [ -n "${KAM_ENV_SETUP:-}" ]; then
    source "$KAM_ENV_SETUP"
fi
root="${1:?runtime directory required}"
mode="${2:?mode required}"
work="$root/vintage_revisions_v1"
export PYTHONPATH="$work"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
shard="${SLURM_ARRAY_TASK_ID:-0}"
python -u "$work/run_kam_vintage_revisions.py" "$mode" --assets "$root/assets" \
 --fixture "$work/fed-vintage-fixture.json" --weights "$work/weights.pt" --output "$work/$mode-$shard.json" \
 --shard "$shard" --shards "${SLURM_ARRAY_TASK_COUNT:-1}"
