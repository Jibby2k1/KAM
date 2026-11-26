#!/usr/bin/env bash
#SBATCH --job-name=kam_vintage_global
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=18G
#SBATCH --time=00:20:00
set -euo pipefail
module purge
module load pytorch/2.8.0
if [ -n "${KAM_ENV_SETUP:-}" ]; then
    source "$KAM_ENV_SETUP"
fi
root="${1:?runtime directory required}"
mode="${2:?mode required}"
label="${3:?platform label required}"
work="$root/vintage_global_v1"
export PYTHONPATH="$work"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
shard="${SLURM_ARRAY_TASK_ID:-0}"
python -u "$work/run_kam_vintage_global.py" "$mode" --assets "$root/assets" \
 --fixture "$root/vintage_revisions_v1/fed-vintage-fixture.json" --catalog "$work/fed-vintage-catalog.json" \
 --bank "$work" --output "$work/$mode-$label-$shard.json" --shard "$shard" --shards 8
