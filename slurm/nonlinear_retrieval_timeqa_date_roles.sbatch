#!/usr/bin/env bash
#SBATCH --job-name=kam_timeqa_date_roles
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:15:00
set -euo pipefail
module purge
module load pytorch/2.8.0
if [ -n "${KAM_ENV_SETUP:-}" ]; then
    source "$KAM_ENV_SETUP"
fi
root="${1:?runtime directory required}"
mode="${2:?mode required}"
variant="${3:-donor}"
work="$root/timeqa_date_roles_v1"
train="$root/timeqa_contrasts_v1"
human="$root/timeqa_human_v1"
export PYTHONPATH="$work"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
shard="${SLURM_ARRAY_TASK_ID:-0}"
python -u "$work/run_kam_timeqa_date_roles.py" "$mode" --assets "$root/assets" \
 --training-fixture "$train/timeqa-fixture.json" --human-fixture "$human/timeqa-human-fixture.json" \
 --reference-weights "$train/weights.pt" --weights "$work/weights.pt" --variant "$variant" \
 --output "$work/$mode-$variant-$shard.json" --shard "$shard" --shards "${SLURM_ARRAY_TASK_COUNT:-1}"
