#!/usr/bin/env bash
#SBATCH --job-name=kam_timeqa_collisions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=18G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
if [ -n "${KAM_ENV_SETUP:-}" ]; then
    source "$KAM_ENV_SETUP"
fi
root="${1:?runtime directory required}"
variant="${2:?variant required}"
work="$root/timeqa_collisions_v1"
source_study="$root/timeqa_contrasts_v1"
export PYTHONPATH="$work:$source_study"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_timeqa_contrasts.py" eval --assets "$root/assets" \
  --fixture "$source_study/timeqa-fixture.json" --weights "$source_study/weights.pt" \
  --noise-variant "$variant" --output "$work/$variant-$SLURM_ARRAY_TASK_ID.json" \
  --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT"
