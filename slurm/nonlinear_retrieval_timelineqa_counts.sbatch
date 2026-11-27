#!/usr/bin/env bash
#SBATCH --job-name=kam_timelineqa_counts
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=14G
#SBATCH --time=00:05:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/timelineqa_counts_v1"
export PYTHONPATH="$work:$root/native_twowiki_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_timelineqa_counts.py" --fixture "$work/fixture.json.gz" --assets "$root/assets" \
    --shard "$i" --shards "$SLURM_ARRAY_TASK_COUNT" --output "$work/counts-$SLURM_ARRAY_JOB_ID-$i.json.gz"
