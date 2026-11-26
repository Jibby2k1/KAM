#!/usr/bin/env bash
#SBATCH --job-name=kam_dataset_breadth
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
if [ -n "${KAM_ENV_SETUP:-}" ]; then
    source "$KAM_ENV_SETUP"
fi
root="${1:?runtime directory required}"
dataset="${2:?dataset required}"
work="$root/dataset_breadth_v1"
export PYTHONPATH="$work:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
opts=()
if [ "$dataset" = news ]; then
    opts+=(--corpus "$work/news-corpus.json")
fi
python -u "$work/run_kam_dataset_breadth.py" --assets "$root/assets" \
 --config "$work/hotpot_fullwiki_native_v1.json" --fixture "$work/$dataset-fixture.json" \
 --baseline "$work/baseline.pt" --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833 \
 --heads "$work/heads.pt" --output "$work/$dataset-$SLURM_ARRAY_TASK_ID.json" \
 --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" "${opts[@]}"
