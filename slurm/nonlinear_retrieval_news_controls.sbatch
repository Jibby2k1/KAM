#!/usr/bin/env bash
#SBATCH --job-name=kam_news_controls
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
work="$root/dataset_breadth_v1"
export PYTHONPATH="$work:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c news-controls.sha256)
python -u "$work/run_kam_news_reader_controls.py" --assets "$root/assets" \
 --base "$root/assets/reader_base_7bcac572ce56db69c1ea7c8af255c5d7c9672fc2" \
 --fixture "$work/news-fixture.json" --corpus "$work/news-corpus.json" \
 --replays "$work/news-0.json" "$work/news-1.json" \
 --output "$work/news-controls-$SLURM_ARRAY_TASK_ID.json" \
 --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT"
