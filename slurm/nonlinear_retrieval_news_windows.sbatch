#!/usr/bin/env bash
#SBATCH --job-name=kam_news_windows
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:05:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/news_windows_v1"
export PYTHONPATH="$work:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$root/native_musique_v1:$root/native_twowiki_v1:$root/fullwiki_native_v1/python:$root/coverage_gate_v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_news_windows.py" --fixture "$work/fixture.json.gz" --assets "$root/assets" \
    --corpus "$work/corpus.json" --shard "$i" --shards "$SLURM_ARRAY_TASK_COUNT" \
    --output "$work/windows-$SLURM_ARRAY_JOB_ID-$i.json.gz"
