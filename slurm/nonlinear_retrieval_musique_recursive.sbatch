#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_recursive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
baseline_id="${2:?frozen MuSiQue array job id required}"
work="$root/native_musique_recursive_v1"
native="$root/native_musique_v1"
export PYTHONPATH="$work:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_musique_recursive.py" search --corpus "$native/corpus.jsonl.gz" --fixture "$native/fixture.json" \
    --assets "$root/assets" --oracle "$work/oracle.json" --vectors "$native"/embedding-[0-3].pt \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
    --baseline "$native/search-$baseline_id-$SLURM_ARRAY_TASK_ID.json.gz" \
    --output "$work/recursive-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
