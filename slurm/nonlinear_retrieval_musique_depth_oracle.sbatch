#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_depth_oracle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?predict or search required}"
work="$root/native_musique_depth_oracle_v1"
native="$root/native_musique_v1"
export PYTHONPATH="$work:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
case "$mode" in
  predict)
    fit="${3:?known training job ID required}"
    python -u "$work/train_kam_musique_decomposer.py" inference --assets "$root/assets" --fixture "$native/fixture.json" \
        --weights "$root/native_musique_decomposition_v1/decomposer-$fit-$SLURM_ARRAY_TASK_ID.pt" --oracle-depth \
        --output "$work/depth-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
    ;;
  search)
    prior="${3:?baseline array job ID required}"
    generation="${4:?frozen generated depth plan required}"
    python -u "$work/run_kam_musique_learned_search.py" --assets "$root/assets" --corpus "$native/corpus.jsonl.gz" \
        --fixture "$native/fixture.json" --generated "$generation" --vectors "$native"/embedding-[0-3].pt \
        --baseline "$native/search-$prior-$SLURM_ARRAY_TASK_ID.json.gz" \
        --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
        --output "$work/rank-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
    ;;
  *) exit 2 ;;
esac
