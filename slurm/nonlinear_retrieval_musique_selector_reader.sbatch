#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_selector_reader
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=00:16:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
baseline="${2:?baseline array job ID required}"
work="$root/${KAM_MUSIQUE_SELECTOR_READER_STAGE:-native_musique_selector_reader_v1}"
native="$root/native_musique_v1"
learned="$root/native_musique_learned_search_v1"
export PYTHONPATH="$work:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
shift 2
rankings=()
for job in "$@"; do rankings+=("$learned/learned-$job-$SLURM_ARRAY_TASK_ID.json.gz"); done
python -u "$work/run_kam_musique_final_reader.py" run --assets "$root/assets" --corpus "$native/corpus.jsonl.gz" \
    --fixture "$native/fixture.json" --labels "$root/native_musique_final_reader_v1/answers.json" \
    --vectors "$native"/embedding-[0-3].pt --baseline "$native/search-$baseline-$SLURM_ARRAY_TASK_ID.json.gz" \
    --learned "${rankings[@]}" --selector-picks "$work/picks.json" --selector-policy "$work/policy.json" \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
    --output "$work/read-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
