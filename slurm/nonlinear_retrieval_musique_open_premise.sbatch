#!/usr/bin/env bash
#SBATCH --job-name=kam_open_premise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=23G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/${KAM_OPEN_STAGE:-musique_endpoint_open_v1}"
fixed="$root/musique_endpoint_profile_v1"
native="$root/native_musique_v1"
export PYTHONPATH="$work:$fixed:$root/musique_reader_window_v3:$root/musique_reader_composition_v1:$root/native_musique_final_reader_v1:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
mode=()
if [[ "${KAM_OPEN_BOTH:-0}" == 1 ]]; then
    mode+=(--fully-open)
fi
if [[ "${KAM_OPEN_RERANK:-0}" == 1 ]]; then
    mode+=(--rerank-frozen "$root/musique_open_both_v1/open-53020278-$i.json.gz")
fi
python -u "$work/run_kam_musique_open_premise.py" --fixture "$fixed/fixture.json.gz" --assets "$root/assets" \
  --previous "$fixed/endpoints-53018115-$i.json.gz" --native-fixture "$native/fixture.json" --corpus "$native/corpus.jsonl.gz" \
  --lexical "$native/lexical" --vectors "$native"/embedding-{0,1,2,3}.pt --shard "$i" --output "$work/open-$SLURM_ARRAY_JOB_ID-$i.json.gz" "${mode[@]}"
