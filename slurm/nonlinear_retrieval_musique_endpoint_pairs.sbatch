#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_endpoint_pairs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=23G
#SBATCH --time=00:14:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/${KAM_ENDPOINT_STAGE:-musique_endpoint_pairs_v1}"
export PYTHONPATH="$work:$root/musique_reader_window_v3:$root/musique_reader_composition_v1:$root/native_musique_final_reader_v1:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$root/native_musique_v1:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
mode=()
if [[ "${KAM_ENDPOINT_ROUTED:-0}" == 1 ]]; then
    mode+=(--reference "$root/musique_endpoint_pairs_v1/endpoints-53016711-$i.json.gz")
fi
if [[ "${KAM_ENDPOINT_PROFILE:-0}" == 1 ]]; then
    mode+=(--profile)
fi
python -u "$work/run_kam_musique_endpoint_pairs.py" --fixture "$work/fixture.json.gz" --assets "$root/assets" \
  --shard "$i" --shards 4 --output "$work/endpoints-$SLURM_ARRAY_JOB_ID-$i.json.gz" "${mode[@]}"
