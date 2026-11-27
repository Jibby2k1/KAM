#!/usr/bin/env bash
#SBATCH --job-name=kam_role_scope
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=17G
#SBATCH --time=00:09:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/${KAM_LEARNED_SCOPE_STAGE:-eurostat_learned_scope_v1}"
base="$root/eurostat_untyped_v1"
export PYTHONPATH="$work:$base:$root/eurostat_scope_v1:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$root/native_musique_v1:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
mode=()
if [[ "${SLURM_ARRAY_TASK_ID:-0}" == 1 ]]; then
    mode+=(--cue-only)
fi
if [[ "${SLURM_ARRAY_TASK_ID+x}" ]]; then
    mode+=(--force-diagnostics --relative-abstention)
fi
if [[ "${KAM_MIXED_AMBIGUITY:-0}" == 1 ]]; then
    mode+=(--mixed-ambiguous-cues)
fi
python -u "$work/run_kam_eurostat_learned_scope.py" --fixture "$base/fixture.json.gz" --ranked "$base/ranked-53008597.json.gz" \
   --assets "$root/assets" --output "$work/learned-scope-$SLURM_JOB_ID.json.gz" "${mode[@]}"
