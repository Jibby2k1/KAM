#!/usr/bin/env bash
#SBATCH --job-name=kam_guarded_sdpa_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=22G
#SBATCH --time=00:14:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/${KAM_SDPA_TRAIN_STAGE:-typed_guarded_sdpa_train_v1}"
sources="$root/clock_release_mixture_v1"
export PYTHONPATH="$work:$root/typed_guarded_sdpa_v1:$root/rank_cache_graph_v1:$root/temporal_correction_curriculum_v1:$root/curriculum_geometry_v1:$root/radial_archive_training_v1:$root/radial_training_graph_diagnostic_v1"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(791513 791543 791553)
i="$SLURM_ARRAY_TASK_ID"
mode=()
if [[ "${KAM_SDPA_CURRICULUM:-0}" == 1 ]]; then
    mode+=(--curriculum-reference "$root/typed_guarded_sdpa_train_v1/train-53015476-$i.json")
fi
python -u "$work/run_kam_guarded_sdpa_training.py" --seed "${seeds[$i]}" \
  --unemployment "$root/eurostat_vintages_v1/fixture.json" --dallas "$root/oecd_revisions_l4_v1/oecd-observed.json" \
  --industry "$sources/industry-fixture.json" --later-unemployment "$sources/later-unemployment.json" \
  --later-industry "$sources/later-industry.json" --output "$work/train-$SLURM_ARRAY_JOB_ID-$i.json" "${mode[@]}"
