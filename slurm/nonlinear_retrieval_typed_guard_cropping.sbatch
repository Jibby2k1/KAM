#!/usr/bin/env bash
#SBATCH --job-name=kam_guard_cropping
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=19G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/${KAM_TYPED_CROP_STAGE:-typed_guard_cropping_v1}"
export PYTHONPATH="$work:$root/temporal_correction_curriculum_v1:$root/curriculum_geometry_v1:$root/radial_archive_training_v1:$root/radial_training_graph_diagnostic_v1"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(791513 791543 791553)
i="$SLURM_ARRAY_TASK_ID"
mode=()
if [[ "${KAM_TYPED_TERMINAL:-0}" == 1 ]]; then
    mode+=(--terminal-readout)
fi
python -u "$work/benchmark_kam_typed_guard_cropping.py" --seed "${seeds[$i]}" \
  --reference "$root/typed_clock_guard_v2/guard-53011213-$i.json" --output "$work/crop-$SLURM_ARRAY_JOB_ID-$i.json" "${mode[@]}"
