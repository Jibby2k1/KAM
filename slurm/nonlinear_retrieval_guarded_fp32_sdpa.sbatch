#!/usr/bin/env bash
#SBATCH --job-name=kam_guarded_sdpa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=19G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
phase="${2:?refit or run required}"
work="$root/typed_guarded_sdpa_v1"
export PYTHONPATH="$work:$root/temporal_correction_curriculum_v1:$root/curriculum_geometry_v1:$root/radial_archive_training_v1:$root/radial_training_graph_diagnostic_v1"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
seeds=(791513 791543 791553)
model="$work/weights-$i.pt"
case "$phase" in
refit) mode=(--refit-reference "$root/typed_clock_guard_v2/guard-53011213-$i.json") ;;
run) mode=(--output "$work/sdpa-$SLURM_ARRAY_JOB_ID-$i.json") ;;
*) exit 2 ;;
esac
python -u "$work/benchmark_kam_guarded_fp32_sdpa.py" --seed "${seeds[$i]}" --weights "$model" "${mode[@]}"
