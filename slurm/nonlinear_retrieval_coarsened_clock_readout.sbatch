#!/usr/bin/env bash
#SBATCH --job-name=kam_coarse_clock_readout
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=19G
#SBATCH --time=00:07:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/${KAM_COARSE_CLOCK_STAGE:-coarsened_clock_readout_v1}"
archives="$root/clock_release_mixture_v1"
export PYTHONPATH="$work:$root/typed_guarded_sdpa_v1:$root/temporal_correction_curriculum_v1:$root/curriculum_geometry_v1:$root/radial_archive_training_v1:$root/radial_training_graph_diagnostic_v1"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(791513 791543 791553)
i="$SLURM_ARRAY_TASK_ID"
mode=()
if [[ "${KAM_COARSE_OUTPUT_FILTER:-0}" == 1 ]]; then
    mode+=(--reference-coarse "$root/coarsened_clock_readout_v1/coarse-read-53019080-$i.json")
fi
python -u "$work/run_kam_coarsened_clock_readout.py" --seed "${seeds[$i]}" \
  --weights "$root/typed_guarded_sdpa_v1/weights-$i.pt" --reference "$root/typed_clock_guard_v2/guard-53011213-$i.json" \
  --unemployment "$root/eurostat_vintages_v1/fixture.json" --industry "$archives/industry-fixture.json" \
  --later-unemployment "$archives/later-unemployment.json" --later-industry "$archives/later-industry.json" \
  --output "$work/coarse-read-$SLURM_ARRAY_JOB_ID-$i.json" "${mode[@]}"
