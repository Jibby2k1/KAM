#!/usr/bin/env bash
#SBATCH --job-name=kam_temporal_clock_encoding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=00:07:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/${KAM_TEMPORAL_CLOCK_ENCODING_STAGE:-temporal_clock_encoding_v1}"
export PYTHONPATH="$work:$root/temporal_correction_curriculum_v1:$root/curriculum_geometry_v1:$root/radial_archive_training_v1:$root/radial_training_graph_diagnostic_v1"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(791513 791543 791553)
i="$SLURM_ARRAY_TASK_ID"
shift
python -u "$work/run_kam_temporal_clock_encoding.py" --seed "${seeds[$i]}" --order "$((2*i))" \
    --earlier "$root/radial_archive_training_v1/earlier.json" --later "$root/radial_archive_training_v1/later.json" \
    --output "$work/clocks-$SLURM_ARRAY_JOB_ID-$i.json" "$@"
