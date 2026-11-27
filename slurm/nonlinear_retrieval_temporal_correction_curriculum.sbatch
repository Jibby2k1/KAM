#!/usr/bin/env bash
#SBATCH --job-name=kam_temporal_correction_curriculum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=18G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/temporal_correction_curriculum_v1"
export PYTHONPATH="$work:$root/curriculum_geometry_v1:$root/radial_archive_training_v1:$root/radial_training_graph_diagnostic_v1:$root/fullwiki_native_v1/python"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(684791 684823 684881)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_temporal_correction_curriculum.py" --seed "${seeds[$i]}" --order "$i" \
    --earlier "$root/radial_archive_training_v1/earlier.json" --later "$root/radial_archive_training_v1/later.json" \
    --output "$work/curriculum-$SLURM_ARRAY_JOB_ID-$i.json"
