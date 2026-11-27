#!/usr/bin/env bash
#SBATCH --job-name=kam_curriculum_wallclock
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:14:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/${KAM_CURRICULUM_WALLCLOCK_STAGE:-curriculum_wallclock_v2}"
export PYTHONPATH="$work:$root/curriculum_geometry_v1:$root/radial_training_graph_diagnostic_v1" OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(669211 669269 669283)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_curriculum_wallclock.py" --seed "${seeds[$i]}" --order "$i" \
    --output "$work/wallclock-$SLURM_ARRAY_JOB_ID-$i.json"
