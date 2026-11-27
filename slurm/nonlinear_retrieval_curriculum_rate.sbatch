#!/usr/bin/env bash
#SBATCH --job-name=kam_curriculum_rate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:09:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/curriculum_rate_v1"
export PYTHONPATH="$work:$root/curriculum_wallclock_v2:$root/curriculum_geometry_v1:$root/radial_training_graph_diagnostic_v1"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(669211 669269 669283)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_curriculum_rate.py" --seed "${seeds[$i]}" --order "$i" --output "$work/rate-$SLURM_ARRAY_JOB_ID-$i.json"
