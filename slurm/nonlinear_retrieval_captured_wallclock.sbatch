#!/usr/bin/env bash
#SBATCH --job-name=kam_captured_wallclock
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
device="${2:?device name required}"
work="$root/captured_wallclock_v1"
export PYTHONPATH="$work:$root/curriculum_wallclock_v2:$root/curriculum_geometry_v1:$root/radial_training_graph_diagnostic_v1"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(774157 774181 774221)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_captured_wallclock.py" --seed "${seeds[$i]}" --order "$i" \
    --output "$work/train-$device-$SLURM_ARRAY_JOB_ID-$i.json"
