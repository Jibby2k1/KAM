#!/usr/bin/env bash
#SBATCH --job-name=kam_radial_curriculum_order
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=00:22:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/radial_curriculum_order_v1"
export PYTHONPATH="$work:$root/radial_training_graph_diagnostic_v1" OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(669211 669269 669283)
index="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_radial_curriculum_order.py" --seed "${seeds[$index]}" --order "$(( index % 2 ))" \
    --output "$work/curriculum-$SLURM_ARRAY_JOB_ID-$index.json"
