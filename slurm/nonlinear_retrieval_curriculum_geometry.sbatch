#!/usr/bin/env bash
#SBATCH --job-name=kam_curriculum_geometry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=17G
#SBATCH --time=00:18:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/curriculum_geometry_v1"
export PYTHONPATH="$work:$root/radial_training_graph_diagnostic_v1" OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(669211 669269 669283)
geometries=(dot radial)
i="$SLURM_ARRAY_TASK_ID"
seed_index="$(( i % 3 ))"
choice="${geometries[$(( i / 3 ))]}"
python -u "$work/run_kam_radial_curriculum_order.py" --seed "${seeds[$seed_index]}" --order "$(( seed_index % 2 ))" \
    --geometry "$choice" --execution eager --output "$work/geometry-$choice-$SLURM_ARRAY_JOB_ID-$seed_index.json"
