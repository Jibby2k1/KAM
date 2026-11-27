#!/usr/bin/env bash
#SBATCH --job-name=kam_rank_cache_graph
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=17G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
device="${2:?device label required}"
work="$root/${KAM_RANK_GRAPH_STAGE:-rank_cache_graph_v1}"
export PYTHONPATH="$work:$root/temporal_correction_curriculum_v1:$root/curriculum_geometry_v1:$root/radial_archive_training_v1:$root/radial_training_graph_diagnostic_v1"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(791513 791543 791553)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_rank_cache_graph.py" --seed "${seeds[$i]}" --output "$work/captured-$device-$SLURM_ARRAY_JOB_ID-$i.json"
