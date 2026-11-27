#!/usr/bin/env bash
#SBATCH --job-name=kam_packed_clock_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
device="${2:?GPU name required}"
work="$root/packed_clock_inference_v1"
export PYTHONPATH="$work:$root/temporal_correction_curriculum_v1:$root/curriculum_geometry_v1:$root/radial_archive_training_v1:$root/radial_training_graph_diagnostic_v1"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(751691 751702 751733)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_packed_clock_inference.py" --seed "${seeds[$i]}" \
    --checkpoint "$root/bitemporal_graph_v1/decoder.pt" --sha256 3e1196f8f88c42e812a05580a52438f167135ff9129e5086af59c14d6ffd777f \
    --output "$work/inference-$device-$SLURM_ARRAY_JOB_ID-$i.json"
