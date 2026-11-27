#!/usr/bin/env bash
#SBATCH --job-name=kam_radial_shape_graphs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:04:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
device="${2:?device name required}"
work="$root/radial_shape_graphs_v1"
export PYTHONPATH="$work:$root/radial_training_graph_diagnostic_v1"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(729801 729823 729839)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/benchmark_kam_radial_shape_graphs.py" --checkpoint "$root/bitemporal_sdpa_v1/decoder.pt" \
    --checkpoint-sha256 3e1196f8f88c42e812a05580a52438f167135ff9129e5086af59c14d6ffd777f \
    --seed "${seeds[$i]}" --order "$i" --output "$work/shape-$device-$SLURM_ARRAY_JOB_ID-$i.json"
