#!/usr/bin/env bash
#SBATCH --job-name=kam_radial_train_graph
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:08:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
platform="${2:?platform required}"
work="$root/${3:-radial_training_graph_v1}"
shift "$(( $# >= 3 ? 3 : 2 ))"
export PYTHONPATH="$work" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(729801 729823 729839)
python -u "$work/benchmark_kam_radial_training_graph.py" --checkpoint "$root/bitemporal_sdpa_v1/decoder.pt" \
    --checkpoint-sha256 3e1196f8f88c42e812a05580a52438f167135ff9129e5086af59c14d6ffd777f \
    --seed "${seeds[$SLURM_ARRAY_TASK_ID]}" --order "$SLURM_ARRAY_TASK_ID" \
    --output "$work/graph-$platform-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json" "$@"
