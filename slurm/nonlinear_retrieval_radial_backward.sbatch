#!/usr/bin/env bash
#SBATCH --job-name=kam_radial_backward
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:06:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
platform="${2:?b200 or l4}"
work="$root/radial_backward_v1"
export PYTHONPATH="$work" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/benchmark_kam_radial_backward.py" --checkpoint "$root/bitemporal_sdpa_v1/decoder.pt" \
    --checkpoint-sha256 3e1196f8f88c42e812a05580a52438f167135ff9129e5086af59c14d6ffd777f \
    --output "$work/benchmark-$platform-$SLURM_JOB_ID.json"
