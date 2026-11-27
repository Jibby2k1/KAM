#!/usr/bin/env bash
#SBATCH --job-name=kam_iirc_sparse_beam
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:03:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/iirc_sparse_beam_v1"
export PYTHONPATH="$work" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
(cd "$work" && sha256sum --quiet -c staged.sha256)
srun --cpu-bind=cores python -u "$work/benchmark_kam_iirc_sparse_beam.py" \
    --static "$work/train-features.json.gz" --reference "$work/replay.json.gz" --repeats 8 \
    --output "$work/profile-$SLURM_JOB_ID.json.gz"
