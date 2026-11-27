#!/usr/bin/env bash
#SBATCH --job-name=kam_lexical_prewarm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:04:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
export work="$root/lexical_prewarm_v1"
export lexical="$root/fullwiki_native_v1/lexical"
export jobdir="$work/jobs/$SLURM_JOB_ID"
export PYTHONPATH="$root/fullwiki_native_v1/lexical_segment_v1/python:$work" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p "$jobdir"
(cd "$work" && sha256sum --quiet -c staged.sha256)
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpu-bind=cores bash -c '
    set -euo pipefail
    python -u "$work/benchmark_kam_lexical_prewarm.py" --fixture "$work/fixture.json.gz" --lexical "$lexical" \
        --working-dir "$jobdir" --rank "$SLURM_PROCID" --output "$jobdir/prewarm-$SLURM_PROCID.json"
'
