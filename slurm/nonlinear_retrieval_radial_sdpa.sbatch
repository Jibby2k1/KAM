#!/usr/bin/env bash
#SBATCH --job-name=kam_radial_sdpa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
work="${1:?runtime directory required}"
mode="${2:?fit or benchmark required}"
(cd "$work" && sha256sum --quiet -c staged.sha256)
export PYTHONPATH="$work" OMP_NUM_THREADS=4
if [[ "$mode" == fit ]]; then
  python -u "$work/run_kam_bitemporal_decoder.py" --arm radial_without_position --seed 6430511 \
    --steps 3200 --schedule wide --no-query-clock --eval-examples 512 --checkpoint "$work/decoder.pt" \
    --output "$work/fit.json"
else
  shift 2
  python -u "$work/benchmark_kam_radial_sdpa.py" --checkpoint "$work/decoder.pt" --output "$work/$mode.json" "$@"
fi
