#!/usr/bin/env bash
#SBATCH --job-name=kam_archive_factors
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
work="${1:?runtime directory required}"
(cd "$work" && sha256sum --quiet -c staged.sha256)
export PYTHONPATH="$work"
seeds=(9531719 10255849 10805317)
seed="${seeds[$((SLURM_ARRAY_TASK_ID / 8))]}"
condition=$((SLURM_ARRAY_TASK_ID % 8))
args=()
if (( condition & 1 )); then args+=(--spread-effective-dates); fi
if (( condition & 2 )); then args+=(--mostly-earlier-revisions); fi
if (( condition & 4 )); then args+=(--allow-duplicate-distractor-payloads); fi
OMP_NUM_THREADS=4 python -u "$work/run_kam_bitemporal_decoder.py" --seed "$seed" --arm radial_without_absolute \
  --steps 3200 --bank 8192 --eval-examples 256 --no-query-clock --training-data procedural_archive_topology \
  --archive-fixture "$work/archive-fixture.json" "${args[@]}" --output "$work/eval-$seed-$condition.json"
