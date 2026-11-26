#!/usr/bin/env bash
#SBATCH --job-name=kam_bitemporal_curriculum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=00:20:00
set -euo pipefail
module purge
module load pytorch/2.8.0
work="${1:?runtime directory required}"
(cd "$work" && sha256sum --quiet -c staged.sha256)
export PYTHONPATH="$work"
arms=(dot_context radial_context dot_without_absolute radial_without_absolute radial_without_absolute)
seeds=(343733 846123 1097879 1582091)
variant=$(( (SLURM_ARRAY_TASK_ID % 10) / 2 ))
arm="${arms[$variant]}"
seed="${seeds[$((SLURM_ARRAY_TASK_ID / 10))]}"
if (( SLURM_ARRAY_TASK_ID % 2 == 0 )); then schedule=short; else schedule=wide; fi
extra=()
suffix=""
if (( variant == 4 )); then extra+=(--no-query-clock); suffix=-no-clock; fi
OMP_NUM_THREADS=4 python -u "$work/run_kam_bitemporal_decoder.py" --seed "$seed" --arm "$arm" \
  --steps 3200 --schedule "$schedule" --extended --eval-examples 512 "${extra[@]}" \
  --output "$work/confirm-$seed-$arm-$schedule$suffix.json"
