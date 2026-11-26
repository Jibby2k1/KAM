#!/usr/bin/env bash
#SBATCH --job-name=kam_bitemporal_decoder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=18G
#SBATCH --time=00:18:00
set -euo pipefail
module purge
module load pytorch/2.8.0
work="${1:?runtime directory required}"
steps="${2:?number of steps required}"
group="${3:?schedule label required}"
(cd "$work" && sha256sum --quiet -c staged.sha256)
export PYTHONPATH="$work"
arms=(dot_context cosine_context cosine_fixed4 cosine_learned radial_context dot_hybrid radial_hybrid
      dot_without_absolute radial_without_absolute dot_without_position radial_without_position)
seeds=(495939 862047 249173 883049 1034797 1099321 1439127 1999417)
arm="${arms[$((SLURM_ARRAY_TASK_ID % 11))]}"
seed="${seeds[$((SLURM_ARRAY_TASK_ID / 11))]}"
OMP_NUM_THREADS=4 python -u "$work/run_kam_bitemporal_decoder.py" --seed "$seed" --arm "$arm" --steps "$steps" \
  --output "$work/$group-$seed-$arm.json"
