#!/usr/bin/env bash
#SBATCH --job-name=kam_vintage_stream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=6G
#SBATCH --time=00:08:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/vintage_stream_v1"
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(795173 230619 923918 741885)
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"
python -u "$work/run_kam_vintage_stream.py" --catalog "$root/vintage_global_v1/fed-vintage-catalog.json" \
  --seed "$seed" --queries 200000 --output "$work/seed-$seed.json"
