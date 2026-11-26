#!/usr/bin/env bash
#SBATCH --job-name=kam_oecd_masks
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:14:00
set -euo pipefail
module purge
module load pytorch/2.8.0
work="${1:?runtime directory required}"
(cd "$work" && sha256sum --quiet -c staged.sha256)
export PYTHONPATH="$work" OMP_NUM_THREADS=4
arms=(dot_without_absolute radial_without_absolute)
seeds=(12004937 12679517 13791913 15293789)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_bitemporal_decoder.py" --arm "${arms[$((i % 2))]}" --seed "${seeds[$((i / 2))]}" \
  --training-data procedural_archive_topology --steps 3200 --bank 8192 --eval-examples 128 --no-query-clock \
  --archive-fixture "$work/oecd-observed.json" --archive-mask-check --output "$work/eval-$i.json"
