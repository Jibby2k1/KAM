#!/usr/bin/env bash
#SBATCH --job-name=kam_iirc_reader
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:03:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/iirc_reader_v1"
export PYTHONPATH="$work:$root/coverage_gate_v1:$root/fullwiki_native_v1/python" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_iirc_reader.py" --assets "$root/assets" --fixture "$work/reader.json" \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
    --output "$work/reader-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json"
