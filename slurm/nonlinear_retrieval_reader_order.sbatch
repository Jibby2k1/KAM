#!/usr/bin/env bash
#SBATCH --job-name=kam_reader_order
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:02:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/reader_order_v1"
index="$root/fullwiki_native_v1"
export PYTHONPATH="$work:$root/coverage_gate_v1:$index/python" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_reader_order.py" --inputs "$index" --assets "$root/assets" \
    --fixture "$work/fixture.json" --reference "$work/replay.json.gz" \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" --output "$work/order-$SLURM_ARRAY_TASK_ID.json.gz"
