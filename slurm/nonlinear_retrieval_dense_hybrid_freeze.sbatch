#!/usr/bin/env bash
#SBATCH --job-name=kam_dense_hybrid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:03:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/dense_hybrid_anchor_v1/freeze"
index="$root/fullwiki_native_v1"
support="$index/support_link_v1"
export PYTHONPATH="$work:$root/coverage_gate_v1:$index/python" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_dense_hybrid_freeze.py" --inputs "$index" --assets "$root/assets" \
    --config "$support/hotpot_fullwiki_native_v1.json" --fixture "$work/fixture.json" --output "$work" \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT"
