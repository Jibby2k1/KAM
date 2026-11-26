#!/usr/bin/env bash
#SBATCH --job-name=kam_iirc_pair
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:03:00
set -euo pipefail
root="${1:?runtime directory required}"
work="$root/iirc_pair_v1"
index="$root/fullwiki_native_v1"
support="$index/support_link_v1"
export PYTHONPATH="$work:$root/coverage_gate_v1:$index/python" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_iirc_pair.py" --assets "$root/assets" \
    --config "$support/hotpot_fullwiki_native_v1.json" --baseline "$support/baseline.pt" \
    --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833 \
    --fixture "$work/fixture.json" --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
    --output "$work/routing-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json"
