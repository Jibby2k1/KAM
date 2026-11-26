#!/usr/bin/env bash
#SBATCH --job-name=kam_frozen_sequential_oracle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:03:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/frozen_sequential_oracle_v1"
index="$root/fullwiki_native_v1"
support="$index/support_link_v1"
export PYTHONPATH="$work:$root/coverage_gate_v1:$index/python" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_frozen_sequential_oracle.py" --inputs "$index" --assets "$root/assets" \
  --config "$support/hotpot_fullwiki_native_v1.json" --fixture "$work/fixture.json" \
  --pools "$work/pools.json.gz" --broad-replay "$work/replay.json.gz" --baseline "$support/baseline.pt" \
  --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833 \
  --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" --output "$work/eval-$SLURM_ARRAY_TASK_ID.json.gz"
