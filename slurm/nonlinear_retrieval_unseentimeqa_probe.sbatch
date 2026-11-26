#!/usr/bin/env bash
#SBATCH --job-name=kam_unseentimeqa_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load pytorch/2.8.0
if [ -n "${KAM_ENV_SETUP:-}" ]; then
    source "$KAM_ENV_SETUP"
fi
root="${1:?runtime directory required}"
work="$root/unseentimeqa_probe_v1"
(cd "$work" && sha256sum --quiet -c staged.sha256)
export PYTHONPATH="$work"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 TOKENIZERS_PARALLELISM=false
python -u "$work/run_kam_unseentimeqa_probe.py" --assets "$root/assets" --fixture "$work/unseentimeqa-probe-fixture.json" \
  --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" --output "$work/eval-$SLURM_ARRAY_TASK_ID.json"
