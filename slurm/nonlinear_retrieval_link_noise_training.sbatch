#!/usr/bin/env bash
#SBATCH --job-name=kam_link_noise_fit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:14:00
set -euo pipefail
module purge
module load pytorch/2.8.0
if [ -n "${KAM_ENV_SETUP:-}" ]; then
    source "$KAM_ENV_SETUP"
fi
root="${1:?runtime directory required}"
mode="${2:?mode or dataset required}"
work="$root/link_noise_training_v1"
support="$root/fullwiki_native_v1/support_link_v1"
export PYTHONPATH="$work:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
base=(--assets "$root/assets" --config "$support/hotpot_fullwiki_native_v1.json" \
      --baseline "$support/baseline.pt" --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833)
if [ "$mode" = fit ]; then
    python -u "$work/run_kam_link_noise_training.py" "${base[@]}" --inputs "$root/fullwiki_native_v1" \
      --fixture "$support/fixture.json" --fit-records "$support/fit-0.json" \
      --output "$work/fit.json" --weights "$work/weights.pt"
else
    python -u "$work/run_kam_link_noise.py" "${base[@]}" --heads "$support/heads.pt" \
      --fixture "$work/$mode-independent-noise-fixture.json" --robust-heads "$work/weights.pt" \
      --injection-template 'For further reading, consult {title}.' \
      --output "$work/$mode-$SLURM_ARRAY_TASK_ID.json" \
      --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT"
fi
