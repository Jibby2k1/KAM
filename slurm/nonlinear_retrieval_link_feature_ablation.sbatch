#!/usr/bin/env bash
#SBATCH --job-name=kam_link_ablation
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
work="$root/link_feature_ablation_v1"
support="$root/fullwiki_native_v1/support_link_v1"
breadth="$root/dataset_breadth_v1"
export PYTHONPATH="$work:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
base=(--assets "$root/assets" --config "$support/hotpot_fullwiki_native_v1.json" \
      --baseline "$support/baseline.pt" --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833)
if [ "$mode" = fit ]; then
    python -u "$work/run_kam_link_feature_ablation.py" "${base[@]}" --inputs "$root/fullwiki_native_v1" \
      --fixture "$support/fixture.json" --fit-audit "$support/fit-0.json" \
      --reference-heads "$support/heads.pt" --output "$work/weights.pt" --audit "$work/fit.json"
else
    fixture="$breadth/$mode-fixture.json"
    opts=()
    if [ "$mode" = hotpot ]; then
        fixture="$work/hotpot-fixture.json"
        opts+=(--inputs "$root/fullwiki_native_v1")
    elif [ "$mode" = news ]; then
        opts+=(--corpus "$breadth/news-corpus.json")
    fi
    python -u "$work/run_kam_dataset_breadth.py" "${base[@]}" --heads "$support/heads.pt" \
      --fixture "$fixture" --ablations "$work/weights.pt" \
      --output "$work/$mode-$SLURM_ARRAY_TASK_ID.json" \
      --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" "${opts[@]}"
fi
