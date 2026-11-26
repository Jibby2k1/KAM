#!/usr/bin/env bash
#SBATCH --job-name=kam_support_links
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
if [ -n "${KAM_ENV_SETUP:-}" ]; then
    source "$KAM_ENV_SETUP"
fi
root="${1:?runtime directory required}"
mode="${2:?fit or eval required}"
core="$root/fullwiki_native_v1"
work="$core/support_link_v1"
shard="${SLURM_ARRAY_TASK_ID:-0}"
shards="${SLURM_ARRAY_TASK_COUNT:-1}"
export PYTHONPATH="$work:$core/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
(
    cd "$work"
    sha256sum --quiet -c staged.sha256
)
if [ -n "${KAM_BASELINE_SHA256:-}" ]; then
    baseline=(--weights "${KAM_BASELINE_FILE:-$work/baseline.pt}" --baseline-sha256 "$KAM_BASELINE_SHA256")
else
    baseline=(--weights "$root/musique_transfer_v1/shared_weights.pt"
              --training-cache "$root/router_v1/features.pt" --title-cache "$root/title_cache_v1/features.pt")
fi
python -u "$work/run_kam_fullwiki_support_links.py" "$mode" --inputs "$core" --assets "$root/assets" \
 --fixture "$work/fixture.json" --config "$work/hotpot_fullwiki_native_v1.json" \
 "${baseline[@]}" --checkpoint "$work/heads.pt" \
 --output "$work/$mode-$shard.json" --shard "$shard" --shards "$shards"
