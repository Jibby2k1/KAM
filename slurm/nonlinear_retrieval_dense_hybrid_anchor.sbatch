#!/usr/bin/env bash
#SBATCH --job-name=kam_scope_anchor
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:03:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
method="${2:?dense or hybrid}"
mode="${3:?affinity or broad}"
case "$method" in dense|hybrid) ;; *) exit 2;; esac
case "$mode" in affinity|broad) ;; *) exit 2;; esac
work="$root/dense_hybrid_anchor_v1/$method/$mode"
index="$root/fullwiki_native_v1"
support="$index/support_link_v1"
export PYTHONPATH="$work:$root/coverage_gate_v1:$index/python" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
extra=()
if [ "$mode" = broad ]; then extra=(--anchor-reference "$work/affinity-replay.json.gz"); fi
python -u "$work/run_kam_conditioned_union.py" --inputs "$index" --assets "$root/assets" \
    --config "$support/hotpot_fullwiki_native_v1.json" --fixture "$work/fixture.json" \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" --output "$work/$mode-$SLURM_ARRAY_TASK_ID.json.gz" \
    --guard-split test --pools "$work/pools.json.gz" --fitted "$work/fitted.json" --baseline "$support/baseline.pt" \
    --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833 \
    --heads "$support/heads.pt" --ablations "$root/link_feature_ablation_v1/weights.pt" "${extra[@]}"
