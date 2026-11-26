#!/usr/bin/env bash
#SBATCH --job-name=kam_native_sentence_refit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:06:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/native_sentence_refit_v1"
mode=()
prefix=features
if [ "${2:-}" = reader ]; then
    work="$root/native_sentence_refit_reader_rtx_v1"
    mode=(--fitted "$work/fitted.json" --old-policy "$work/policy.json")
    prefix=reader
elif [ "${2:-}" = reader-portability ]; then
    work="$root/native_sentence_refit_reader_v1"
    mode=(--fitted "$work/fitted.json" --old-policy "$work/policy.json" --test-only-reader-batches)
    prefix=reader
fi
index="$root/fullwiki_native_v1"
support="$index/support_link_v1"
export PYTHONPATH="$work:$root/coverage_gate_v1:$index/python" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_native_sentence_refit.py" --inputs "$index" --assets "$root/assets" \
  --config "$support/hotpot_fullwiki_native_v1.json" --fixture "$work/fixture.json" --baseline "$support/baseline.pt" \
  --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833 \
  --heads "$support/heads.pt" --ablations "$root/link_feature_ablation_v1/weights.pt" \
  --shard "$SLURM_ARRAY_TASK_ID" --shards "${KAM_NATIVE_SHARDS:-$SLURM_ARRAY_TASK_COUNT}" \
  --output "$work/$prefix-$SLURM_ARRAY_TASK_ID.json.gz" "${mode[@]}"
