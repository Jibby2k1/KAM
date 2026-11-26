#!/usr/bin/env bash
#SBATCH --job-name=kam_sentence_verifier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
if [ -n "${KAM_ENV_SETUP:-}" ]; then
    source "$KAM_ENV_SETUP"
fi
root="${1:?runtime directory required}"
dataset="${2:?dataset required}"
work="$root/sentence_verifier_v1"
probe=()
if [ "${3:-}" = "contextual" ]; then
    work="$root/sentence_verifier_context_v1"
    probe+=(--contextual-test)
elif [ "${3:-}" = "profile" ]; then
    work="$root/sentence_verifier_profile_v1"
    probe+=(--profile)
elif [ "${3:-}" = "reader" ]; then
    work="$root/sentence_verifier_reader_v1"
    probe+=(--reader-policy "$work/policy.json")
fi
support="$root/fullwiki_native_v1/support_link_v1"
gate="$root/coverage_gate_v1"
export PYTHONPATH="$gate:$root/fullwiki_native_v1/python" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_sentence_verifier.py" --assets "$root/assets" \
  --config "$support/hotpot_fullwiki_native_v1.json" --fixture "$gate/gate-fixture.json" --splits "$gate/gate-splits.json" \
  --baseline "$support/baseline.pt" --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833 \
  --heads "$support/heads.pt" --ablations "$root/link_feature_ablation_v1/weights.pt" \
  --dataset "$dataset" --output "$work/$dataset-$SLURM_ARRAY_TASK_ID.json" \
  --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" "${probe[@]}"
