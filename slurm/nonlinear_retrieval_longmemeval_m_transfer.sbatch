#!/usr/bin/env bash
#SBATCH --job-name=kam_longmem_transfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --time=00:20:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?generate or score required}"
work="$root/longmemeval_m_transfer_v1"
native="$root/longmemeval_m_v1"
export PYTHONPATH="$work:$native:$root/native_musique_decomposition_v1:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$root/native_musique_v1:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
runner=(python -u "$work/run_kam_longmemeval_m_transfer.py" "$mode" --assets "$root/assets" --fixture "$native/fixture.json")
case "$mode" in
  generate)
    fit="${3:?known decomposer fit job ID required}"
    i="$SLURM_ARRAY_TASK_ID"
    "${runner[@]}" --weights "$root/native_musique_decomposition_v1/decomposer-$fit-$i.pt" \
        --output "$work/plans-$SLURM_ARRAY_JOB_ID-$i.json.gz"
    ;;
  score)
    plan="${3:?plan generation job ID required}"
    encode="${4:?long memory embedding job ID required}"
    ref="${5:?reference retrieval job ID required}"
    i="$SLURM_ARRAY_TASK_ID"
    "${runner[@]}" --plans "$work"/plans-"$plan"-[0-2].json.gz --vectors "$native"/vector-"$encode"-[0-7].pt \
        --banks "$native"/sessions-[0-7].jsonl.gz --bank-index "$native/bank-index.json" \
        --reference "$native/eval-$ref-$i.json.gz" --shard "$i" --shards "$SLURM_ARRAY_TASK_COUNT" \
        --output "$work/transfer-$SLURM_ARRAY_JOB_ID-$i.json.gz"
    ;;
  *) exit 2 ;;
esac
