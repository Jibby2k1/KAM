#!/usr/bin/env bash
#SBATCH --job-name=kam_longmemeval_m
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=18G
#SBATCH --time=00:24:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?embed or evaluate required}"
work="$root/longmemeval_m_v1"
export PYTHONPATH="$work:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
run=(python -u "$work/run_kam_longmemeval_m.py" "$mode" --assets "$root/assets" --fixture "$work/fixture.json"
     --bank-index "$work/bank-index.json" --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT")
case "$mode" in
  embed)
    "${run[@]}" --bank "$work/sessions-$SLURM_ARRAY_TASK_ID.jsonl.gz" \
        --output "$work/vector-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.pt"
    ;;
  evaluate)
    bank="${3:?embedding job ID required}"
    "${run[@]}" --vectors "$work"/vector-"$bank"-[0-7].pt \
        --output "$work/eval-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
    ;;
  *) exit 2 ;;
esac
