#!/usr/bin/env bash
#SBATCH --job-name=kam_fullwiki_coarse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=22G
#SBATCH --time=00:14:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?build or replay required}"
work="$root/${KAM_FULLWIKI_COARSE_STAGE:-fullwiki_coarse_v1}"
index="$root/fullwiki_native_v1"
support="$index/support_link_v1"
export PYTHONPATH="$work:$root/conditioned_union_v1/score:$root/coverage_gate_v1:$index/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
common=(--inputs "$index" --config "$support/hotpot_fullwiki_native_v1.json")
case "$mode" in
  build) python -u "$work/run_kam_fullwiki_coarse.py" build "${common[@]}" --output "$work/partition-$SLURM_JOB_ID.pt" ;;
  replay|fused)
      grouping="${3:?build job ID required}"
      flags=()
      [[ "$mode" != fused ]] || flags=(--fused)
      python -u "$work/run_kam_fullwiki_coarse.py" replay "${common[@]}" --assets "$root/assets" \
          --fixture "$work/fixture.json.gz" --pools "$work/pools.json.gz" --reference "$work/replay.json.gz" \
          --baseline "$support/baseline.pt" \
          --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833 \
          --partition "$root/fullwiki_coarse_v1/partition-$grouping.pt" \
          --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" "${flags[@]}" \
          --output "$work/coarse-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
      ;;
  *) exit 2 ;;
esac
