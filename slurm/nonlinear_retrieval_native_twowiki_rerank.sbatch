#!/usr/bin/env bash
#SBATCH --job-name=kam_native_twowiki_rerank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
source /blue/uf-dsi/rvalle1/venvs/kam/bin/activate
root="${1:?runtime directory required}"
work="$root/native_twowiki_rerank_v1"
initial="$root/native_twowiki_v1"
index="$root/fullwiki_native_v1"
support="$index/support_link_v1"
export PYTHONPATH="$work:$initial:$root/coverage_gate_v1:$index/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_native_twowiki_rerank.py" --inputs "$index" --assets "$root/assets" \
    --source "$root/cross_dataset_wikipedia_v1/twowiki.parquet" --fixture "$initial/fixture.json" \
    --config "$support/hotpot_fullwiki_native_v1.json" --baseline "$support/baseline.pt" \
    --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833 \
    --heads "$support/heads.pt" --ablations "$root/link_feature_ablation_v1/weights.pt" \
    --fitted "$root/live_anchor_gc_readiness_v1/fitted.json" \
    --fitted-sha256 0abfd208cecf86133ffeaf581996b7a6bb6686ed28d9f0ac73eaef8966d3636a \
    --search "$initial"/search-52968757-[0-2].json.gz "$initial/search-52967945-3.json.gz" \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
    --output "$work/rerank-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
