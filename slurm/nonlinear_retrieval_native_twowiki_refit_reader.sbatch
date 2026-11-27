#!/usr/bin/env bash
#SBATCH --job-name=kam_native_twowiki_refit_reader
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=17G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load pytorch/2.8.0
source /blue/uf-dsi/rvalle1/venvs/kam/bin/activate
root="${1:?runtime directory required}"
prior_id="${2:?previous reranking array job id required}"
work="$root/native_twowiki_refit_reader_v1"
initial="$root/native_twowiki_v1"
index="$root/fullwiki_native_v1"
export PYTHONPATH="$work:$root/native_twowiki_rerank_v1:$initial:$root/coverage_gate_v1:$index/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_native_twowiki_refit_reader.py" --assets "$root/assets" --inputs "$index" \
    --source "$root/cross_dataset_wikipedia_v1/twowiki.parquet" --fixture "$initial/fixture.json" \
    --prior "$root/native_twowiki_rerank_v1/rerank-$prior_id-$SLURM_ARRAY_TASK_ID.json.gz" \
    --picks "$work/picks.json" --policy "$work/policy.json" \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
    --output "$work/refit-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
