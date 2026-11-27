#!/usr/bin/env bash
#SBATCH --job-name=kam_native_twowiki_scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load pytorch/2.8.0
source /blue/uf-dsi/rvalle1/venvs/kam/bin/activate
root="${1:?runtime directory required}"
model="${2:?small, base, or base_bf16 required}"
pairs_id="${3:?refit reader array job id required}"
fixed_id="${4:?fixed gold reader array job id required}"
work="$root/${KAM_TWOWIKI_READER_SCALE_DIR:-native_twowiki_reader_scale_v1}"
initial="$root/native_twowiki_v1"
index="$root/fullwiki_native_v1"
export PYTHONPATH="$work:$root/native_twowiki_rerank_v1:$initial:$root/coverage_gate_v1:$index/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_native_twowiki_reader_scale.py" --assets "$root/assets" --inputs "$index" \
    --source "$root/cross_dataset_wikipedia_v1/twowiki.parquet" --fixture "$initial/fixture.json" \
    --pairs "$root/native_twowiki_refit_reader_v1/refit-$pairs_id-$SLURM_ARRAY_TASK_ID.json.gz" \
    --gold "$root/native_twowiki_reader_v1/read-$fixed_id-$SLURM_ARRAY_TASK_ID.json.gz" --model "$model" \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
    --output "$work/scale-$model-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
