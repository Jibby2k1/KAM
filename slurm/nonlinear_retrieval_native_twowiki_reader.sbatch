#!/usr/bin/env bash
#SBATCH --job-name=kam_native_twowiki_reader
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
source /blue/uf-dsi/rvalle1/venvs/kam/bin/activate
root="${1:?runtime directory required}"
work="$root/native_twowiki_reader_v1"
index="$root/fullwiki_native_v1"
export PYTHONPATH="$work:$root/coverage_gate_v1:$index/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_native_twowiki_reader.py" --inputs "$index" --assets "$root/assets" \
    --fixture "$root/native_twowiki_v1/fixture.json" --source "$root/cross_dataset_wikipedia_v1/twowiki.parquet" \
    --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
    --output "$work/read-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
