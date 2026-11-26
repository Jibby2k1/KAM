#!/usr/bin/env bash
#SBATCH --job-name=kam_relation_profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:02:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/union_relation_features_v1"
index="$root/fullwiki_native_v1"
export PYTHONPATH="$work:$root/coverage_gate_v1:$index/python" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python "$work/benchmark_kam_union_relation_features.py" --fixture "$work/fixture.json.gz" \
    --pools "$work/pools.json.gz" --replay "$work/replay.json.gz" --database "$index/lexical/articles.sqlite" \
    --stride "$SLURM_ARRAY_TASK_COUNT" --offset "$SLURM_ARRAY_TASK_ID" --output "$work/profile-$SLURM_ARRAY_TASK_ID.json"
