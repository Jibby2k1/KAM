#!/usr/bin/env bash
#SBATCH --job-name=kam_eurostat_scope
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=17G
#SBATCH --time=00:06:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
index="${2:?source document bank array required}"
work="$root/eurostat_scope_v1"
base="$root/eurostat_untyped_v1"
export PYTHONPATH="$work:$base:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/benchmark_kam_eurostat_scope.py" --fixture "$base/fixture.json.gz" --assets "$root/assets" \
    --embeddings "$base"/embed-"$index"-[0-3].pt --output "$work/scope-$SLURM_JOB_ID.json.gz"
