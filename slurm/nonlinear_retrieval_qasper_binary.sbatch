#!/usr/bin/env bash
#SBATCH --job-name=kam_qasper_binary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=17G
#SBATCH --time=00:04:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/qasper_binary_v1"
export PYTHONPATH="$work:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$root/native_musique_v1:$root/native_twowiki_v1:$root/fullwiki_native_v1/python:$root/coverage_gate_v1"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_qasper_binary.py" --fixture "$work/fixture.json.gz" --assets "$root/assets" \
    --shard "$i" --shards "$SLURM_ARRAY_TASK_COUNT" --output "$work/binary-$SLURM_ARRAY_JOB_ID-$i.json.gz"
