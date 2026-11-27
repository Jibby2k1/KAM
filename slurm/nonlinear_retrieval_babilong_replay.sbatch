#!/usr/bin/env bash
#SBATCH --job-name=kam_babilong_replay
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=13G
#SBATCH --time=00:35:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/babilong_replay_v1"
export PYTHONPATH="$work:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
tasks=(qa2 qa3 qa7 qa8)
lengths=(16k 64k)
task="${tasks[$((i % 4))]}"
length="${lengths[$((i / 4))]}"
python -u "$work/run_kam_babilong_retrieval.py" --fixture "$work/fixture.json.gz" --assets "$root/assets" \
    --task "$task" --length "$length" --output "$work/retrieval-$SLURM_ARRAY_JOB_ID-$i.json.gz"
