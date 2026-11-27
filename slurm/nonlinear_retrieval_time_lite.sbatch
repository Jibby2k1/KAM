#!/usr/bin/env bash
#SBATCH --job-name=kam_time_lite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=21G
#SBATCH --time=00:16:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/time_lite_v1"
export PYTHONPATH="$work:$root/longmemeval_m_reader_v1:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$root/native_musique_v1:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
domains=(TIME-Lite-Dial TIME-Lite-News TIME-Lite-Wiki)
sizes=(base large)
i="$SLURM_ARRAY_TASK_ID"
domain="${domains[$((i % 3))]}"
size="${sizes[$((i / 3))]}"
python -u "$work/run_kam_time_lite.py" --assets "$root/assets" --fixture "$work/$domain.json.gz" \
    --domain "$domain" --size "$size" --output "$work/choices-$SLURM_ARRAY_JOB_ID-$i.json.gz"
