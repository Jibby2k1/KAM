#!/bin/bash
#SBATCH --job-name=kam-musique-cache
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:25:00
set -euo pipefail
module load pytorch/2.8.0
root=${1:?checkpoint root required}
work=${2:?experiment directory required}
assets=${3:?offline encoder assets required}
operation=${4:?operation required}
export PYTHONPATH="$work:$root/runtime:$root/repo:$root"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd "$work"
if [[ "$operation" == evaluate ]]; then
    python -u kam_musique_cached_retrieval.py "$operation" "$root" "$work" "$assets" "$SLURM_ARRAY_TASK_ID"
else
    python -u kam_musique_cached_retrieval.py "$operation" "$root" "$work" "$assets"
fi
