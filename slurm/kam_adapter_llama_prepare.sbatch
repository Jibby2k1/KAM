#!/bin/bash
#SBATCH --job-name=kam-llama-data
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:10:00
set -euo pipefail
module load pytorch/2.8.0
root=$1
export OMP_NUM_THREADS=4
cd "$root/model"
sha256sum config.json tokenizer.json model.safetensors.index.json model-*.safetensors > "$root/model-sha256.txt"
python "$root/prepare_kam_pretrained.py" "$root"
