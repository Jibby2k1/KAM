#!/bin/bash
#SBATCH --job-name=kam-preserve-fit
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:08:00
#SBATCH --array=0-17
set -euo pipefail
module load pytorch/2.8.0
root=$1
mode=()
if [[ ${KAM_FACTORIAL:-0} == 1 ]]; then mode+=(--factorial); fi
export PYTHONPATH="$root/runtime:$root/repo"
export OMP_NUM_THREADS=4
cd "$root"
read -r seed recipe arm < <(python kam_adapter_preservation.py "${mode[@]}" --train "$SLURM_ARRAY_TASK_ID")
python train_kam_adapter_preservation.py "$root" --seed "$seed" --recipe "$recipe" --arm "$arm"
