#!/bin/bash
#SBATCH --job-name=kam-pretrained-eval
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=40G
#SBATCH --time=00:35:00
#SBATCH --array=0-14
set -euo pipefail
module load pytorch/2.8.0
root=$1
export PYTHONPATH="$root/runtime:$root/repo"
export OMP_NUM_THREADS=4
tasks=(humaneval mbpp gpqa mmlu gsm8k)
arms=(kam dot pretrained)
task=${tasks[$((SLURM_ARRAY_TASK_ID / 3))]}
arm=${arms[$((SLURM_ARRAY_TASK_ID % 3))]}
python "$root/eval_kam_pretrained.py" "$root" --task "$task" --arm "$arm"
