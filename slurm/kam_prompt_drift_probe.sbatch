#!/bin/bash
#SBATCH --job-name=kam-prompt-kl
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --array=0-1
set -euo pipefail
module load pytorch/2.8.0
root=$1
export PYTHONPATH="$root/runtime:$root/repo"
export OMP_NUM_THREADS=4
tasks=(gpqa mmlu)
task=${tasks[$SLURM_ARRAY_TASK_ID]}
python "$root/probe_kam_prompt_drift.py" "$root" "$task" "$root/prompt-drift-$task.json"
