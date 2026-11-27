#!/bin/bash
#SBATCH --job-name=kam-long-eval
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:35:00
#SBATCH --array=0-9
set -euo pipefail
module load pytorch/2.8.0
root=$1
export PYTHONPATH="$root/runtime:$root/repo"
export OMP_NUM_THREADS=4
tasks=(humaneval mbpp gpqa mmlu gsm8k)
arms=(T-KAM-F T-WIDE)
task=${tasks[$((SLURM_ARRAY_TASK_ID / 2))]}
arm=${arms[$((SLURM_ARRAY_TASK_ID % 2))]}
extra=()
if [[ "$arm" == T-KAM-F ]]; then extra+=(--omit-route-diagnostics); fi
python "$root/run_kam_general_benchmarks.py" --task "$task" --arm "$arm" \
  --pair wiki_code_arithmetic_2048:731109 --data "$root/data" --checkpoints "$root" \
  --batch 24 --output "$root/$task-$arm.json" "${extra[@]}"
