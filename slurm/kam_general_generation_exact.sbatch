#!/bin/bash
#SBATCH --job-name=kam-eval-exact
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --array=0-2
set -euo pipefail
module load pytorch/2.8.0
root=$1
archive=$2
export PYTHONPATH="$root/runtime:$root/repo"
export OMP_NUM_THREADS=4
tasks=(humaneval mbpp gsm8k)
task=${tasks[$SLURM_ARRAY_TASK_ID]}
python "$root/run_kam_general_benchmarks.py" --task "$task" --arm T-KAM-F \
  --data "$root/data" --checkpoints "$archive" --omit-route-diagnostics \
  --output "$root/$task-T-KAM-F-fast.json"
