#!/bin/bash
#SBATCH --job-name=kam-teacher-hidden
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:20:00
set -euo pipefail
module load pytorch/2.8.0
root=$1
shift
export PYTHONPATH="$root/teacher-cache-code:$root/runtime:$root/repo:$root"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python "$root/teacher-cache-code/probe_kam_teacher_hidden_cache.py" "$root" "$@"
