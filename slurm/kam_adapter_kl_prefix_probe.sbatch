#!/bin/bash
#SBATCH --job-name=kam-kl-prefix
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:05:00
#SBATCH --array=0-1
set -euo pipefail
module load pytorch/2.8.0
root=$1
export PYTHONPATH="$root/runtime:$root/repo"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
arms=(kam dot)
arm=${arms[$SLURM_ARRAY_TASK_ID]}
kind=${2:-loss-backward}
gpu_suffix=${3:-}
case "$kind" in
  loss-backward)
    python "$root/probe_kam_kl_prefix.py" "$root" "$root/kl-prefix${gpu_suffix}-$arm.json" --arm "$arm"
    ;;
  optimizer)
    python "$root/probe_kam_kl_prefix.py" "$root" "$root/kl-prefix-optimizer${gpu_suffix}-$arm.json" --arm "$arm" --optimizer-replay
    ;;
  *) exit 2 ;;
esac
