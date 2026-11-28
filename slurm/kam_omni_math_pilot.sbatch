#!/bin/bash
#SBATCH --job-name=kam-omni-math-pilot
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:04:00
set -euo pipefail
module load pytorch/2.8.0
root=$1
work=$2
export PYTHONPATH="$work:$root/runtime:$root/repo:$root"
export OMP_NUM_THREADS=4
python "$work/kam_omni_math_untouched_pilot.py" generate "$root" "$work"
