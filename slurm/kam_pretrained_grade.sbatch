#!/bin/bash
#SBATCH --job-name=kam-pretrained-code
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-5
set -euo pipefail
module load pytorch/2.8.0
root=$1
suffix=${2:-}
export PYTHONPATH="$root/runtime:$root/repo"
tasks=(humaneval mbpp)
arms=(kam dot pretrained)
task=${tasks[$((SLURM_ARRAY_TASK_ID / 3))]}
arm=${arms[$((SLURM_ARRAY_TASK_ID % 3))]}
python "$root/grade_kam_general_code.py" --task "$task" --data "$root/data" \
 --image "$root/python.sif" --generation "$root/$task-$arm.json" \
 --output "$root/$task-$arm-graded$suffix.json"
