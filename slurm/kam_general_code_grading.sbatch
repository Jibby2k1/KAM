#!/bin/bash
#SBATCH --job-name=kam-general-code
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-3
set -euo pipefail
module load pytorch/2.8.0
root=$1
export PYTHONPATH="$root/runtime:$root/repo"
tasks=(humaneval mbpp)
arms=(T-KAM-F T-WIDE)
task=${tasks[$((SLURM_ARRAY_TASK_ID / 2))]}
arm=${arms[$((SLURM_ARRAY_TASK_ID % 2))]}
python "$root/grade_kam_general_code.py" --task "$task" --data "$root/data" \
  --image "$root/python.sif" --generation "$root/$task-$arm.json" \
  --output "$root/$task-$arm-graded.json"
