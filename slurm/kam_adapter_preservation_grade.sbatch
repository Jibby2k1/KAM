#!/bin/bash
#SBATCH --job-name=kam-preserve-grade
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --array=0-18
set -euo pipefail
module load pytorch/2.8.0
root=$1
mode=()
if [[ ${KAM_FACTORIAL:-0} == 1 ]]; then mode+=(--factorial); fi
export PYTHONPATH="$root/runtime:$root/repo"
cd "$root"
read -r seed recipe arm < <(python kam_adapter_preservation.py "${mode[@]}" "$SLURM_ARRAY_TASK_ID")
ident="$seed-$recipe-$arm"
for task in humaneval mbpp; do
 python grade_kam_general_code.py --task "$task" --data "$root/data" --image "$root/python.sif" \
 --generation "$root/$task-$ident.json" --output "$root/$task-$ident-graded.json"
done
