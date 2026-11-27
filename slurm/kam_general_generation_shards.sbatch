#!/bin/bash
#SBATCH --job-name=kam-eval-gen
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --array=0-15
set -euo pipefail
module load pytorch/2.8.0
root=$1
archive=$2
export PYTHONPATH="$root/runtime:$root/repo"
export OMP_NUM_THREADS=4
index=$SLURM_ARRAY_TASK_ID
if ((index < 3)); then
  task=humaneval; shard=$index; shards=3
elif ((index < 7)); then
  task=mbpp; shard=$((index - 3)); shards=4
else
  task=gsm8k; shard=$((index - 7)); shards=9
fi
python "$root/run_kam_general_benchmarks.py" --task "$task" --arm T-KAM-F \
  --data "$root/data" --checkpoints "$archive" --shard "$shard" --shards "$shards" \
  --output "$root/$task-T-KAM-F-shard$shard.json"
