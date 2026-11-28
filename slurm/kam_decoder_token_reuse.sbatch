#!/bin/bash
#SBATCH --job-name=kam-token-reuse
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:04:00
#SBATCH --array=0-1%2
set -euo pipefail
module load pytorch/2.8.0
root=$1
work=$2
arms=(kam dot)
export PYTHONPATH="$work:$root/runtime:$root/repo:$root"
export OMP_NUM_THREADS=4
case "${3:-transport}" in
  transport)
    python "$work/probe_kam_decoder_token_reuse.py" "$root" "$work" "${arms[SLURM_ARRAY_TASK_ID]}" "$work/host_upload_reference.py"
    ;;
  grouped-eos)
    python "$work/probe_kam_decoder_eos_poll.py" "$root" "$work" "${arms[SLURM_ARRAY_TASK_ID]}"
    ;;
  *) exit 2 ;;
esac
