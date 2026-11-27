#!/usr/bin/env bash
#SBATCH --job-name=kam_fullwiki_quantized_search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=25G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/fullwiki_quantized_search_v1"
index="$root/fullwiki_native_v1"
export PYTHONPATH="$work:$root/fullwiki_coarse_v1:$root/conditioned_union_v1/score:$root/coverage_gate_v1:$index/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_fullwiki_quantized_search.py" --inputs "$index" --assets "$root/assets" \
    --config "$index/support_link_v1/hotpot_fullwiki_native_v1.json" --fixture "$root/conditioned_union_v1/score/fixture.json" \
    --shard "$i" --shards "$SLURM_ARRAY_TASK_COUNT" --output "$work/rank-$SLURM_ARRAY_JOB_ID-$i.json.gz"
