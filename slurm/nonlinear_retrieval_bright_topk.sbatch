#!/usr/bin/env bash
#SBATCH --job-name=kam_bright_topk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:08:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/bright_topk_v1"
runtime="$root/bright_transfer_v1"
export PYTHONPATH="$work:$runtime:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$root/native_musique_v1:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
domains=(economics earth_science psychology sustainable_living)
i="$SLURM_ARRAY_TASK_ID"
domain="${domains[$i]}"
if (( i < 2 )); then
    data="$runtime/$domain"
    extra="$root/bright_passage_windows_v2/$domain"
    base=52996924
    window=53002056
    result=53002057
    shard="$i"
else
    data="$root/bright_windows_holdout_v1/$domain"
    extra="$data"
    base=53003099
    window=53003099
    result=53003100
    shard="$((i-2))"
fi
shift
python -u "$work/run_kam_bright_topk.py" --assets "$root/assets" --inputs "$data" \
    --head "$data"/key-"$base"-[0-3].pt --extra "$extra"/windows-"$window"-[0-3].pt \
    --lexical "$data/bm25.json.gz" --reference "$extra/rank-$result-$shard.json.gz" \
    --output "$work/rank-$SLURM_ARRAY_JOB_ID-$i.json.gz" "$@"
