#!/usr/bin/env bash
#SBATCH --job-name=kam_eurostat_text
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=17G
#SBATCH --time=00:10:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?encode or score required}"
work="$root/eurostat_untyped_v1"
export PYTHONPATH="$work:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
cmd=(python -u "$work/run_kam_eurostat_text_routing.py" "$mode" --fixture "$work/fixture.json.gz" --assets "$root/assets")
case "$mode" in
    encode)
        i="$SLURM_ARRAY_TASK_ID"
        "${cmd[@]}" --shard "$i" --output "$work/embed-$SLURM_ARRAY_JOB_ID-$i.pt"
        ;;
    score)
        "${cmd[@]}" --parts "$work"/embed-"${3:?embedding array required}"-[0-3].pt --output "$work/ranked-$SLURM_JOB_ID.json.gz"
        ;;
    *) exit 2 ;;
esac
