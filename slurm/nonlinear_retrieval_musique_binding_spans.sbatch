#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_binding_spans
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:22:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/musique_binding_spans_v1"
native="$root/native_musique_v1"
export PYTHONPATH="$work:$root/native_musique_bindings_v1:$root/native_musique_decomposition_v1:$root/native_musique_recursive_v1:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
seeds=(829414 829427 829483)
arms=(frozen trained)
if (( i % 2 )); then arms=(trained frozen); fi
for arm in "${arms[@]}"; do
    name="$SLURM_ARRAY_JOB_ID-$i-$arm"
    python -u "$work/train_kam_musique_binding_spans.py" --assets "$root/assets" --data "$work/data.json.gz" \
       --weights "$work/weights-$name.pt" --output "$work/fit-$name.json" \
       --seed "${seeds[$i]}" --arm "$arm"
done
