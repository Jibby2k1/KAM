#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_composition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=21G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/musique_reader_composition_v1"
native="$root/native_musique_v1"
bind="$root/native_musique_bindings_v1"
final="$root/native_musique_final_reader_v1"
export PYTHONPATH="$work:$final:$bind:$root/musique_binding_branch_v1:$root/native_musique_recursive_v1:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_musique_reader_composition.py" --corpus "$native/corpus.jsonl.gz" --fixture "$native/fixture.json" \
  --assets "$root/assets" --labels "$final/answers.json" --templates "$bind/templates.json" --intermediate "$bind/labels.json" \
  --gold-reference "$final/read-52978195-$i.json.gz" --primary-reference "$root/musique_binding_branch_v1/branch-52998368-$i.json.gz" \
  --shard "$i" --shards 4 --output "$work/composition-$SLURM_ARRAY_JOB_ID-$i.json.gz"
