#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_terminal_evidence
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=23G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/musique_terminal_evidence_v2"
native="$root/native_musique_v1"
final="$root/native_musique_final_reader_v1"
bind="$root/native_musique_bindings_v1"
windows="$root/musique_reader_window_v3"
export PYTHONPATH="$work:$windows:$root/musique_reader_composition_v1:$final:$bind:$root/native_musique_recursive_v1:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_musique_terminal_evidence.py" --corpus "$native/corpus.jsonl.gz" --fixture "$native/fixture.json" \
  --assets "$root/assets" --labels "$final/answers.json" --intermediate "$bind/labels.json" \
  --window-reference "$windows/window-53012831-$i.json.gz" --shard "$i" --shards 4 \
  --output "$work/terminal-evidence-$SLURM_ARRAY_JOB_ID-$i.json.gz"
