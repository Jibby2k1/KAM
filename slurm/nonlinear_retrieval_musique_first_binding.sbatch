#!/usr/bin/env bash
#SBATCH --job-name=kam_musique_first_binding
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=21G
#SBATCH --time=00:12:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
fit="${2:?span-reader training job required}"
base="${3:?baseline retrieval job required}"
work="$root/${KAM_MUSIQUE_FIRST_STAGE:-musique_first_binding_v1}"
span="$root/musique_binding_spans_v1"
native="$root/native_musique_v1"
refs="$root/native_musique_bindings_v1"
export PYTHONPATH="$work:$span:$refs:$root/native_musique_decomposition_v1:$root/native_musique_recursive_v1:$native:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
shard="$((i % 4))"
seed_index="$((i / 4))"
python -u "$work/run_kam_musique_bindings.py" run --corpus "$native/corpus.jsonl.gz" --fixture "$native/fixture.json" \
  --assets "$root/assets" --templates "$refs/templates.json" --labels "$refs/labels.json" \
  --baseline "$native/search-$base-$shard.json.gz" --vectors "$native"/embedding-[0-3].pt \
  --shard "$shard" --shards 4 --span-checkpoint "$span/weights-$fit-$seed_index-trained.pt" \
  --early-branch --branch-labels "$root/native_musique_final_reader_v1/answers.json" \
  --output "$work/branch-$SLURM_ARRAY_JOB_ID-$i.json.gz"
