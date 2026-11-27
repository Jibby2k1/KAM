#!/usr/bin/env bash
#SBATCH --job-name=kam_longmem_reader
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=15G
#SBATCH --time=00:05:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
ref="${2:?reference retrieval job ID required}"
work="$root/longmemeval_m_reader_v1"
native="$root/longmemeval_m_v1"
export PYTHONPATH="$work:$native:$root/native_musique_bindings_v1:$root/native_musique_recursive_v1:$root/native_musique_v1:$root/native_twowiki_v1:$root/coverage_gate_v1:$root/fullwiki_native_v1/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/run_kam_longmemeval_m_reader.py" read --assets "$root/assets" --fixture "$native/fixture.json" \
    --bank-index "$native/bank-index.json" --banks "$native"/sessions-[0-7].jsonl.gz \
    --labels "$work/labels.json" --reference "$native/eval-$ref-$i.json.gz" --shard "$i" --shards "$SLURM_ARRAY_TASK_COUNT" \
    --output "$work/read-$SLURM_ARRAY_JOB_ID-$i.json.gz"
