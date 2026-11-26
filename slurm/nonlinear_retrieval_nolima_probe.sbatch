#!/usr/bin/env bash
#SBATCH --job-name=kam_nolima_probe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=18G
#SBATCH --time=00:14:00
set -euo pipefail
module purge
module load pytorch/2.8.0
if [ -n "${KAM_ENV_SETUP:-}" ]; then
    source "$KAM_ENV_SETUP"
fi
root="${1:?runtime directory required}"
work="$root/nolima_probe_v1"
support="$root/fullwiki_native_v1/support_link_v1"
export PYTHONPATH="$work"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=6 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_nolima_probe.py" --assets "$root/assets" \
 --baseline "$support/baseline.pt" --baseline-sha256 95a0c2d6f53ffa7369efb251328540b66c5ba24f46c491b80238b354b4848833 \
 --config "$support/hotpot_fullwiki_native_v1.json" --reader "$root/assets/reader_base_7bcac572ce56db69c1ea7c8af255c5d7c9672fc2" \
 --minimal "$work/nolima-needle_set.json" --direct "$work/nolima-needle_set_ONLYDirect.json" \
 --distractors "$work/nolima-needle_set_w_Distractor.json" \
 --books "$work/nolima-rand_book_1.txt" "$work/nolima-rand_book_2.txt" --output "$work/eval-$SLURM_ARRAY_TASK_ID.json" \
 --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT"
