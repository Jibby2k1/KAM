#!/usr/bin/env bash
#SBATCH --job-name=kam_article_staging
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:04:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/article_staging_v1"
database="$root/fullwiki_native_v1/lexical/articles.sqlite"
scratch="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
df -T "$scratch" "$database"
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/benchmark_kam_article_staging.py" --database "$database" --scratch "$scratch" \
    --pools "$work/pools.json.gz" --reference "$work/replay.json.gz" \
    --offset "$SLURM_ARRAY_TASK_ID" --output "$work/staging-$SLURM_ARRAY_TASK_ID.json"
