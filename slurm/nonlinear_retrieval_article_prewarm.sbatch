#!/usr/bin/env bash
#SBATCH --job-name=kam_article_prewarm
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:04:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
export work="$root/article_prewarm_v1"
export database="$root/fullwiki_native_v1/lexical/articles.sqlite"
export jobdir="$work/jobs/$SLURM_JOB_ID"
mkdir -p "$jobdir"
(cd "$work" && sha256sum --quiet -c staged.sha256)
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c '
    set -euo pipefail
    python -u "$work/benchmark_kam_article_prewarm.py" --database "$database" --working-dir "$jobdir" \
        --pools "$work/pools.json.gz" --reference "$work/replay.json.gz" \
        --rank "$SLURM_PROCID" --output "$jobdir/prewarm-$SLURM_PROCID.json"
'
