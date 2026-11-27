#!/usr/bin/env bash
#SBATCH --job-name=kam_cross_dataset_wiki
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:08:00
set -euo pipefail
module purge
module load pytorch/2.8.0
source /blue/uf-dsi/rvalle1/venvs/kam/bin/activate
root="${1:?runtime directory required}"
work="$root/cross_dataset_wikipedia_v1"
export PYTHONPATH="$work" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/audit_kam_cross_dataset_wikipedia.py" --inputs "$root/fullwiki_native_v1" \
    --twowiki "$work/twowiki.parquet" --musique "$work/musique.jsonl" \
    --output "$work/coverage-$SLURM_JOB_ID.json.gz"
