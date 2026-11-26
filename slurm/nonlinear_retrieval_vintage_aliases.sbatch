#!/usr/bin/env bash
#SBATCH --job-name=kam_vintage_aliases
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:08:00
set -euo pipefail
module purge
module load pytorch/2.8.0
if [ -n "${KAM_ENV_SETUP:-}" ]; then
  source "$KAM_ENV_SETUP"
fi
root="${1:?runtime directory required}"
work="$root/vintage_aliases_v1"
export PYTHONPATH="$work"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
python -u "$work/run_kam_vintage_aliases.py" --assets "$root/assets" \
  --vintage-fixture "$root/vintage_revisions_v1/fed-vintage-fixture.json" --output "$work/results.json"
