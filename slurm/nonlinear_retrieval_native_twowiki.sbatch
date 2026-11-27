#!/usr/bin/env bash
#SBATCH --job-name=kam_native_twowiki
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=28G
#SBATCH --time=00:08:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
mode="${2:?prepare or search required}"
purpose="${3:-test}"
initial="$root/native_twowiki_v1"
if [[ "$purpose" == fit ]]; then
    work="$root/native_twowiki_fit_v1"
    holdout=(--holdout "$initial/fixture.json")
else
    work="$initial"
    holdout=()
fi
source_data="$root/cross_dataset_wikipedia_v1"
index="$root/fullwiki_native_v1"
export PYTHONPATH="$work:$initial:$root/coverage_gate_v1:$index/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
(cd "$work" && sha256sum --quiet -c staged.sha256)
case "$mode" in
    prepare)
        source /blue/uf-dsi/rvalle1/venvs/kam/bin/activate
        python -u "$work/prepare_kam_native_twowiki.py" --source "$source_data/twowiki.parquet" \
            --coverage "$source_data/coverage-52967153.json.gz" --reserve "$initial/prior-breadth.json" \
            --reserve "$initial/prior-noise.json" --reserve "$initial/prior-verifier.json" --output "$work/fixture.json" \
            "${holdout[@]}"
        ;;
    search)
        python -u "$work/run_kam_native_twowiki.py" --inputs "$index" --assets "$root/assets" \
            --config "$index/support_link_v1/hotpot_fullwiki_native_v1.json" --fixture "$work/fixture.json" \
            --shard "$SLURM_ARRAY_TASK_ID" --shards "$SLURM_ARRAY_TASK_COUNT" \
            --output "$work/search-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID.json.gz"
        ;;
    *) exit 2 ;;
esac
