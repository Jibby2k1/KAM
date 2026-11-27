#!/usr/bin/env bash
#SBATCH --job-name=kam_integer_clock_rank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=19G
#SBATCH --time=00:09:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
work="$root/integer_clock_rank_v1"
sources="$root/clock_release_mixture_v1"
weights="$root/typed_guarded_sdpa_v1"
export PYTHONPATH="$work:$weights:$root/temporal_correction_curriculum_v1:$root/curriculum_geometry_v1:$root/radial_archive_training_v1:$root/radial_training_graph_diagnostic_v1"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(791513 791543 791553)
i="$SLURM_ARRAY_TASK_ID"
python -u "$work/benchmark_kam_integer_clock_rank.py" --seed "${seeds[$i]}" --weights "$weights/weights-$i.pt" \
  --unemployment "$root/eurostat_vintages_v1/fixture.json" --dallas "$root/oecd_revisions_l4_v1/oecd-observed.json" \
  --industry "$sources/industry-fixture.json" --later-unemployment "$sources/later-unemployment.json" \
  --later-industry "$sources/later-industry.json" --output "$work/clock-$SLURM_ARRAY_JOB_ID-$i.json"
