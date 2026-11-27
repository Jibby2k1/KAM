#!/usr/bin/env bash
#SBATCH --job-name=kam_final_row_attention
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=18G
#SBATCH --time=00:18:00
set -euo pipefail
module purge
module load pytorch/2.8.0
root="${1:?runtime directory required}"
kind="${2:?fixed or relative checkpoint required}"
work="$root/${KAM_FINAL_ROW_STAGE:-final_row_attention_v1}"
export PYTHONPATH="$work:$root/packed_clock_inference_v1:$root/temporal_clock_outliers_v1:$root/temporal_correction_curriculum_v1:$root/curriculum_geometry_v1:$root/radial_archive_training_v1:$root/radial_training_graph_diagnostic_v1"
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
(cd "$work" && sha256sum --quiet -c staged.sha256)
seeds=(791513 791543 791553)
i="$SLURM_ARRAY_TASK_ID"
case "$kind" in
    fixed) weights=(--checkpoint "$root/bitemporal_graph_v1/decoder.pt"
                    --checkpoint-sha256 3e1196f8f88c42e812a05580a52438f167135ff9129e5086af59c14d6ffd777f) ;;
    relative) weights=(--fit-reference "$root/temporal_relative_log_v2/clocks-52997769-$i.json") ;;
    *) exit 2 ;;
esac
python -u "$work/run_kam_final_row_attention.py" --seed "${seeds[$i]}" "${weights[@]}" \
    --dallas "$root/oecd_revisions_l4_v1/oecd-observed.json" --eurostat "$root/eurostat_vintages_v1/fixture.json" \
    --output "$work/attention-$kind-$SLURM_ARRAY_JOB_ID-$i.json"
