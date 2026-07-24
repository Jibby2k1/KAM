#!/usr/bin/env bash
set -euo pipefail
MODE="${1:---plan-only}"
HPG_ACCOUNT="${HPG_ACCOUNT:-uf-dsi}"
HPG_QOS="${HPG_QOS:-uf-dsi}"
HPG_PARTITION="${HPG_PARTITION:-hpg-turin}"
HPG_BLUE_ROOT="${HPG_BLUE_ROOT:-/blue/uf-dsi/rvalle1}"
HPG_REPO="${HPG_REPO:-${HPG_BLUE_ROOT}/KAM}"
HPG_ENV="${HPG_ENV:-${HPG_BLUE_ROOT}/venvs/kam}"
MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-4}"
PHASE4_CONFIG="${PHASE4_CONFIG:-${HPG_REPO}/configs/phase4/factorial_screen.yaml}"
PHASE4_MANIFEST="${PHASE4_MANIFEST:-${HPG_REPO}/results/phase4/manifests/factorial_screen.jsonl}"
PHASE4_RUN_ROOT="${PHASE4_RUN_ROOT:-${HPG_REPO}/results/phase4/factorial_screen}"
LOG_ROOT="${LOG_ROOT:-${HPG_BLUE_ROOT}/logs/phase4}"
for value in "$HPG_ACCOUNT" "$HPG_QOS" "$HPG_PARTITION" "$HPG_BLUE_ROOT" "$MAX_CONCURRENT_GPUS"; do
  case "$value" in
    ""|*\<*|*\>*) echo "Unset or placeholder HiPerGator value: $value" >&2; exit 2 ;;
  esac
done
if [[ ! -f "$PHASE4_MANIFEST" ]]; then
  source "$HPG_ENV/bin/activate"
  python -m kam.phase4.manifest --config "$PHASE4_CONFIG" --output "$PHASE4_MANIFEST"
fi
ROW_COUNT="$(wc -l < "$PHASE4_MANIFEST")"
EXPORTS="ALL,HPG_REPO=${HPG_REPO},HPG_ENV=${HPG_ENV},PHASE4_MANIFEST=${PHASE4_MANIFEST},PHASE4_RUN_ROOT=${PHASE4_RUN_ROOT}"
COMMON=(--parsable --account="$HPG_ACCOUNT" --qos="$HPG_QOS" --partition="$HPG_PARTITION" --export="$EXPORTS")
if [[ "$MODE" == "--plan-only" ]]; then
  cat <<PLAN
Phase IV HPG bounded data-regime screen
  account: $HPG_ACCOUNT
  qos: $HPG_QOS
  partition: $HPG_PARTITION
  repository: $HPG_REPO
  environment: $HPG_ENV
  max concurrent GPUs: $MAX_CONCURRENT_GPUS
  config: $PHASE4_CONFIG
  manifest: $PHASE4_MANIFEST
  rows: $ROW_COUNT
  run root: $PHASE4_RUN_ROOT
  logs: $LOG_ROOT
PLAN
  exit 0
fi
if [[ "$MODE" == "--submit" ]]; then
  mkdir -p "$LOG_ROOT" "$PHASE4_RUN_ROOT"
  array_job="$(sbatch "${COMMON[@]}" --array="0-$((ROW_COUNT - 1))%${MAX_CONCURRENT_GPUS}" --output="$LOG_ROOT/array_%A_%a.out" --error="$LOG_ROOT/array_%A_%a.err" "$HPG_REPO/slurm/phase4_array.sbatch" | tail -1)"
  report_job="$(sbatch "${COMMON[@]}" --dependency="afterok:${array_job}" --output="$LOG_ROOT/report_%j.out" --error="$LOG_ROOT/report_%j.err" "$HPG_REPO/slurm/phase4_aggregate.sbatch" | tail -1)"
  echo "array=$array_job report=$report_job"
  exit 0
fi
echo "Usage: $0 --plan-only|--submit" >&2
exit 2
