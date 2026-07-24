#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---plan-only}"
HPG_ACCOUNT="${HPG_ACCOUNT:-uf-dsi}"
HPG_QOS="${HPG_QOS:-uf-dsi}"
HPG_PARTITION="${HPG_PARTITION:-hpg-turin}"
HPG_GROUP="${HPG_GROUP:-uf-dsi}"
HPG_BLUE_ROOT="${HPG_BLUE_ROOT:-/blue/uf-dsi/rvalle1}"
HPG_REPO="${HPG_REPO:-${HPG_BLUE_ROOT}/KAM}"
HPG_ENV="${HPG_ENV:-${HPG_BLUE_ROOT}/venvs/kam}"
MAX_CONCURRENT_GPUS="${MAX_CONCURRENT_GPUS:-4}"
PHASE3_CONFIG="${PHASE3_CONFIG:-${HPG_REPO}/configs/phase3/cluster_development.yaml}"
PHASE3_MANIFEST="${PHASE3_MANIFEST:-${HPG_REPO}/results/phase3/manifests/cluster_development.jsonl}"
PHASE3_RUN_ROOT="${PHASE3_RUN_ROOT:-${HPG_REPO}/results/phase3/cluster_development}"
LOG_ROOT="${LOG_ROOT:-${HPG_BLUE_ROOT}/logs/phase3}"

for value in "$HPG_ACCOUNT" "$HPG_QOS" "$HPG_PARTITION" "$HPG_GROUP" "$HPG_BLUE_ROOT" "$MAX_CONCURRENT_GPUS"; do
  case "$value" in
    ""|*\<*|*\>*) echo "Unset or placeholder HiPerGator value: $value" >&2; exit 2 ;;
  esac
done

if [[ ! -f "$PHASE3_MANIFEST" ]]; then
  source "$HPG_ENV/bin/activate"
  python -m kam.phase3.manifest --config "$PHASE3_CONFIG" --output "$PHASE3_MANIFEST"
fi

ROW_COUNT="$(wc -l < "$PHASE3_MANIFEST")"
mkdir -p "$LOG_ROOT" "$HPG_REPO/results/phase3/gates"

EXPORTS="ALL,HPG_REPO=${HPG_REPO},HPG_ENV=${HPG_ENV},HPG_BLUE_ROOT=${HPG_BLUE_ROOT},PHASE3_MANIFEST=${PHASE3_MANIFEST},PHASE3_RUN_ROOT=${PHASE3_RUN_ROOT}"
COMMON=(--parsable --account="$HPG_ACCOUNT" --qos="$HPG_QOS" --partition="$HPG_PARTITION" --export="$EXPORTS")

submit_job() {
  local script="$1"
  shift
  sbatch "${COMMON[@]}" "$@" "$HPG_REPO/$script" | tail -1
}

if [[ "$MODE" == "--plan-only" ]]; then
  cat <<PLAN
Phase III HiPerGator plan
  account: $HPG_ACCOUNT
  qos: $HPG_QOS
  partition: $HPG_PARTITION
  group: $HPG_GROUP
  Blue root: $HPG_BLUE_ROOT
  repository: $HPG_REPO
  environment: $HPG_ENV
  max concurrent GPUs: $MAX_CONCURRENT_GPUS
  config: $PHASE3_CONFIG
  manifest: $PHASE3_MANIFEST
  rows: $ROW_COUNT
  run root: $PHASE3_RUN_ROOT
  logs: $LOG_ROOT
  CUDA module: cuda/12.8.1
PLAN
  exit 0
fi

case "$MODE" in
  --submit-audit)
    audit_job="$(submit_job slurm/phase3_audit_array.sbatch --array=0-0 --output="$LOG_ROOT/audit_%A_%a.out" --error="$LOG_ROOT/audit_%A_%a.err")"
    aggregate_job="$(submit_job slurm/phase3_audit_aggregate.sbatch --dependency="afterok:${audit_job}" --output="$LOG_ROOT/aggregate_a_%j.out" --error="$LOG_ROOT/aggregate_a_%j.err")"
    audit_exports="${EXPORTS},PHASE3_GATE_TYPE=audit,PHASE3_GATE_OUTPUT=${HPG_REPO}/results/phase3/gates/gate_a.json"
    gate_job="$(sbatch --parsable --account="$HPG_ACCOUNT" --qos="$HPG_QOS" --partition="$HPG_PARTITION" --export="$audit_exports" --dependency="afterok:${aggregate_job}" --output="$LOG_ROOT/gate_a_%j.out" --error="$LOG_ROOT/gate_a_%j.err" "$HPG_REPO/slurm/phase3_gate.sbatch")"
    echo "audit=$audit_job aggregate_a=$aggregate_job gate_a=$gate_job"
    ;;
  --submit-search-after-gate-a)
    gate_a="${PHASE3_GATE_A_JOB:-${gate_job:-}}"
    [[ -n "$gate_a" ]] || { echo "Set PHASE3_GATE_A_JOB to the Gate A job id." >&2; exit 2; }
    search_job="$(submit_job slurm/phase3_search_array.sbatch --dependency="afterok:${gate_a}" --array="0-$((ROW_COUNT - 1))%${MAX_CONCURRENT_GPUS}" --output="$LOG_ROOT/search_%A_%a.out" --error="$LOG_ROOT/search_%A_%a.err")"
    aggregate_search="$(submit_job slurm/phase3_aggregate.sbatch --dependency="afterok:${search_job}" --output="$LOG_ROOT/aggregate_search_%j.out" --error="$LOG_ROOT/aggregate_search_%j.err")"
    gate_b="$(submit_job slurm/phase3_gate.sbatch --dependency="afterok:${aggregate_search}" --output="$LOG_ROOT/gate_b_%j.out" --error="$LOG_ROOT/gate_b_%j.err")"
    echo "search=$search_job aggregate_search=$aggregate_search gate_b=$gate_b"
    ;;
  --submit-confirm-after-gates)
    gate_b="${PHASE3_GATE_B_JOB:-${gate_b:-}}"
    [[ -n "$gate_b" ]] || { echo "Set PHASE3_GATE_B_JOB to the development gate job id." >&2; exit 2; }
    confirm_manifest="${PHASE3_CONFIRM_MANIFEST:-${HPG_REPO}/results/phase3/manifests/confirmatory.jsonl}"
    confirm_root="${PHASE3_CONFIRM_ROOT:-${HPG_REPO}/results/phase3/confirmatory}"
    [[ -f "$confirm_manifest" ]] || { echo "Confirmatory manifest is not present: $confirm_manifest" >&2; exit 2; }
    export PHASE3_MANIFEST="$confirm_manifest" PHASE3_RUN_ROOT="$confirm_root"
    EXPORTS="ALL,HPG_REPO=${HPG_REPO},HPG_ENV=${HPG_ENV},HPG_BLUE_ROOT=${HPG_BLUE_ROOT},PHASE3_MANIFEST=${PHASE3_MANIFEST},PHASE3_RUN_ROOT=${PHASE3_RUN_ROOT}"
    COMMON=(--parsable --account="$HPG_ACCOUNT" --qos="$HPG_QOS" --partition="$HPG_PARTITION" --export="$EXPORTS")
    confirm_rows="$(wc -l < "$confirm_manifest")"
    confirm_job="$(submit_job slurm/phase3_confirm_array.sbatch --dependency="afterok:${gate_b}" --array="0-$((confirm_rows - 1))%${MAX_CONCURRENT_GPUS}" --output="$LOG_ROOT/confirm_%A_%a.out" --error="$LOG_ROOT/confirm_%A_%a.err")"
    aggregate_confirm="$(submit_job slurm/phase3_aggregate.sbatch --dependency="afterok:${confirm_job}" --output="$LOG_ROOT/aggregate_confirm_%j.out" --error="$LOG_ROOT/aggregate_confirm_%j.err")"
    final_gate="$(submit_job slurm/phase3_gate.sbatch --dependency="afterok:${aggregate_confirm}" --output="$LOG_ROOT/gate_final_%j.out" --error="$LOG_ROOT/gate_final_%j.err")"
    echo "confirm=$confirm_job aggregate_confirm=$aggregate_confirm final_gate=$final_gate"
    ;;
  --resume-failed)
    export PHASE3_MANIFEST PHASE3_RUN_ROOT
    search_job="$(submit_job slurm/phase3_search_array.sbatch --array="0-$((ROW_COUNT - 1))%${MAX_CONCURRENT_GPUS}" --output="$LOG_ROOT/resume_%A_%a.out" --error="$LOG_ROOT/resume_%A_%a.err")"
    echo "resume_search=$search_job"
    ;;
  *)
    if [[ "$MODE" != "--submit-audit" && "$MODE" != "--submit-search-after-gate-a" && "$MODE" != "--submit-confirm-after-gates" && "$MODE" != "--resume-failed" ]]; then
      echo "Usage: $0 --plan-only|--submit-audit|--submit-search-after-gate-a|--submit-confirm-after-gates|--resume-failed" >&2
      exit 2
    fi
    ;;
esac
