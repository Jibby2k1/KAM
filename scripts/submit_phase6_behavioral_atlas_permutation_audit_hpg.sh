#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---plan-only}"
case "${MODE}" in
  --plan-only|--submit) ;;
  *) echo "usage: $0 [--plan-only|--submit]" >&2; exit 2 ;;
esac

HPG_ACCOUNT="${HPG_ACCOUNT:-uf-dsi}"
HPG_QOS="${HPG_QOS:-uf-dsi}"
HPG_PARTITION="${HPG_PARTITION:-hpg-turin}"
HPG_BLUE_ROOT="${HPG_BLUE_ROOT:-/blue/uf-dsi/rvalle1}"
HPG_REPO="${HPG_REPO:-${HPG_BLUE_ROOT}/KAM_behavioral_atlas_repair_stage1}"
HPG_ENV="${HPG_ENV:-${HPG_BLUE_ROOT}/venvs/kam}"
BASE="${PHASE6_ATLAS_BASE:-${HPG_BLUE_ROOT}/KAM_behavioral_atlas_v2_results}"
STAGE1_ROOT="${PHASE6_ATLAS_STAGE1_ROOT:-${BASE}/results/phase6/behavioral_atlas_v2/stage1_core_lifecycle_r1}"
AUDIT_ROOT="${PHASE6_ATLAS_PERMUTATION_AUDIT_ROOT:-${BASE}/results/phase6/behavioral_atlas_v2/stage1_permutation_checkpoint_audit_r1}"
SMOKE_ROOT="${PHASE6_ATLAS_PERMUTATION_AUDIT_SMOKE_ROOT:-${BASE}/results/phase6/behavioral_atlas_v2/stage1_permutation_checkpoint_audit_r1_gpu_preflight}"
REPORT_ROOT="${PHASE6_ATLAS_PERMUTATION_AUDIT_REPORT_ROOT:-${BASE}/reports/phase6/behavioral_atlas_v2/stage1_permutation_checkpoint_audit_r1}"
LOG_ROOT="${PHASE6_ATLAS_PERMUTATION_AUDIT_LOG_ROOT:-${BASE}/logs/phase6/behavioral_atlas_v2/stage1_permutation_checkpoint_audit_r1}"
MANIFEST="${AUDIT_ROOT}/manifest.jsonl"
GRAPH="${AUDIT_ROOT}/job_graph.json"

printf 'Phase 6.2 Stage 1 checkpoint permutation audit\n  checkout: %s\n  source: %s\n  output: %s\n  execution: checkpoint-only L4 array, %%4\n' \
  "${HPG_REPO}" "${STAGE1_ROOT}" "${AUDIT_ROOT}"
[[ "${MODE}" == "--plan-only" ]] && exit 0

source "${HPG_ENV}/bin/activate"
cd "${HPG_REPO}"
export PYTHONPATH="${HPG_REPO}:${PYTHONPATH:-}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Deployment blocked: HPG checkout must be a clean committed tree" >&2
  exit 1
fi
if [[ -f "${GRAPH}" ]]; then
  echo "Existing audit job graph requires inspection: ${GRAPH}" >&2
  exit 1
fi
mkdir -p "${AUDIT_ROOT}/rows" "${SMOKE_ROOT}/rows" "${REPORT_ROOT}" "${LOG_ROOT}"
python -m kam.phase6.behavioral_atlas_permutation_audit manifest \
  --stage1-root "${STAGE1_ROOT}" \
  --output "${MANIFEST}"
python -m pytest -q tests/test_phase6_behavioral_atlas_permutation_audit.py
rows="$(wc -l < "${MANIFEST}")"
if [[ "${rows}" -lt 1 ]]; then
  echo "Deployment blocked: audit manifest is empty" >&2
  exit 1
fi
last_index="$((rows - 1))"
common_export="PHASE6_ATLAS_REPO=${HPG_REPO},PHASE6_ATLAS_ENV=${HPG_ENV},PHASE6_ATLAS_STAGE1_ROOT=${STAGE1_ROOT},PHASE6_ATLAS_PERMUTATION_AUDIT_MANIFEST=${MANIFEST}"
smoke_export="ALL,${common_export},PHASE6_ATLAS_PERMUTATION_AUDIT_ROOT=${SMOKE_ROOT}"
audit_export="ALL,${common_export},PHASE6_ATLAS_PERMUTATION_AUDIT_ROOT=${AUDIT_ROOT},PHASE6_ATLAS_PERMUTATION_AUDIT_REPORT_ROOT=${REPORT_ROOT}"
smoke_job="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --array=0 --export="${smoke_export}" --output="${LOG_ROOT}/smoke_%A_%a.out" --error="${LOG_ROOT}/smoke_%A_%a.err" "${HPG_REPO}/slurm/phase6_behavioral_atlas_permutation_audit_array.sbatch")"
audit_job="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --array="0-${last_index}%4" --dependency="afterok:${smoke_job}" --export="${audit_export}" --output="${LOG_ROOT}/row_%A_%a.out" --error="${LOG_ROOT}/row_%A_%a.err" "${HPG_REPO}/slurm/phase6_behavioral_atlas_permutation_audit_array.sbatch")"
report_job="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --dependency="afterany:${audit_job}" --export="${audit_export}" --output="${LOG_ROOT}/final_%j.out" --error="${LOG_ROOT}/final_%j.err" "${HPG_REPO}/slurm/phase6_behavioral_atlas_permutation_audit_final.sbatch")"
python - "${GRAPH}" "${MANIFEST}" "${rows}" "${smoke_job}" "${audit_job}" "${report_job}" <<'PY'
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

graph, manifest, rows, smoke, audit, report = sys.argv[1:]
payload = {
    "campaign": "phase6_behavioral_atlas_v2",
    "audit": "stage1_permutation_checkpoint_audit_r1",
    "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "manifest": manifest,
    "manifest_sha256": hashlib.sha256(pathlib.Path(manifest).read_bytes()).hexdigest(),
    "rows": int(rows),
    "retraining": False,
    "inferential": False,
    "gpu_type": "NVIDIA L4",
    "jobs": {"gpu_preflight": smoke, "array": audit, "report": report},
    "dependencies": {"array": f"afterok:{smoke}", "report": f"afterany:{audit}"},
}
pathlib.Path(graph).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
