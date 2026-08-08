#!/usr/bin/env bash
set -euo pipefail

MODE="${1:---plan-only}"
case "${MODE}" in
  --plan-only) STAGE="l4_profile" ;;
  --profile) STAGE="l4_profile_r3" ;;
  --stage0) STAGE="stage0" ;;
  *) echo "usage: $0 [--plan-only|--profile|--stage0]" >&2; exit 2 ;;
esac

HPG_ACCOUNT="${HPG_ACCOUNT:-uf-dsi}"
HPG_QOS="${HPG_QOS:-uf-dsi}"
HPG_PARTITION="${HPG_PARTITION:-hpg-turin}"
HPG_BLUE_ROOT="${HPG_BLUE_ROOT:-/blue/uf-dsi/rvalle1}"
HPG_REPO="${HPG_REPO:-${HPG_BLUE_ROOT}/KAM_behavioral_atlas_v2_20260730}"
HPG_ENV="${HPG_ENV:-${HPG_BLUE_ROOT}/venvs/kam}"
BASE="${PHASE6_ATLAS_BASE:-${HPG_BLUE_ROOT}/KAM_behavioral_atlas_v2_results}"
RUN_ROOT="${PHASE6_ATLAS_RUN_ROOT:-${BASE}/results/phase6/behavioral_atlas_v2/${STAGE}}"
REPORT_ROOT="${PHASE6_ATLAS_REPORT_ROOT:-${BASE}/reports/phase6/behavioral_atlas_v2/${STAGE}}"
LOG_ROOT="${PHASE6_ATLAS_LOG_ROOT:-${BASE}/logs/phase6/behavioral_atlas_v2/${STAGE}}"
MANIFEST="${RUN_ROOT}/manifest.jsonl"
GRAPH="${RUN_ROOT}/job_graph.json"
ROWS=1
THROTTLE=1
WALL_LIMIT="02:00:00"
if [[ "${STAGE}" == "stage0" ]]; then
  ROWS=24
  THROTTLE=4
  WALL_LIMIT="04:00:00"
fi

printf 'Phase 6.2 behavioral atlas\n  stage: %s\n  rows: %s\n  checkout: %s\n  run root: %s\n  throttle/GPU: %s / one NVIDIA L4\n' "${STAGE}" "${ROWS}" "${HPG_REPO}" "${RUN_ROOT}" "${THROTTLE}"
[[ "${MODE}" == "--plan-only" ]] && exit 0

source "${HPG_ENV}/bin/activate"
cd "${HPG_REPO}"
export PYTHONPATH="${HPG_REPO}:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}/rows/behavioral_atlas_v2" "${REPORT_ROOT}" "${LOG_ROOT}"
if [[ -f "${GRAPH}" ]]; then
  echo "Existing graph requires inspection: ${GRAPH}" >&2
  exit 1
fi
if [[ "${STAGE}" == "stage0" ]]; then
  PROFILE_SUMMARY="${BASE}/results/phase6/behavioral_atlas_v2/l4_profile_r3/behavioral_atlas_summary.json"
  python - "${PROFILE_SUMMARY}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]); payload = json.loads(path.read_text()) if path.is_file() else {}
if payload.get("decision") != "L4_PROFILE_PASS":
    raise SystemExit("Stage 0 blocked: bounded NVIDIA L4 profile must pass")
PY
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "Stage 0 blocked: HPG checkout must be a clean committed tree" >&2
    exit 1
  fi
fi

python scripts/build_phase6_behavioral_atlas_manifest.py --stage "${STAGE}" --output "${MANIFEST}"
python -m pytest -q tests/test_phase6_behavioral_atlas.py -k manifest
common="ALL,PHASE6_ATLAS_REPO=${HPG_REPO},PHASE6_ATLAS_ENV=${HPG_ENV},PHASE6_ATLAS_RUN_ROOT=${RUN_ROOT},PHASE6_ATLAS_REPORT_ROOT=${REPORT_ROOT},PHASE6_ATLAS_MANIFEST=${MANIFEST}"
last=$((ROWS - 1))
array_job="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --time="${WALL_LIMIT}" --array="0-${last}%${THROTTLE}" --export="${common}" --output="${LOG_ROOT}/row_%A_%a.out" --error="${LOG_ROOT}/row_%A_%a.err" "${HPG_REPO}/slurm/phase6_behavioral_atlas_array.sbatch")"
final_job="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --dependency="afterany:${array_job}" --export="${common}" --output="${LOG_ROOT}/final_%j.out" --error="${LOG_ROOT}/final_%j.err" "${HPG_REPO}/slurm/phase6_behavioral_atlas_final.sbatch")"
python - "${GRAPH}" "${MANIFEST}" "${STAGE}" "${ROWS}" "${array_job}" "${final_job}" <<'PY'
import datetime, hashlib, json, pathlib, subprocess, sys
graph, manifest, stage, rows, array_job, final_job = sys.argv[1:]
data = pathlib.Path(manifest).read_bytes()
payload = {
    "campaign": "phase6_behavioral_atlas_v2",
    "stage": stage,
    "submission_status": "submitted",
    "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "manifest": manifest,
    "manifest_sha256": hashlib.sha256(data).hexdigest(),
    "row_count": int(rows),
    "inferential": False,
    "jobs": {"array": array_job, "final_report": final_job},
    "dependency": "afterany:array",
    "gpu_type": "NVIDIA L4",
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
}
pathlib.Path(graph).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
