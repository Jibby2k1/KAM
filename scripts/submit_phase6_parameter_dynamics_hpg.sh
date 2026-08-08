#!/usr/bin/env bash
set -euo pipefail
MODE="${1:---plan-only}"
case "${MODE}" in --plan-only) STAGE="pilot";; --pilot) STAGE="pilot";; --main) STAGE="main";; *) echo "usage: $0 [--plan-only|--pilot|--main]" >&2; exit 2;; esac
HPG_ACCOUNT="${HPG_ACCOUNT:-uf-dsi}"; HPG_QOS="${HPG_QOS:-uf-dsi}"; HPG_PARTITION="${HPG_PARTITION:-hpg-turin}"
HPG_BLUE_ROOT="${HPG_BLUE_ROOT:-/blue/uf-dsi/rvalle1}"
HPG_REPO="${HPG_REPO:-${HPG_BLUE_ROOT}/KAM_parameter_dynamics_v1}"
HPG_ENV="${HPG_ENV:-${HPG_BLUE_ROOT}/venvs/kam}"
BASE="${PHASE6_PD_BASE:-${HPG_BLUE_ROOT}/KAM_parameter_dynamics_v1_results}"
RUN_ROOT="${PHASE6_PD_RUN_ROOT:-${BASE}/results/phase6/parameter_dynamics_v1/${STAGE}}"
REPORT_ROOT="${PHASE6_PD_REPORT_ROOT:-${BASE}/reports/phase6/parameter_dynamics_v1/${STAGE}}"
LOG_ROOT="${PHASE6_PD_LOG_ROOT:-${BASE}/logs/phase6/parameter_dynamics_v1/${STAGE}}"
MANIFEST="${RUN_ROOT}/manifest.jsonl"; GRAPH="${RUN_ROOT}/job_graph.json"
ROWS=10; [[ "${STAGE}" == "main" ]] && ROWS=60
printf 'Phase 6.1 parameter dynamics\n  stage: %s\n  rows: %s\n  checkout: %s\n  run root: %s\n  throttle/GPU: %%4 / one NVIDIA L4\n' "${STAGE}" "${ROWS}" "${HPG_REPO}" "${RUN_ROOT}"
[[ "${MODE}" == "--plan-only" ]] && exit 0
source "${HPG_ENV}/bin/activate"; cd "${HPG_REPO}"; export PYTHONPATH="${HPG_REPO}:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}/rows/parameter_dynamics_v1" "${REPORT_ROOT}" "${LOG_ROOT}"
if [[ -f "${GRAPH}" ]]; then echo "Existing graph requires inspection: ${GRAPH}" >&2; exit 1; fi
if [[ "${STAGE}" == "main" ]] && [[ -n "$(git status --porcelain)" ]]; then
  echo "Main blocked: HPG checkout must be a clean committed tree" >&2
  exit 1
fi
if [[ "${STAGE}" == "main" ]]; then
  PILOT_SUMMARY="${BASE}/results/phase6/parameter_dynamics_v1/pilot/parameter_dynamics_summary.json"
  python - "${PILOT_SUMMARY}" <<'PY'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1]); payload = json.loads(path.read_text()) if path.is_file() else {}
if payload.get("decision") != "PILOT_PASS" or len(payload.get("figures", [])) != 16:
    raise SystemExit("Main blocked: instrumentation pilot and all eight PNG/SVG figures must pass")
PY
fi
python scripts/build_phase6_parameter_dynamics_manifest.py --stage "${STAGE}" --output "${MANIFEST}"
python -m pytest -q tests/test_phase6_parameter_dynamics.py -k manifest
common="ALL,PHASE6_PD_REPO=${HPG_REPO},PHASE6_PD_ENV=${HPG_ENV},PHASE6_PD_RUN_ROOT=${RUN_ROOT},PHASE6_PD_REPORT_ROOT=${REPORT_ROOT},PHASE6_PD_MANIFEST=${MANIFEST}"
last=$((ROWS - 1))
array_job="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --time=08:00:00 --array="0-${last}%4" --export="${common}" --output="${LOG_ROOT}/row_%A_%a.out" --error="${LOG_ROOT}/row_%A_%a.err" "${HPG_REPO}/slurm/phase6_parameter_dynamics_array.sbatch")"
final_job="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --dependency="afterany:${array_job}" --export="${common}" --output="${LOG_ROOT}/final_%j.out" --error="${LOG_ROOT}/final_%j.err" "${HPG_REPO}/slurm/phase6_parameter_dynamics_final.sbatch")"
python - "${GRAPH}" "${MANIFEST}" "${STAGE}" "${ROWS}" "${array_job}" "${final_job}" <<'PY'
import datetime, hashlib, json, pathlib, subprocess, sys
graph, manifest, stage, rows, array_job, final_job = sys.argv[1:]
data = pathlib.Path(manifest).read_bytes()
payload = {"campaign":"phase6_parameter_dynamics_v1","stage":stage,"submission_status":"submitted","submitted_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"manifest":manifest,"manifest_sha256":hashlib.sha256(data).hexdigest(),"row_count":int(rows),"fixed_sample_no_optional_stopping":True,"jobs":{"array":array_job,"final_report":final_job},"dependency":"afterany:array","array_throttle":4,"gpu_type":"NVIDIA L4","wall_limit":"08:00:00","git_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"git_dirty":bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip())}
pathlib.Path(graph).write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n"); print(json.dumps(payload, indent=2, sort_keys=True))
PY
