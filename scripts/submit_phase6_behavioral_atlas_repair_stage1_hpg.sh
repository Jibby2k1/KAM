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
CORPUS_SOURCE_REPO="${PHASE6_ATLAS_CORPUS_SOURCE_REPO:-${HPG_BLUE_ROOT}/KAM_behavioral_atlas_stage0_f7d69ec}"
BASE="${PHASE6_ATLAS_BASE:-${HPG_BLUE_ROOT}/KAM_behavioral_atlas_v2_results}"
STAGE0_ROOT="${PHASE6_ATLAS_STAGE0_ROOT:-${BASE}/results/phase6/behavioral_atlas_v2/stage0}"
STAGE0_MANIFEST="${PHASE6_ATLAS_STAGE0_MANIFEST:-${STAGE0_ROOT}/manifest.jsonl}"
REPAIR_ROOT="${PHASE6_ATLAS_REPAIR_ROOT:-${BASE}/results/phase6/behavioral_atlas_v2/stage0_measurement_repair_r2}"
REPAIR_REPORT_ROOT="${PHASE6_ATLAS_REPAIR_REPORT_ROOT:-${BASE}/reports/phase6/behavioral_atlas_v2/stage0_measurement_repair_r2}"
REPAIR_LOG_ROOT="${PHASE6_ATLAS_REPAIR_LOG_ROOT:-${BASE}/logs/phase6/behavioral_atlas_v2/stage0_measurement_repair_r2}"
REPAIR_MANIFEST="${REPAIR_ROOT}/manifest.jsonl"
STAGE1_ROOT="${PHASE6_ATLAS_STAGE1_ROOT:-${BASE}/results/phase6/behavioral_atlas_v2/stage1_core_lifecycle_r1}"
STAGE1_SMOKE_ROOT="${PHASE6_ATLAS_STAGE1_SMOKE_ROOT:-${BASE}/results/phase6/behavioral_atlas_v2/stage1_core_lifecycle_r1_gpu_preflight}"
STAGE1_REPORT_ROOT="${PHASE6_ATLAS_STAGE1_REPORT_ROOT:-${BASE}/reports/phase6/behavioral_atlas_v2/stage1_core_lifecycle_r1}"
STAGE1_LOG_ROOT="${PHASE6_ATLAS_STAGE1_LOG_ROOT:-${BASE}/logs/phase6/behavioral_atlas_v2/stage1_core_lifecycle_r1}"
STAGE1_MANIFEST="${STAGE1_ROOT}/manifest.jsonl"
GRAPH="${REPAIR_ROOT}/repair_stage1_job_graph.json"

printf 'Phase 6.2 repair r2 and Stage 1 gated deployment\n  checkout: %s\n  original Stage 0: %s\n  corpus source: %s\n  repair: 2 L4 rows, %%2\n  Stage 1 GPU preflight: cosine arm, 250k-token noninferential smoke\n  Stage 1: 168 L4 rows, %%4, dependency afterok GPU preflight\n  Stage 1 per-row wall limit: 08:00:00\n  Stage 1 projected: 670-720 L4 GPU-hours\n' "${HPG_REPO}" "${STAGE0_ROOT}" "${CORPUS_SOURCE_REPO}"
[[ "${MODE}" == "--plan-only" ]] && exit 0

source "${HPG_ENV}/bin/activate"
cd "${HPG_REPO}"
export PYTHONPATH="${HPG_REPO}:${PYTHONPATH:-}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Deployment blocked: HPG checkout must be a clean committed tree" >&2
  exit 1
fi
if [[ ! -f "${STAGE0_MANIFEST}" ]]; then
  echo "Deployment blocked: original Stage 0 manifest is missing: ${STAGE0_MANIFEST}" >&2
  exit 1
fi
exclude_path="$(git rev-parse --git-path info/exclude)"
grep -qxF "/data/phase6_confirmation/" "${exclude_path}" || printf "%s\n" "/data/phase6_confirmation/" >> "${exclude_path}"
mkdir -p "${HPG_REPO}/data/phase6_confirmation"
for corpus_file in TinyStoriesV2-GPT4-train.128MiB.txt TinyStories-valid.validation.txt TinyStories-valid.test.txt; do
  source_path="${CORPUS_SOURCE_REPO}/data/phase6_confirmation/${corpus_file}"
  if [[ ! -s "${source_path}" ]]; then
    echo "Deployment blocked: corpus source is missing: ${source_path}" >&2
    exit 1
  fi
  ln -sf "${source_path}" "${HPG_REPO}/data/phase6_confirmation/${corpus_file}"
done
if [[ -f "${GRAPH}" ]]; then
  echo "Existing job graph requires inspection: ${GRAPH}" >&2
  exit 1
fi
mkdir -p "${REPAIR_ROOT}/rows/repair" "${REPAIR_REPORT_ROOT}" "${REPAIR_LOG_ROOT}" \
  "${STAGE1_ROOT}/rows/behavioral_atlas_v2" "${STAGE1_SMOKE_ROOT}/rows/behavioral_atlas_v2" \
  "${STAGE1_REPORT_ROOT}" "${STAGE1_LOG_ROOT}"
python -m kam.phase6.behavioral_atlas_repair manifest --stage0-manifest "${STAGE0_MANIFEST}" --output "${REPAIR_MANIFEST}"
python scripts/build_phase6_behavioral_atlas_manifest.py --stage stage1_core_lifecycle --output "${STAGE1_MANIFEST}"
python -m pytest -q tests/test_phase6_behavioral_atlas.py tests/test_phase6_behavioral_atlas_repair.py -k "manifest or paired_randomization"

repair_export="ALL,PHASE6_ATLAS_REPO=${HPG_REPO},PHASE6_ATLAS_ENV=${HPG_ENV},PHASE6_ATLAS_STAGE0_ROOT=${STAGE0_ROOT},PHASE6_ATLAS_STAGE0_MANIFEST=${STAGE0_MANIFEST},PHASE6_ATLAS_REPAIR_ROOT=${REPAIR_ROOT},PHASE6_ATLAS_REPAIR_REPORT_ROOT=${REPAIR_REPORT_ROOT},PHASE6_ATLAS_REPAIR_MANIFEST=${REPAIR_MANIFEST}"
stage1_export="ALL,PHASE6_ATLAS_REPO=${HPG_REPO},PHASE6_ATLAS_ENV=${HPG_ENV},PHASE6_ATLAS_STAGE1_ROOT=${STAGE1_ROOT},PHASE6_ATLAS_STAGE1_REPORT_ROOT=${STAGE1_REPORT_ROOT},PHASE6_ATLAS_STAGE1_MANIFEST=${STAGE1_MANIFEST}"
stage1_smoke_export="ALL,PHASE6_ATLAS_REPO=${HPG_REPO},PHASE6_ATLAS_ENV=${HPG_ENV},PHASE6_ATLAS_STAGE1_ROOT=${STAGE1_SMOKE_ROOT},PHASE6_ATLAS_STAGE1_REPORT_ROOT=${STAGE1_REPORT_ROOT},PHASE6_ATLAS_STAGE1_MANIFEST=${STAGE1_MANIFEST},PHASE6_BEHAVIORAL_ATLAS_SMOKE_TOKENS=250000"
repair_array="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --time=04:00:00 --array=0-1%2 --export="${repair_export}" --output="${REPAIR_LOG_ROOT}/row_%A_%a.out" --error="${REPAIR_LOG_ROOT}/row_%A_%a.err" "${HPG_REPO}/slurm/phase6_behavioral_atlas_repair_array.sbatch")"
repair_report="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --dependency="afterany:${repair_array}" --export="${repair_export}" --output="${REPAIR_LOG_ROOT}/final_%j.out" --error="${REPAIR_LOG_ROOT}/final_%j.err" "${HPG_REPO}/slurm/phase6_behavioral_atlas_repair_final.sbatch")"
stage1_smoke="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --time=01:00:00 --array=123 --dependency="afterok:${repair_report}" --export="${stage1_smoke_export}" --output="${STAGE1_LOG_ROOT}/smoke_r2_%A_%a.out" --error="${STAGE1_LOG_ROOT}/smoke_r2_%A_%a.err" "${HPG_REPO}/slurm/phase6_behavioral_atlas_stage1_array.sbatch")"
stage1_array="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --time=08:00:00 --array=0-167%4 --dependency="afterok:${stage1_smoke}" --export="${stage1_export}" --output="${STAGE1_LOG_ROOT}/row_r2_%A_%a.out" --error="${STAGE1_LOG_ROOT}/row_r2_%A_%a.err" "${HPG_REPO}/slurm/phase6_behavioral_atlas_stage1_array.sbatch")"
stage1_report="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --dependency="afterany:${stage1_array}" --export="${stage1_export}" --output="${STAGE1_LOG_ROOT}/final_r2_%j.out" --error="${STAGE1_LOG_ROOT}/final_r2_%j.err" "${HPG_REPO}/slurm/phase6_behavioral_atlas_stage1_final.sbatch")"
python - "${GRAPH}" "${REPAIR_MANIFEST}" "${STAGE1_MANIFEST}" "${repair_array}" "${repair_report}" "${stage1_smoke}" "${stage1_array}" "${stage1_report}" <<'PY'
import datetime, hashlib, json, pathlib, subprocess, sys
graph, repair_manifest, stage1_manifest, repair_array, repair_report, stage1_smoke, stage1_array, stage1_report = sys.argv[1:]
def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
payload = {
    "campaign": "phase6_behavioral_atlas_v2",
    "repair_revision": "stage0_measurement_repair_r2",
    "submission_status": "submitted",
    "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    "tracked_source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True).strip()),
    "gpu_type": "NVIDIA L4",
    "repair": {
        "manifest": repair_manifest,
        "manifest_sha256": digest(repair_manifest),
        "rows": 2,
        "jobs": {"array": repair_array, "audit": repair_report},
    },
    "stage1": {
        "manifest": stage1_manifest,
        "manifest_sha256": digest(stage1_manifest),
        "rows": 168,
        "inferential": True,
        "throttle": 4,
        "per_row_wall_limit": "08:00:00",
        "projected_l4_gpu_hours": [670, 720],
        "jobs": {"gpu_preflight": stage1_smoke, "array": stage1_array, "report": stage1_report},
    },
    "dependencies": {
        "repair_audit": f"afterany:{repair_array}",
        "stage1_gpu_preflight": f"afterok:{repair_report}",
        "stage1_array": f"afterok:{stage1_smoke}",
        "stage1_report": f"afterany:{stage1_array}",
    },
    "execution_policy": "Stage 1 is locked to eager execution for this inferential manifest; the repaired compile candidate is descriptive systems evidence for later stages.",
}
pathlib.Path(graph).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
