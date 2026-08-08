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
BLUE="${HPG_BLUE_ROOT:-/blue/uf-dsi/rvalle1}"
REPO="${PHASE6_LOCALIZATION_REPO:-${BLUE}/KAM_permutation_localization_clean}"
ENV="${PHASE6_LOCALIZATION_ENV:-${BLUE}/venvs/kam_permutation_localization_clean}"
BASE="${PHASE6_ATLAS_BASE:-${BLUE}/KAM_behavioral_atlas_v2_results}"
STAGE1_ROOT="${PHASE6_ATLAS_STAGE1_ROOT:-${BASE}/results/phase6/behavioral_atlas_v2/stage1_core_lifecycle_r1}"
AUDIT_ROOT="${PHASE6_ATLAS_PERMUTATION_AUDIT_ROOT:-${BASE}/results/phase6/behavioral_atlas_v2/stage1_permutation_checkpoint_audit_r1}"
ROOT="${PHASE6_LOCALIZATION_ROOT:-${BASE}/results/phase6/behavioral_atlas_v2/stage1_permutation_localization_r1}"
REPORT="${PHASE6_LOCALIZATION_REPORT_ROOT:-${BASE}/reports/phase6/behavioral_atlas_v2/stage1_permutation_localization_r1}"
LOGS="${PHASE6_LOCALIZATION_LOG_ROOT:-${BASE}/logs/phase6/behavioral_atlas_v2/stage1_permutation_localization_r1}"
MANIFEST="${ROOT}/manifest.jsonl"
GRAPH="${ROOT}/job_graph.json"

printf 'Two-checkpoint permutation localization\n  clean checkout: %s\n  clean environment: %s\n  GPU: 6 independent L4 processes\n  CPU: 2 strict-reference processes\n' "${REPO}" "${ENV}"
[[ "${MODE}" == "--plan-only" ]] && exit 0

if [[ ! -x "${ENV}/bin/python" ]]; then
  echo "Deployment blocked: clean environment missing: ${ENV}" >&2
  exit 1
fi
cd "${REPO}"
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Deployment blocked: localization checkout must be clean" >&2
  exit 1
fi
source "${ENV}/bin/activate"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
mkdir -p "${ROOT}/rows" "${REPORT}" "${LOGS}"
if [[ -f "${GRAPH}" ]]; then
  echo "Existing localization graph requires inspection: ${GRAPH}" >&2
  exit 1
fi
python -m kam.phase6.behavioral_atlas_permutation_localization manifest \
  --audit-root "${AUDIT_ROOT}" --output "${MANIFEST}"
python -m pytest -q tests/test_phase6_behavioral_atlas_permutation_localization.py
export_spec="ALL,PHASE6_LOCALIZATION_REPO=${REPO},PHASE6_LOCALIZATION_ENV=${ENV},PHASE6_ATLAS_STAGE1_ROOT=${STAGE1_ROOT},PHASE6_LOCALIZATION_MANIFEST=${MANIFEST},PHASE6_LOCALIZATION_ROOT=${ROOT},PHASE6_LOCALIZATION_REPORT_ROOT=${REPORT}"
gpu="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --array=0-5%3 --export="${export_spec}" --output="${LOGS}/gpu_%A_%a.out" --error="${LOGS}/gpu_%A_%a.err" "${REPO}/slurm/phase6_behavioral_atlas_permutation_localization_gpu.sbatch")"
cpu="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --array=6-7%2 --export="${export_spec}" --output="${LOGS}/cpu_%A_%a.out" --error="${LOGS}/cpu_%A_%a.err" "${REPO}/slurm/phase6_behavioral_atlas_permutation_localization_cpu.sbatch")"
final="$(sbatch --parsable --account="${HPG_ACCOUNT}" --qos="${HPG_QOS}" --partition="${HPG_PARTITION}" --dependency="afterany:${gpu}:${cpu}" --export="${export_spec}" --output="${LOGS}/final_%j.out" --error="${LOGS}/final_%j.err" "${REPO}/slurm/phase6_behavioral_atlas_permutation_localization_final.sbatch")"
python - "${GRAPH}" "${MANIFEST}" "${gpu}" "${cpu}" "${final}" <<'PY'
import datetime, hashlib, json, pathlib, subprocess, sys
graph, manifest, gpu, cpu, final = sys.argv[1:]
payload = {
    "campaign": "phase6_behavioral_atlas_v2",
    "localization": "stage1_permutation_localization_r1",
    "submitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "manifest": manifest,
    "manifest_sha256": hashlib.sha256(pathlib.Path(manifest).read_bytes()).hexdigest(),
    "retraining": False,
    "inferential": False,
    "jobs": {"gpu": gpu, "cpu": cpu, "report": final},
    "dependencies": {"report": f"afterany:{gpu}:{cpu}"},
}
pathlib.Path(graph).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, indent=2, sort_keys=True))
PY
