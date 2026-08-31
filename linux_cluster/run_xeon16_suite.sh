#!/usr/bin/env bash
# Functional and strong-scaling validation for one 16-physical-core Linux host.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
INPUT_ROOT="${SCRIPT_DIR}/inputs/xeon16"

SUITE_MODE="${1:-${SUITE_MODE:-all}}"
MAX_RANKS="${MAX_RANKS:-16}"
RANK_SWEEP="${RANK_SWEEP:-1 2 4 8 16}"
RANK_SWEEP="${RANK_SWEEP//$'\r'/ }"
RANK_SWEEP="${RANK_SWEEP//$'\n'/ }"
RANK_SWEEP="${RANK_SWEEP//$'\t'/ }"
SCALING_REPEATS="${SCALING_REPEATS:-1}"
EXPECTED_NODES="${EXPECTED_NODES:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
ENV_NAME="${ENV_NAME:-phasefield-fenicsx-linux}"
ALLOW_UNKNOWN_TOPOLOGY="${ALLOW_UNKNOWN_TOPOLOGY:-0}"
ALLOW_PETSC_OPTIONS="${ALLOW_PETSC_OPTIONS:-0}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-}"
SUITE_ID="xeon16_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
OUT_ROOT="${OUT_ROOT:-${PROJECT_ROOT}/results/${SUITE_ID}}"

case "${SUITE_MODE}" in
  smoke|validation|scaling|all) ;;
  *)
    echo "Usage: bash linux_cluster/run_xeon16_suite.sh [smoke|validation|scaling|all]" >&2
    exit 2
    ;;
esac
if ! [[ "${MAX_RANKS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_RANKS must be a positive integer." >&2
  exit 2
fi
if ! [[ "${SCALING_REPEATS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "SCALING_REPEATS must be a positive integer." >&2
  exit 2
fi
if [[ "${EXPECTED_NODES}" != "1" ]]; then
  echo "The Xeon16 suite is a one-workstation test; EXPECTED_NODES must be 1." >&2
  exit 2
fi
if [[ -e "${OUT_ROOT}" ]]; then
  echo "Suite output already exists; choose a new OUT_ROOT: ${OUT_ROOT}" >&2
  exit 2
fi

if [[ -n "${CONDA_BASE:-}" ]]; then
  if [[ ! -r "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    echo "CONDA_BASE does not contain a readable conda.sh: ${CONDA_BASE}" >&2
    exit 2
  fi
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "Conda was not found. Load Miniconda or set CONDA_BASE." >&2
  exit 2
fi
conda activate "${ENV_NAME}"
hash -r
export ENV_NAME
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX}/bin/python}"
if ! "${PYTHON_BIN}" -c 'import sys; assert sys.version_info >= (3, 10)' \
  >/dev/null 2>&1; then
  echo "The Xeon16 suite requires Python 3.10 or newer." >&2
  exit 2
fi

# This is an MPI strong-scaling suite: one software thread per MPI rank is
# mandatory, regardless of inherited interactive-shell settings.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
if [[ -n "${PETSC_OPTIONS:-}" && "${ALLOW_PETSC_OPTIONS}" != "1" ]]; then
  echo "PETSC_OPTIONS is set and would invalidate the reference comparisons." >&2
  echo "Unset it, or set ALLOW_PETSC_OPTIONS=1 for a deliberately modified suite." >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/runs"
OUT_ROOT="$(cd -- "${OUT_ROOT}" && pwd)"
LEDGER="${OUT_ROOT}/cases.tsv"
SETTINGS="${OUT_ROOT}/suite_settings.tsv"
HARDWARE_REPORT="${OUT_ROOT}/hardware.json"
PREFLIGHT_REPORT="${OUT_ROOT}/preflight.json"
PREFLIGHT_LOG="${OUT_ROOT}/logs/preflight.log"

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  order case category ranks repeat input_sha256 input_path output_path \
  log_path start_utc end_utc wall_seconds launcher_rc > "${LEDGER}"
{
  printf 'key\tvalue\n'
  printf 'suite_id\t%s\n' "${SUITE_ID}"
  printf 'suite_mode\t%s\n' "${SUITE_MODE}"
  printf 'started_utc\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'project_root\t%s\n' "${PROJECT_ROOT}"
  printf 'max_ranks\t%s\n' "${MAX_RANKS}"
  printf 'rank_sweep\t%s\n' "${RANK_SWEEP}"
  printf 'scaling_repeats\t%s\n' "${SCALING_REPEATS}"
  printf 'expected_nodes\t%s\n' "${EXPECTED_NODES}"
  printf 'conda_environment\t%s\n' "${ENV_NAME}"
  printf 'conda_prefix\t%s\n' "${CONDA_PREFIX}"
  printf 'mpi_preference\t%s\n' "${MPI_PREFERENCE:-${MPI_LAUNCHER:-auto}}"
  printf 'mpi_home\t%s\n' "${MPI_HOME:-}"
  printf 'mpi_launcher_legacy\t%s\n' "${MPI_LAUNCHER:-}"
  printf 'mpi_extra_args\t%s\n' "${MPI_EXTRA_ARGS}"
  printf 'fail_fast\t%s\n' "${FAIL_FAST}"
  printf 'petsc_options\t%s\n' "${PETSC_OPTIONS:-}"
  printf 'allow_petsc_options\t%s\n' "${ALLOW_PETSC_OPTIONS}"
} > "${SETTINGS}"

declare -a HARDWARE_ARGS
HARDWARE_ARGS=(
  --max-ranks "${MAX_RANKS}"
  --rank-sweep "${RANK_SWEEP}"
  --mpi-extra-args "${MPI_EXTRA_ARGS}"
  --output "${HARDWARE_REPORT}"
)
if [[ "${SUITE_MODE}" == "scaling" || "${SUITE_MODE}" == "all" ]]; then
  HARDWARE_ARGS+=(--require-sweep-endpoints)
fi
if [[ "${ALLOW_UNKNOWN_TOPOLOGY}" == "1" ]]; then
  HARDWARE_ARGS+=(--allow-unknown-topology)
fi

echo "Checking affinity-visible physical cores..."
if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/xeon16_hardware.py" \
  "${HARDWARE_ARGS[@]}"; then
  echo "Hardware-capacity check failed; no MPI job was launched." >&2
  exit 2
fi
if ! RANKS_TEXT="$(
  "${PYTHON_BIN}" - "${HARDWARE_REPORT}" <<'PY'
import json
from pathlib import Path
import sys
for rank in json.loads(Path(sys.argv[1]).read_text())["requested_rank_sweep"]:
    print(rank)
PY
)"; then
  echo "Could not read the validated rank sweep from hardware.json." >&2
  exit 2
fi
if [[ -z "${RANKS_TEXT}" ]]; then
  echo "Validated rank sweep is empty." >&2
  exit 2
fi
mapfile -t RANKS <<< "${RANKS_TEXT}"

echo "Running one fatal ${MAX_RANKS}-rank PETSc/MPI/XDMF preflight..."
set +e
NPROCS="${MAX_RANKS}" \
EXPECTED_NODES="${EXPECTED_NODES}" \
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS}" \
OUTPUT_DIR="${OUT_ROOT}/preflight_probe" \
PREFLIGHT_ONLY=1 \
PREFLIGHT_REPORT_FILE="${PREFLIGHT_REPORT}" \
SKIP_PREFLIGHT=0 \
bash "${SCRIPT_DIR}/run_linux.sh" 2>&1 | tee "${PREFLIGHT_LOG}"
PREFLIGHT_PIPESTATUS=("${PIPESTATUS[@]}")
PREFLIGHT_LAUNCHER_RC=${PREFLIGHT_PIPESTATUS[0]}
PREFLIGHT_TEE_RC=${PREFLIGHT_PIPESTATUS[1]}
set -e
if [[ "${PREFLIGHT_LAUNCHER_RC}" -ne 0 ]]; then
  echo "Preflight failed; simulation cases were not started." >&2
  exit "${PREFLIGHT_LAUNCHER_RC}"
fi
if [[ "${PREFLIGHT_TEE_RC}" -ne 0 ]]; then
  echo "Preflight logging failed; simulation cases were not started." >&2
  exit "${PREFLIGHT_TEE_RC}"
fi

RUN_ORDER=0
FAILED_RUNS=0
STOP_REQUESTED=0

input_sha256() {
  "${PYTHON_BIN}" - "$1" <<'PY'
from hashlib import sha256
from pathlib import Path
import sys
print(sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
}

run_case() {
  local case_name="$1"
  local category="$2"
  local ranks="$3"
  local repeat="$4"
  local input_path="$5"
  local padded slug output_path log_path start_utc end_utc
  local start_epoch end_epoch wall_seconds rc digest launcher_rc tee_rc
  local -a pipeline_statuses

  RUN_ORDER=$((RUN_ORDER + 1))
  printf -v padded '%02d' "${RUN_ORDER}"
  slug="${padded}_${case_name}_${ranks}r_rep${repeat}"
  output_path="${OUT_ROOT}/runs/${slug}"
  log_path="${OUT_ROOT}/logs/${slug}.log"
  digest="$(input_sha256 "${input_path}")"
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  start_epoch="$(date +%s)"

  echo
  echo "[${slug}] ${category}: ${ranks} MPI rank(s)"
  set +e
  NPROCS="${ranks}" \
  EXPECTED_NODES="${EXPECTED_NODES}" \
  MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS}" \
  INPUT_FILE="${input_path}" \
  OUTPUT_DIR="${output_path}" \
  SKIP_PREFLIGHT=1 \
  bash "${SCRIPT_DIR}/run_linux.sh" 2>&1 | tee "${log_path}"
  pipeline_statuses=("${PIPESTATUS[@]}")
  launcher_rc=${pipeline_statuses[0]}
  tee_rc=${pipeline_statuses[1]}
  if [[ "${launcher_rc}" -ne 0 ]]; then
    rc=${launcher_rc}
  else
    rc=${tee_rc}
  fi
  set -e

  end_epoch="$(date +%s)"
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  wall_seconds=$((end_epoch - start_epoch))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${RUN_ORDER}" "${case_name}" "${category}" "${ranks}" "${repeat}" \
    "${digest}" "${input_path}" "${output_path}" "${log_path}" \
    "${start_utc}" "${end_utc}" "${wall_seconds}" "${rc}" >> "${LEDGER}"

  if [[ "${rc}" -ne 0 ]]; then
    FAILED_RUNS=$((FAILED_RUNS + 1))
    if [[ "${launcher_rc}" -ne 0 ]]; then
      echo "[${slug}] FAILED with launcher status ${launcher_rc}." >&2
    else
      echo "[${slug}] FAILED with logging status ${tee_rc}." >&2
    fi
    if [[ "${FAIL_FAST}" == "1" ]]; then
      STOP_REQUESTED=1
    fi
  else
    echo "[${slug}] completed."
  fi
}

summarize_partial_on_interrupt() {
  trap - INT TERM
  echo "Suite interrupted; writing a partial report." >&2
  "${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_xeon16_suite.py" \
    "${OUT_ROOT}" --allow-incomplete || true
  exit 130
}
trap summarize_partial_on_interrupt INT TERM

run_case smoke smoke "${MAX_RANKS}" 1 \
  "${INPUT_ROOT}/xeon16_smoke.in"

if [[ "${STOP_REQUESTED}" -eq 0 ]] \
  && [[ "${SUITE_MODE}" == "validation" || "${SUITE_MODE}" == "all" ]]; then
  run_case mode_i functional "${MAX_RANKS}" 1 \
    "${INPUT_ROOT}/xeon16_mode_i.in"
  [[ "${STOP_REQUESTED}" -eq 1 ]] || run_case mixed_mode functional \
    "${MAX_RANKS}" 1 "${INPUT_ROOT}/xeon16_mixed_mode.in"
  [[ "${STOP_REQUESTED}" -eq 1 ]] || run_case mixed_symmetric functional \
    "${MAX_RANKS}" 1 "${INPUT_ROOT}/xeon16_mixed_symmetric.in"
  [[ "${STOP_REQUESTED}" -eq 1 ]] || run_case graded_linear functional \
    "${MAX_RANKS}" 1 "${INPUT_ROOT}/xeon16_graded_linear.in"
  [[ "${STOP_REQUESTED}" -eq 1 ]] || run_case graded_inclusion functional \
    "${MAX_RANKS}" 1 "${INPUT_ROOT}/xeon16_graded_inclusion.in"
fi

if [[ "${STOP_REQUESTED}" -eq 0 ]] \
  && [[ "${SUITE_MODE}" == "scaling" || "${SUITE_MODE}" == "all" ]]; then
  run_case scaling_warmup warmup "${MAX_RANKS}" 0 \
    "${INPUT_ROOT}/xeon16_scaling.in"
  for ((repeat = 1; repeat <= SCALING_REPEATS; repeat++)); do
    for ranks in "${RANKS[@]}"; do
      if [[ "${STOP_REQUESTED}" -eq 1 ]]; then
        break 2
      fi
      run_case scaling scaling "${ranks}" "${repeat}" \
        "${INPUT_ROOT}/xeon16_scaling.in"
    done
  done
fi

trap - INT TERM
echo
echo "Aggregating correctness and scaling results..."
set +e
"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_xeon16_suite.py" "${OUT_ROOT}"
SUMMARY_RC=$?
set -e

echo "Suite directory: ${OUT_ROOT}"
echo "Human report  : ${OUT_ROOT}/suite_report.md"
if [[ "${FAILED_RUNS}" -ne 0 || "${SUMMARY_RC}" -ne 0 ]]; then
  echo "Xeon16 suite FAILED. Inspect suite_report.md and logs/." >&2
  exit 1
fi
echo "Xeon16 suite PASSED."
