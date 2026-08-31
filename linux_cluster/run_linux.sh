#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: bash linux_cluster/run_linux.sh [--ranks N|auto] [--mpi CHOICE]

Options:
  --ranks N|auto   MPI ranks. "auto" uses allocation-visible physical cores.
  --mpi CHOICE     auto, conda, system, or an mpiexec/mpirun/srun path.
  --show-options   Detect cores and MPI, report the selection, then stop.
  -h, --help       Show this help message.

The legacy NPROCS environment variable remains supported.  Rank precedence is
--ranks, NPROCS, SLURM_NTASKS, then PBS_NP/PBS_NODEFILE.

MPI_HOME may name a nonstandard MPI prefix or bin directory.  Automatic mode
tests that MPI with the active Python/PETSc stack and falls back to Conda MPI
when needed.  Use RUNTIME_ENV=active for a complete site-provided stack and
STRICT_RESOURCES=1 to reject rather than reduce an oversized rank request.
EOF
}

CLI_NPROCS=""
CLI_NPROCS_SET=0
CLI_MPI_CHOICE=""
CLI_MPI_CHOICE_SET=0
SHOW_OPTIONS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ranks)
      if [[ $# -lt 2 ]]; then
        echo "--ranks requires a positive integer argument." >&2
        exit 2
      fi
      CLI_NPROCS="$2"
      CLI_NPROCS_SET=1
      shift 2
      ;;
    --ranks=*)
      CLI_NPROCS="${1#*=}"
      CLI_NPROCS_SET=1
      shift
      ;;
    --mpi)
      if [[ $# -lt 2 ]]; then
        echo "--mpi requires auto, conda, system, or a launcher path." >&2
        exit 2
      fi
      CLI_MPI_CHOICE="$2"
      CLI_MPI_CHOICE_SET=1
      shift 2
      ;;
    --mpi=*)
      CLI_MPI_CHOICE="${1#*=}"
      CLI_MPI_CHOICE_SET=1
      shift
      ;;
    --show-options)
      SHOW_OPTIONS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

ENV_NAME="${ENV_NAME:-phasefield-fenicsx-linux}"
if [[ "${CLI_NPROCS_SET}" == "1" ]]; then
  NPROCS="${CLI_NPROCS}"
else
  NPROCS="${NPROCS:-${SLURM_NTASKS:-${PBS_NP:-auto}}}"
fi
if [[ "${CLI_MPI_CHOICE_SET}" == "1" ]]; then
  MPI_PREFERENCE="${CLI_MPI_CHOICE}"
else
  MPI_PREFERENCE="${MPI_PREFERENCE:-${MPI_LAUNCHER:-auto}}"
fi
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-}"
RUNTIME_ENV="${RUNTIME_ENV:-conda}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MPI_HOME="${MPI_HOME:-}"
ALLOW_EXTERNAL_MPI="${ALLOW_EXTERNAL_MPI:-0}"
FORCE_INCOMPATIBLE_MPI="${FORCE_INCOMPATIBLE_MPI:-0}"
STRICT_RESOURCES="${STRICT_RESOURCES:-0}"
MPI_PROBE_TIMEOUT_SECONDS="${MPI_PROBE_TIMEOUT_SECONDS:-30}"
MPI_PROBE_VERBOSE="${MPI_PROBE_VERBOSE:-0}"
INPUT_FILE="${INPUT_FILE:-${SCRIPT_DIR}/inputs/notched_tension_cluster.in}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PREFLIGHT_REPORT_FILE="${PREFLIGHT_REPORT_FILE:-}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"

# Capture the user's current MPI before Conda activation changes PATH. MPI_HOME
# and an explicit --mpi path take precedence over these discovered commands.
PREACTIVATION_MPIEXEC="$(command -v mpiexec 2>/dev/null || true)"
PREACTIVATION_MPIRUN="$(command -v mpirun 2>/dev/null || true)"
PREACTIVATION_SRUN="$(command -v srun 2>/dev/null || true)"

for rank_context_name in SLURM_PROCID PMI_RANK PMIX_RANK \
  OMPI_COMM_WORLD_RANK MV2_COMM_WORLD_RANK; do
  if [[ -n "${!rank_context_name:-}" ]]; then
    echo "run_linux.sh is already running inside an MPI/srun rank (${rank_context_name}=${!rank_context_name})." >&2
    echo "Invoke it once from the batch shell; it creates the MPI launch itself." >&2
    exit 2
  fi
done

if [[ ( -z "${NPROCS}" || "${NPROCS}" == "auto" ) && -r "${PBS_NODEFILE:-}" ]]; then
  NPROCS="$(wc -l < "${PBS_NODEFILE}" | tr -d '[:space:]')"
fi
if [[ -z "${EXPECTED_NODES:-}" && -n "${SLURM_NNODES:-}" ]]; then
  export EXPECTED_NODES="${SLURM_NNODES}"
elif [[ -z "${EXPECTED_NODES:-}" && -r "${PBS_NODEFILE:-}" ]]; then
  export EXPECTED_NODES="$(sort -u "${PBS_NODEFILE}" | wc -l | tr -d '[:space:]')"
fi

if [[ "${NPROCS}" != "auto" ]] && ! [[ "${NPROCS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MPI ranks must be 'auto' or a positive integer; got '${NPROCS}'." >&2
  exit 2
fi
for toggle_name in SKIP_PREFLIGHT PREFLIGHT_ONLY ALLOW_EXISTING_OUTPUT \
  ALLOW_EXTERNAL_MPI FORCE_INCOMPATIBLE_MPI STRICT_RESOURCES MPI_PROBE_VERBOSE; do
  toggle_value="${!toggle_name}"
  if [[ "${toggle_value}" != "0" && "${toggle_value}" != "1" ]]; then
    echo "${toggle_name} must be 0 or 1; got '${toggle_value}'." >&2
    exit 2
  fi
done
if ! [[ "${MPI_PROBE_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MPI_PROBE_TIMEOUT_SECONDS must be a positive integer." >&2
  exit 2
fi
if [[ -n "${EXPECTED_NODES:-}" ]] \
  && ! [[ "${EXPECTED_NODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EXPECTED_NODES must be a positive integer when set." >&2
  exit 2
fi
if [[ "${PREFLIGHT_ONLY}" == "1" && "${SKIP_PREFLIGHT}" == "1" ]]; then
  echo "PREFLIGHT_ONLY=1 is incompatible with SKIP_PREFLIGHT=1." >&2
  exit 2
fi

case "${RUNTIME_ENV}" in
  conda)
    if [[ -n "${CONDA_BASE:-}" ]]; then
      if [[ ! -r "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        echo "CONDA_BASE does not contain a readable conda.sh: ${CONDA_BASE}" >&2
        exit 2
      fi
      source "${CONDA_BASE}/etc/profile.d/conda.sh"
    elif command -v conda >/dev/null 2>&1; then
      eval "$(conda shell.bash hook)"
    else
      echo "Conda was not found. Load Miniconda, set CONDA_BASE, or use RUNTIME_ENV=active." >&2
      exit 2
    fi
    conda activate "${ENV_NAME}"
    hash -r
    if [[ -z "${CONDA_PREFIX:-}" ]]; then
      echo "Conda activation did not set CONDA_PREFIX." >&2
      exit 2
    fi
    RUNTIME_DESCRIPTION="Conda environment ${ENV_NAME}"
    ;;
  active)
    hash -r
    RUNTIME_DESCRIPTION="active site/module environment"
    ;;
  *)
    echo "RUNTIME_ENV must be 'conda' or 'active'; got '${RUNTIME_ENV}'." >&2
    exit 2
    ;;
esac

PYTHON_PATH="$(command -v "${PYTHON_BIN}" || true)"
if [[ -z "${PYTHON_PATH}" ]]; then
  echo "Python executable was not found after runtime setup: ${PYTHON_BIN}" >&2
  exit 2
fi

if ! CAPACITY_OUTPUT="$(
  PHASEFIELD_PROJECT_ROOT="${PROJECT_ROOT}" "${PYTHON_PATH}" -c '
import os
import sys
sys.path.insert(0, os.environ["PHASEFIELD_PROJECT_ROOT"])
try:
    from linux_cluster.xeon16_hardware import detect_hardware
    report = detect_hardware(allow_unknown_topology=True)
    scheduler_values = list(report["scheduler_capacities"].values())
    print(
        report["physical_core_capacity"],
        min(scheduler_values) if scheduler_values else 0,
        "complete" if report["physical_topology_complete"] else "estimated",
        sep="\t",
    )
except Exception:
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    print(affinity, 0, "logical-fallback", sep="\t")
' 2>/dev/null
)"; then
  echo "Unable to detect allocation-visible CPU capacity." >&2
  exit 2
fi
IFS=$'\t' read -r PHYSICAL_CORE_CAPACITY SCHEDULER_RANK_CAPACITY TOPOLOGY_STATUS \
  <<< "${CAPACITY_OUTPUT}"
if ! [[ "${PHYSICAL_CORE_CAPACITY}" =~ ^[1-9][0-9]*$ ]] \
  || ! [[ "${SCHEDULER_RANK_CAPACITY}" =~ ^[0-9]+$ ]]; then
  echo "Detected invalid CPU capacity data: ${CAPACITY_OUTPUT}." >&2
  exit 2
fi
RESOURCE_NODE_COUNT="${EXPECTED_NODES:-1}"
ESTIMATED_PHYSICAL_CAPACITY=$((PHYSICAL_CORE_CAPACITY * RESOURCE_NODE_COUNT))
if (( SCHEDULER_RANK_CAPACITY > 0 && RESOURCE_NODE_COUNT > 1 )); then
  if (( SCHEDULER_RANK_CAPACITY < ESTIMATED_PHYSICAL_CAPACITY )); then
    EFFECTIVE_RANK_CAPACITY="${SCHEDULER_RANK_CAPACITY}"
  else
    EFFECTIVE_RANK_CAPACITY="${ESTIMATED_PHYSICAL_CAPACITY}"
  fi
  CAPACITY_SOURCE="estimated physical cores and scheduler slots across ${RESOURCE_NODE_COUNT} allocated nodes"
elif (( SCHEDULER_RANK_CAPACITY > 0 )); then
  if (( SCHEDULER_RANK_CAPACITY < PHYSICAL_CORE_CAPACITY )); then
    EFFECTIVE_RANK_CAPACITY="${SCHEDULER_RANK_CAPACITY}"
  else
    EFFECTIVE_RANK_CAPACITY="${PHYSICAL_CORE_CAPACITY}"
  fi
  CAPACITY_SOURCE="physical cores and scheduler slots on one allocated node"
else
  EFFECTIVE_RANK_CAPACITY="${ESTIMATED_PHYSICAL_CAPACITY}"
  CAPACITY_SOURCE="affinity-visible physical cores across ${RESOURCE_NODE_COUNT} node(s)"
fi
if [[ "${NPROCS}" == "auto" ]]; then
  NPROCS="${EFFECTIVE_RANK_CAPACITY}"
elif (( NPROCS > EFFECTIVE_RANK_CAPACITY )); then
  if [[ "${STRICT_RESOURCES}" == "1" ]]; then
    echo "Requested ${NPROCS} ranks but only ${EFFECTIVE_RANK_CAPACITY} allocation-visible physical core(s)/slot(s) are available." >&2
    exit 2
  fi
  echo "Requested ${NPROCS} ranks; using detected capacity ${EFFECTIVE_RANK_CAPACITY}." >&2
  NPROCS="${EFFECTIVE_RANK_CAPACITY}"
fi
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/linux_petsc_${NPROCS}r_$(date +%Y%m%d_%H%M%S)_${BASHPID}}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -n "${MPI_EXTRA_ARGS}" ]]; then
  if [[ "${MPI_EXTRA_ARGS}" == *\"* || "${MPI_EXTRA_ARGS}" == *\'* ]]; then
    echo "MPI_EXTRA_ARGS accepts whitespace-delimited arguments without shell quotes." >&2
    exit 2
  fi
  read -r -a EXTRA_ARRAY <<< "${MPI_EXTRA_ARGS}"
else
  EXTRA_ARRAY=()
fi

resolve_executable() {
  local requested="$1"
  local resolved=""
  if [[ "${requested}" == */* ]]; then
    [[ -x "${requested}" ]] || return 1
    resolved="${requested}"
  else
    resolved="$(command -v "${requested}" 2>/dev/null || true)"
    [[ -n "${resolved}" ]] || return 1
  fi
  printf '%s/%s\n' \
    "$(cd -- "$(dirname -- "${resolved}")" && pwd -P)" \
    "$(basename -- "${resolved}")"
}

declare -a BUILT_LAUNCH
build_launch() {
  local launcher_path="$1"
  local ranks="$2"
  local launcher_name
  launcher_name="$(basename -- "${launcher_path}")"
  case "${launcher_name}" in
    mpiexec|mpiexec.*|mpiexec-*|mpirun|mpirun.*|mpirun-*|orterun)
      BUILT_LAUNCH=("${launcher_path}" -n "${ranks}")
      ;;
    srun|srun.*|srun-*)
      BUILT_LAUNCH=("${launcher_path}" --ntasks="${ranks}" --cpus-per-task=1)
      ;;
    *)
      return 2
      ;;
  esac
  BUILT_LAUNCH+=("${EXTRA_ARRAY[@]}")
}

MPI_HOME_CANDIDATE=""
if [[ -n "${MPI_HOME}" ]]; then
  for mpi_home_candidate in \
    "${MPI_HOME}/bin/mpiexec" "${MPI_HOME}/mpiexec" \
    "${MPI_HOME}/bin/mpiexec.hydra" "${MPI_HOME}/mpiexec.hydra" \
    "${MPI_HOME}/bin/mpiexec.mpich" "${MPI_HOME}/mpiexec.mpich" \
    "${MPI_HOME}/bin/mpirun" "${MPI_HOME}/mpirun" \
    "${MPI_HOME}/bin/mpirun.mpich" "${MPI_HOME}/mpirun.mpich"; do
    if MPI_HOME_CANDIDATE="$(resolve_executable "${mpi_home_candidate}" 2>/dev/null)"; then
      break
    fi
    MPI_HOME_CANDIDATE=""
  done
fi
PATH_MPI_CANDIDATE=""
if [[ -n "${PREACTIVATION_MPIEXEC}" ]]; then
  PATH_MPI_CANDIDATE="$(resolve_executable "${PREACTIVATION_MPIEXEC}" 2>/dev/null || true)"
fi
if [[ -z "${PATH_MPI_CANDIDATE}" && -n "${PREACTIVATION_MPIRUN}" ]]; then
  PATH_MPI_CANDIDATE="$(resolve_executable "${PREACTIVATION_MPIRUN}" 2>/dev/null || true)"
fi
SYSTEM_MPI_CANDIDATE="${MPI_HOME_CANDIDATE:-${PATH_MPI_CANDIDATE}}"
SRUN_MPI_CANDIDATE=""
if [[ -n "${SLURM_JOB_ID:-}" && -n "${PREACTIVATION_SRUN}" ]]; then
  SRUN_MPI_CANDIDATE="$(resolve_executable "${PREACTIVATION_SRUN}" 2>/dev/null || true)"
fi

CONDA_MPI_CANDIDATE=""
if [[ "${RUNTIME_ENV}" == "conda" ]]; then
  for conda_mpi_candidate in \
    "${CONDA_PREFIX}/bin/mpiexec" "${CONDA_PREFIX}/bin/mpirun"; do
    if CONDA_MPI_CANDIDATE="$(resolve_executable "${conda_mpi_candidate}" 2>/dev/null)"; then
      break
    fi
    CONDA_MPI_CANDIDATE=""
  done
fi

declare -a MPI_CANDIDATES MPI_CANDIDATE_REASONS
MPI_CANDIDATES=()
MPI_CANDIDATE_REASONS=()
append_mpi_candidate() {
  local candidate="$1"
  local reason="$2"
  local existing
  [[ -n "${candidate}" ]] || return 0
  for existing in "${MPI_CANDIDATES[@]}"; do
    [[ "${existing}" == "${candidate}" ]] && return 0
  done
  MPI_CANDIDATES+=("${candidate}")
  MPI_CANDIDATE_REASONS+=("${reason}")
}

case "${MPI_PREFERENCE}" in
  auto)
    append_mpi_candidate "${MPI_HOME_CANDIDATE}" "MPI_HOME candidate"
    append_mpi_candidate "${SRUN_MPI_CANDIDATE}" "Slurm scheduler candidate"
    append_mpi_candidate "${PATH_MPI_CANDIDATE}" "pre-activation PATH candidate"
    append_mpi_candidate "${CONDA_MPI_CANDIDATE}" "Conda MPI candidate"
    ;;
  conda)
    append_mpi_candidate "${CONDA_MPI_CANDIDATE}" "user preference: Conda MPI"
    ;;
  system)
    append_mpi_candidate "${MPI_HOME_CANDIDATE}" "user preference: MPI_HOME"
    append_mpi_candidate "${SRUN_MPI_CANDIDATE}" "user preference: Slurm scheduler"
    append_mpi_candidate "${PATH_MPI_CANDIDATE}" "user preference: pre-activation PATH"
    append_mpi_candidate "${CONDA_MPI_CANDIDATE}" "automatic Conda fallback"
    ;;
  *)
    EXPLICIT_MPI_CANDIDATE="$(resolve_executable "${MPI_PREFERENCE}" 2>/dev/null || true)"
    append_mpi_candidate "${EXPLICIT_MPI_CANDIDATE}" "user-selected launcher"
    append_mpi_candidate "${CONDA_MPI_CANDIDATE}" "automatic Conda fallback"
    ;;
esac
if (( ${#MPI_CANDIDATES[@]} == 0 )); then
  echo "No usable MPI launcher was found for preference '${MPI_PREFERENCE}'." >&2
  echo "Put MPI on PATH, set MPI_HOME, pass --mpi /path/to/mpiexec, or choose --mpi conda." >&2
  exit 2
fi

probe_mpi_candidate() {
  local launcher_path="$1"
  local probe_ranks=1
  local probe_output=""
  local probe_status=0
  if (( NPROCS >= 2 && EFFECTIVE_RANK_CAPACITY >= 2 )); then
    probe_ranks=2
  fi
  build_launch "${launcher_path}" "${probe_ranks}" || return 2
  local -a probe_launch=("${BUILT_LAUNCH[@]}")
  set +e
  if command -v timeout >/dev/null 2>&1; then
    probe_output="$(timeout "${MPI_PROBE_TIMEOUT_SECONDS}s" \
      "${probe_launch[@]}" "${PYTHON_PATH}" -u \
      "${SCRIPT_DIR}/mpi_compatibility_probe.py" \
      --expected-ranks "${probe_ranks}" 2>&1)"
    probe_status=$?
  else
    probe_output="$("${probe_launch[@]}" "${PYTHON_PATH}" -u \
      "${SCRIPT_DIR}/mpi_compatibility_probe.py" \
      --expected-ranks "${probe_ranks}" 2>&1)"
    probe_status=$?
  fi
  set -e
  if [[ "${MPI_PROBE_VERBOSE}" == "1" || "${probe_status}" == "0" ]]; then
    printf '%s\n' "${probe_output}"
  fi
  [[ "${probe_status}" == "0" && "${probe_output}" == *MPI_COMPATIBILITY_OK* ]]
}

SELECTED_INDEX=-1
SELECTED_MPI=""
SELECTION_REASON=""
for ((candidate_index=0; candidate_index<${#MPI_CANDIDATES[@]}; candidate_index++)); do
  candidate_path="${MPI_CANDIDATES[${candidate_index}]}"
  candidate_reason="${MPI_CANDIDATE_REASONS[${candidate_index}]}"
  echo "Testing MPI compatibility: ${candidate_path} (${candidate_reason})"
  if probe_mpi_candidate "${candidate_path}"; then
    SELECTED_INDEX="${candidate_index}"
    SELECTED_MPI="${candidate_path}"
    SELECTION_REASON="${candidate_reason}; compatibility probe passed"
    break
  fi
  if (( candidate_index == 0 )) \
    && [[ "${candidate_path}" != "${CONDA_MPI_CANDIDATE}" \
      && "${FORCE_INCOMPATIBLE_MPI}" == "1" ]]; then
    SELECTED_INDEX="${candidate_index}"
    SELECTED_MPI="${candidate_path}"
    SELECTION_REASON="${candidate_reason}; failed probe deliberately forced"
    break
  fi
  echo "MPI candidate did not pass; trying the next available candidate."
done
if (( SELECTED_INDEX < 0 )); then
  echo "No MPI candidate passed compatibility testing." >&2
  echo "Set MPI_PROBE_VERBOSE=1 for diagnostics or use a coherent site stack." >&2
  exit 2
fi
SELECTED_IS_CONDA=0
if [[ -n "${CONDA_MPI_CANDIDATE}" && "${SELECTED_MPI}" == "${CONDA_MPI_CANDIDATE}" ]]; then
  SELECTED_IS_CONDA=1
fi
if ! build_launch "${SELECTED_MPI}" "${NPROCS}"; then
  echo "MPI choice must resolve to mpiexec, mpirun, or srun: ${SELECTED_MPI}" >&2
  exit 2
fi
LAUNCH=("${BUILT_LAUNCH[@]}")

echo "Runtime     : ${RUNTIME_DESCRIPTION}"
echo "Python      : ${PYTHON_PATH}"
echo "Physical CPU: ${PHYSICAL_CORE_CAPACITY} per visible node (${TOPOLOGY_STATUS} topology)"
echo "Rank capacity: ${EFFECTIVE_RANK_CAPACITY} (${CAPACITY_SOURCE})"
echo "MPI_HOME MPI: ${MPI_HOME_CANDIDATE:-not specified/found}"
echo "PATH MPI    : ${PATH_MPI_CANDIDATE:-not found before runtime setup}"
echo "Slurm MPI   : ${SRUN_MPI_CANDIDATE:-not available}"
echo "Conda MPI   : ${CONDA_MPI_CANDIDATE:-not available}"
echo "MPI ranks   : ${NPROCS}"
echo "MPI nodes   : ${EXPECTED_NODES:-unchecked}"
echo "Launcher    : ${LAUNCH[*]}"
echo "MPI choice  : ${SELECTION_REASON}"
echo "Input       : ${INPUT_FILE}"
echo "Output      : ${OUTPUT_DIR}"

if [[ "${SHOW_OPTIONS}" == "1" ]]; then
  echo "Detection-only request completed; no simulation was started."
  exit 0
fi
if [[ ! -r "${INPUT_FILE}" ]]; then
  echo "Input file is not readable: ${INPUT_FILE}" >&2
  exit 2
fi
if [[ -d "${OUTPUT_DIR}" ]] && [[ "${ALLOW_EXISTING_OUTPUT}" != "1" ]] \
  && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Output directory is not empty: ${OUTPUT_DIR}" >&2
  echo "Choose a new OUTPUT_DIR or set ALLOW_EXISTING_OUTPUT=1." >&2
  exit 2
fi

if [[ "${SKIP_PREFLIGHT}" != "1" ]]; then
  PREFLIGHT_ARGS=(
    --expected-ranks "${NPROCS}"
    --check-xdmf "${OUTPUT_DIR}.preflight"
  )
  if [[ -n "${EXPECTED_NODES:-}" ]]; then
    PREFLIGHT_ARGS+=(--expected-nodes "${EXPECTED_NODES}")
  fi
  if [[ -n "${PREFLIGHT_REPORT_FILE}" ]]; then
    PREFLIGHT_ARGS+=(--report-json "${PREFLIGHT_REPORT_FILE}")
  fi
  if [[ "${NPROCS}" == "1" ]]; then
    PREFLIGHT_ARGS+=(--allow-one-rank)
  fi
  if "${LAUNCH[@]}" "${PYTHON_PATH}" -u \
    "${SCRIPT_DIR}/check_environment.py" "${PREFLIGHT_ARGS[@]}"; then
    PREFLIGHT_STATUS=0
  else
    PREFLIGHT_STATUS=$?
  fi
  if (( PREFLIGHT_STATUS != 0 )) \
    && [[ "${FORCE_INCOMPATIBLE_MPI}" != "1" ]]; then
    for ((fallback_index=SELECTED_INDEX+1; \
      fallback_index<${#MPI_CANDIDATES[@]}; fallback_index++)); do
      fallback_path="${MPI_CANDIDATES[${fallback_index}]}"
      fallback_reason="${MPI_CANDIDATE_REASONS[${fallback_index}]}"
      echo "The selected MPI failed the full preflight; testing fallback ${fallback_path}."
      if ! probe_mpi_candidate "${fallback_path}"; then
        echo "Fallback MPI did not pass the compatibility probe; continuing."
        continue
      fi
      SELECTED_INDEX="${fallback_index}"
      SELECTED_MPI="${fallback_path}"
      SELECTION_REASON="${fallback_reason}; selected after failed full preflight"
      build_launch "${SELECTED_MPI}" "${NPROCS}"
      LAUNCH=("${BUILT_LAUNCH[@]}")
      echo "Fallback launcher: ${LAUNCH[*]}"
      if "${LAUNCH[@]}" "${PYTHON_PATH}" -u \
        "${SCRIPT_DIR}/check_environment.py" "${PREFLIGHT_ARGS[@]}"; then
        PREFLIGHT_STATUS=0
        break
      else
        PREFLIGHT_STATUS=$?
      fi
    done
  fi
  if (( PREFLIGHT_STATUS != 0 )); then
    exit "${PREFLIGHT_STATUS}"
  fi
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "Preflight-only request completed; simulation was not started."
  exit 0
fi

"${LAUNCH[@]}" "${PYTHON_PATH}" -u "${SCRIPT_DIR}/phasefield_crack_petsc.py" \
  -in "${INPUT_FILE}" \
  --output-dir "${OUTPUT_DIR}"
