#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="${ENV_NAME:-phasefield-fenicsx-linux}"
NPROCS="${NPROCS:-16}"
MPI_LAUNCHER="${MPI_LAUNCHER:-mpiexec}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-}"
INPUT_FILE="${INPUT_FILE:-${SCRIPT_DIR}/inputs/notched_tension_cluster.in}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/linux_petsc_${NPROCS}r_$(date +%Y%m%d_%H%M%S)_${BASHPID}}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
PREFLIGHT_REPORT_FILE="${PREFLIGHT_REPORT_FILE:-}"
ALLOW_EXISTING_OUTPUT="${ALLOW_EXISTING_OUTPUT:-0}"

if [[ -z "${EXPECTED_NODES:-}" && -n "${SLURM_NNODES:-}" ]]; then
  export EXPECTED_NODES="${SLURM_NNODES}"
elif [[ -z "${EXPECTED_NODES:-}" && -r "${PBS_NODEFILE:-}" ]]; then
  export EXPECTED_NODES="$(sort -u "${PBS_NODEFILE}" | wc -l)"
fi

if ! [[ "${NPROCS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROCS must be a positive integer; got '${NPROCS}'." >&2
  exit 2
fi
if [[ "${PREFLIGHT_ONLY}" == "1" && "${SKIP_PREFLIGHT}" == "1" ]]; then
  echo "PREFLIGHT_ONLY=1 is incompatible with SKIP_PREFLIGHT=1." >&2
  exit 2
fi
if [[ ! -r "${INPUT_FILE}" ]]; then
  echo "Input file is not readable: ${INPUT_FILE}" >&2
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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -d "${OUTPUT_DIR}" ]] && [[ "${ALLOW_EXISTING_OUTPUT}" != "1" ]] \
  && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Output directory is not empty: ${OUTPUT_DIR}" >&2
  echo "Choose a new OUTPUT_DIR or set ALLOW_EXISTING_OUTPUT=1." >&2
  exit 2
fi

declare -a LAUNCH
case "${MPI_LAUNCHER}" in
  mpiexec|mpirun)
    LAUNCHER_PATH="$(command -v "${MPI_LAUNCHER}" || true)"
    if [[ -z "${LAUNCHER_PATH}" ]]; then
      echo "MPI launcher was not found after Conda activation: ${MPI_LAUNCHER}" >&2
      exit 2
    fi
    if [[ "${LAUNCHER_PATH}" != "${CONDA_PREFIX}/"* ]] \
      && [[ "${ALLOW_EXTERNAL_MPI:-0}" != "1" ]]; then
      echo "Refusing external MPI launcher ${LAUNCHER_PATH}." >&2
      echo "Use the environment launcher or set ALLOW_EXTERNAL_MPI=1 only after an ABI check." >&2
      exit 2
    fi
    LAUNCH=("${MPI_LAUNCHER}" -n "${NPROCS}")
    ;;
  srun)
    LAUNCH=(srun --ntasks="${NPROCS}" --cpus-per-task=1)
    ;;
  *)
    echo "MPI_LAUNCHER must be mpiexec, mpirun, or srun." >&2
    exit 2
    ;;
esac
if [[ -n "${MPI_EXTRA_ARGS}" ]]; then
  read -r -a EXTRA_ARRAY <<< "${MPI_EXTRA_ARGS}"
  LAUNCH+=("${EXTRA_ARRAY[@]}")
fi

echo "Environment : ${ENV_NAME}"
echo "MPI ranks   : ${NPROCS}"
echo "MPI nodes   : ${EXPECTED_NODES:-unchecked}"
echo "Launcher    : ${LAUNCH[*]}"
echo "Input       : ${INPUT_FILE}"
echo "Output      : ${OUTPUT_DIR}"

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
  "${LAUNCH[@]}" python -u "${SCRIPT_DIR}/check_environment.py" \
    "${PREFLIGHT_ARGS[@]}"
fi

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "Preflight-only request completed; simulation was not started."
  exit 0
fi

"${LAUNCH[@]}" python -u "${SCRIPT_DIR}/phasefield_crack_petsc.py" \
  -in "${INPUT_FILE}" \
  --output-dir "${OUTPUT_DIR}"
