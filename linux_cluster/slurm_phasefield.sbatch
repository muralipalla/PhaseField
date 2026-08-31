#!/usr/bin/env bash
#SBATCH --job-name=phasefield-petsc
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=phasefield-%j.out

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "${PHASEFIELD_PROJECT_ROOT:-${SUBMIT_DIR}}/linux_cluster/run_linux.sh" ]]; then
  PROJECT_ROOT="${PHASEFIELD_PROJECT_ROOT:-${SUBMIT_DIR}}"
elif [[ -f "${SUBMIT_DIR}/run_linux.sh" ]]; then
  SCRIPT_DIR="${SUBMIT_DIR}"
  PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
else
  echo "Submit from the project root or set PHASEFIELD_PROJECT_ROOT." >&2
  exit 2
fi
SCRIPT_DIR="${SCRIPT_DIR:-${PROJECT_ROOT}/linux_cluster}"

export NPROCS="${SLURM_NTASKS}"
export EXPECTED_NODES="${SLURM_NNODES}"
export MPI_LAUNCHER="${MPI_LAUNCHER:-mpiexec}"
export INPUT_FILE="${INPUT_FILE:-${SCRIPT_DIR}/inputs/notched_tension_cluster.in}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/slurm_${SLURM_JOB_ID}_${NPROCS}r}"

# If your site requires environment modules, add its documented module-load
# commands here. Do not mix a system MPI launcher with Conda MPICH unless your
# administrators confirm ABI/PMI compatibility.
bash "${SCRIPT_DIR}/run_linux.sh"
