#!/usr/bin/env bash
#SBATCH --job-name=phasefield-petsc
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --output=phasefield-%j.out

# The two-rank request is a safe template default. Override it at submission,
# for example: sbatch --ntasks=4 linux_cluster/slurm_phasefield.sbatch

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
export MPI_PREFERENCE="${MPI_PREFERENCE:-${MPI_LAUNCHER:-auto}}"
export INPUT_FILE="${INPUT_FILE:-${SCRIPT_DIR}/inputs/notched_tension_cluster.in}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/slurm_${SLURM_JOB_ID}_${NPROCS}r}"

# If your site requires environment modules, add its documented module-load
# commands here. The direct launcher compatibility-tests a discovered site MPI
# and automatically falls back to the Conda launcher when needed.
bash "${SCRIPT_DIR}/run_linux.sh"
