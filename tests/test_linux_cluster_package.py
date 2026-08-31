from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

from linux_cluster import check_environment
from phasefield_input import parse_input_file
from linux_cluster.petsc_config import PetscSimulationConfig


ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "linux_cluster"


def _keywords(path: Path) -> set[str]:
    result: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        content = line.split("#", 1)[0].strip()
        if content:
            result.add(content.split()[0].casefold())
    return result


def test_cluster_inputs_parse_and_validate() -> None:
    for name in (
        "cluster_smoke.in",
        "notched_tension_cluster.in",
        "mixed_mode.in",
        "graded_linear_x.in",
        "graded_file.in",
    ):
        config = parse_input_file(
            LINUX / "inputs" / name, PetscSimulationConfig()
        )
        assert isinstance(config, PetscSimulationConfig)
        config.validate()


def test_production_input_lists_every_configuration_field() -> None:
    expected = {field.name for field in fields(PetscSimulationConfig)}
    actual = _keywords(LINUX / "inputs" / "notched_tension_cluster.in")
    assert actual == expected


def test_production_mesh_resolves_length_scale() -> None:
    config = parse_input_file(
        LINUX / "inputs" / "notched_tension_cluster.in",
        PetscSimulationConfig(),
    )
    assert config.cell_diameter <= 0.5 * config.length_scale
    assert config.nx == config.ny == 200


def test_petsc_source_is_valid_python_and_has_no_serial_solve() -> None:
    source = (LINUX / "phasefield_crack_petsc.py").read_text(encoding="utf-8")
    ast.parse(source)
    forbidden = ("spsolve", ".to_scipy(", "scipy.optimize", "L-BFGS-B")
    for token in forbidden:
        assert token not in source


def test_petsc_source_contains_parallel_correctness_mechanisms() -> None:
    source = (LINUX / "phasefield_crack_petsc.py").read_text(encoding="utf-8")
    required = (
        "fem_petsc.assemble_residual",
        "fem_petsc.assemble_jacobian",
        "setVariableBounds",
        "PETSc.ScatterMode.REVERSE",
        "self.comm.allreduce",
        "self.comm.gather",
        "self.comm.bcast",
        "self.rank == 0",
        "getConvergedReason",
        "projected[at_lower & (gradient > 0.0)] = 0.0",
        "boundary_condition.set(self.u.x.array)",
        "displacement PETSc confirmation solve",
        "RecoverableStepError",
        "comm.Abort(1)",
    )
    for token in required:
        assert token in source


def test_linux_environment_is_one_conda_forge_petsc_stack() -> None:
    text = (LINUX / "environment-linux.yml").read_text(encoding="utf-8")
    for dependency in (
        "fenics-dolfinx=0.11.0",
        "mpi=1.0=mpich",
        "mpich",
        "mpi4py",
        "petsc=3.25.*=real_*",
        "petsc4py=3.25.*",
        "hdf5=*=mpi_mpich_*",
    ):
        assert dependency in text
    assert "conda-forge" in text
    assert "nodefaults" in text
    assert "pip:" not in text


def test_preflight_and_job_files_have_expected_rank_controls() -> None:
    preflight = (LINUX / "check_environment.py").read_text(encoding="utf-8")
    ast.parse(preflight)
    assert "--expected-ranks" in preflight
    assert "--expected-nodes" in preflight
    assert "--allow-one-rank" in preflight
    assert "MPI.Get_processor_name()" in preflight
    assert "os.sched_getaffinity(0)" in preflight
    assert "rank_cpu_affinity" in preflight
    assert "ranks_per_node" in preflight
    assert "MPI.Comm.Compare" in preflight
    assert "PETSc.COMM_WORLD.tompi4py()" in preflight
    assert "fem_petsc.assemble_matrix" in preflight
    assert "setVariableBounds" in preflight
    assert "comm.Abort(1)" in preflight
    assert "getBuffer(readonly=True)" in preflight
    assert "io.XDMFFile" in preflight
    assert "read_mesh" in preflight

    slurm = (LINUX / "slurm_phasefield.sbatch").read_text(encoding="utf-8")
    pbs = (LINUX / "pbs_phasefield.pbs").read_text(encoding="utf-8")
    launcher = (LINUX / "run_linux.sh").read_text(encoding="utf-8")
    assert "#SBATCH --ntasks=16" in slurm
    assert 'NPROCS="${SLURM_NTASKS}"' in slurm
    assert "mpiprocs=16" in pbs
    assert "PBS_NODEFILE" in pbs
    assert 'NPROCS="${NPROCS:-16}"' in launcher
    assert "--expected-ranks" in launcher
    assert "--expected-nodes" in launcher
    assert "--check-xdmf" in launcher
    assert 'PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"' in launcher
    assert "--report-json" in preflight
    assert 'if [[ -n "${EXPECTED_NODES:-}" ]]' in launcher


def test_non_linux_preflight_can_tolerate_missing_cpu_affinity(monkeypatch) -> None:
    monkeypatch.delattr(check_environment.os, "sched_getaffinity", raising=False)

    cpu_ids, error = check_environment._read_cpu_affinity(
        "Windows", allow_non_linux=True
    )
    assert cpu_ids == []
    assert error is None

    _cpu_ids, error = check_environment._read_cpu_affinity(
        "Linux", allow_non_linux=True
    )
    assert error is not None
    assert "cannot read CPU affinity" in error


def test_cluster_bundle_has_no_windows_specific_paths() -> None:
    for path in LINUX.rglob("*"):
        if path.is_file() and path.suffix not in {".pyc"}:
            text = path.read_text(encoding="utf-8")
            assert "D:\\" not in text
            assert "MinicondaINSTALL" not in text


def test_cluster_readme_documents_16_32_and_mpi_abi_risk() -> None:
    text = (LINUX / "README.md").read_text(encoding="utf-8")
    for phrase in (
        "NPROCS=16",
        "NPROCS=32",
        "sbatch --ntasks=32",
        "PETSC_OPTIONS",
        "ABI-compatible",
        "SNESVI",
        "shared",
    ):
        assert phrase in text
