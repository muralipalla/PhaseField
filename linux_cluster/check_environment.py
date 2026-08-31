"""Collective Linux/PETSc/MPI preflight for the cluster backend."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import platform
from types import SimpleNamespace
import sys
from typing import Any, Callable

from mpi4py import MPI
import numpy as np


def _environment_integer(*names: str) -> int:
    for name in names:
        value = os.environ.get(name)
        if value:
            try:
                return int(value)
            except ValueError:
                return -1
    return 0


def _read_cpu_affinity(
    host_system: str, *, allow_non_linux: bool
) -> tuple[list[int], str | None]:
    """Read Linux affinity, tolerating its absence for opted-in diagnostics."""

    affinity_optional = host_system != "Linux" and allow_non_linux
    try:
        cpu_ids = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError) as error:
        if affinity_optional:
            return [], None
        return [], f"cannot read CPU affinity: {error}"
    if not cpu_ids and not affinity_optional:
        return [], "MPI rank has an empty CPU affinity mask"
    return cpu_ids, None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the Linux DOLFINx/PETSc/MPI cluster environment."
    )
    parser.add_argument(
        "--expected-ranks",
        type=int,
        default=_environment_integer("EXPECTED_RANKS"),
        help="Require this MPI size; 0 accepts any positive size.",
    )
    parser.add_argument(
        "--check-xdmf",
        type=Path,
        default=None,
        metavar="DIRECTORY",
        help="Collectively write and read a small XDMF/HDF5 result in DIRECTORY.",
    )
    parser.add_argument(
        "--expected-nodes",
        type=int,
        default=_environment_integer("EXPECTED_NODES", "SLURM_NNODES"),
        help="Require this many distinct MPI processor names; 0 disables the check.",
    )
    parser.add_argument(
        "--allow-one-rank",
        action="store_true",
        help="Permit a one-rank developer check instead of requiring an MPI run.",
    )
    parser.add_argument(
        "--allow-non-linux",
        action="store_true",
        help="Permit a non-Linux host for developer diagnostics only.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        metavar="FILE",
        help="Atomically write the rank-0 preflight report to FILE.",
    )
    return parser


def _fail_collectively(comm: MPI.Intracomm, local_errors: list[str]) -> None:
    gathered = comm.allgather(local_errors)
    flattened = [
        f"rank {rank}: {message}"
        for rank, messages in enumerate(gathered)
        for message in messages
    ]
    if flattened:
        if comm.rank == 0:
            print("PETSc/MPI preflight failed:", file=sys.stderr)
            for message in flattened:
                print(f"  - {message}", file=sys.stderr)
        raise SystemExit(1)


def _abort_collective_stage(
    comm: MPI.Intracomm, label: str, action: Callable[[], Any]
) -> Any:
    """Abort the MPI job if one rank fails inside a collective library stage."""

    try:
        return action()
    except Exception as error:
        try:
            print(
                f"[rank {comm.rank}] {label} failed: "
                f"{type(error).__name__}: {error}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            pass
        finally:
            comm.Abort(1)
        raise RuntimeError(f"{label} failed") from error


def _load_runtime() -> SimpleNamespace:
    """Import PETSc-backed DOLFINx only after command-line parsing."""

    from petsc4py import PETSc
    import petsc4py
    import mpi4py
    import dolfinx

    if not dolfinx.has_petsc4py:
        raise RuntimeError("DOLFINx has no petsc4py support")

    from dolfinx import fem, io, mesh
    from dolfinx.fem import petsc as fem_petsc
    import ufl

    return SimpleNamespace(
        PETSc=PETSc,
        petsc4py=petsc4py,
        mpi4py=mpi4py,
        dolfinx=dolfinx,
        fem=fem,
        fem_petsc=fem_petsc,
        io=io,
        mesh=mesh,
        ufl=ufl,
    )


def _distributed_poisson(
    comm: MPI.Intracomm, runtime: SimpleNamespace
) -> tuple[Any, ...]:
    PETSc = runtime.PETSc
    fem = runtime.fem
    fem_petsc = runtime.fem_petsc
    mesh = runtime.mesh
    ufl = runtime.ufl

    matrix = None
    right_hand_side = None
    solver = None
    completed = False
    try:
        nx = max(8, 2 * comm.size)
        domain = mesh.create_unit_square(comm, nx, 8)
        space = fem.functionspace(domain, ("Lagrange", 1))
        trial = ufl.TrialFunction(space)
        test = ufl.TestFunction(space)
        bilinear = fem.form(ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx)
        linear = fem.form(1.0 * test * ufl.dx)

        facets = mesh.locate_entities_boundary(
            domain,
            domain.topology.dim - 1,
            lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0),
        )
        dofs = fem.locate_dofs_topological(
            space, domain.topology.dim - 1, facets
        )
        boundary = fem.dirichletbc(PETSc.ScalarType(0.0), dofs, space)

        matrix = fem_petsc.assemble_matrix(bilinear, bcs=[boundary])
        matrix.assemble()
        right_hand_side = fem_petsc.assemble_vector(linear)
        fem_petsc.apply_lifting(
            right_hand_side, [bilinear], bcs=[[boundary]]
        )
        right_hand_side.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        fem_petsc.set_bc(right_hand_side, [boundary])

        solution = fem.Function(space, name="preflight_solution")
        solver = PETSc.KSP().create(comm)
        solver.setOptionsPrefix("preflight_")
        solver.setType("cg")
        solver.getPC().setType("gamg")
        solver.setTolerances(rtol=1.0e-10, atol=1.0e-14, max_it=500)
        solver.setOperators(matrix)
        solver.setFromOptions()
        solver.solve(right_hand_side, solution.x.petsc_vec)
        solution.x.scatter_forward()
        reason = int(solver.getConvergedReason())
        norm = float(solution.x.petsc_vec.norm(PETSc.NormType.NORM_2))
        cells = int(
            domain.topology.index_map(domain.topology.dim).size_global
        )
        solver_type = f"{solver.getType()}/{solver.getPC().getType()}"
        if reason <= 0:
            raise RuntimeError(
                f"distributed Poisson KSP diverged with reason {reason}."
            )
        if not math.isfinite(norm) or norm <= 0.0:
            raise RuntimeError(f"distributed solution norm is invalid: {norm}.")
        completed = True
        return norm, cells, solver_type, domain, solution
    finally:
        # PETSc destruction is collective. If a rank has already failed,
        # propagate immediately so _abort_collective_stage terminates peers
        # that may still be blocked in a PETSc call.
        if completed:
            for petsc_object in (solver, right_hand_side, matrix):
                if petsc_object is not None:
                    petsc_object.destroy()


def _distributed_snesvi(
    comm: MPI.Intracomm, runtime: SimpleNamespace
) -> tuple[int, float, str]:
    """Solve a tiny distributed bound-constrained system with SNESVI."""

    PETSc = runtime.PETSc
    matrix = None
    solution = None
    residual = None
    right_hand_side = None
    lower = None
    upper = None
    solver = None
    completed = False
    local_size = 2
    global_size = local_size * comm.size

    def assemble_coupled_operator(operator: Any) -> None:
        operator.zeroEntries()
        row_start, row_end = operator.getOwnershipRange()
        for row in range(row_start, row_end):
            operator.setValue(row, row, 2.0)
            if row > 0:
                operator.setValue(row, row - 1, -0.25)
            if row + 1 < global_size:
                operator.setValue(row, row + 1, -0.25)
        operator.assemblyBegin()
        operator.assemblyEnd()

    try:
        solution = PETSc.Vec().createMPI(
            (local_size, global_size), comm=comm
        )
        residual = solution.duplicate()
        right_hand_side = solution.duplicate()
        lower = solution.duplicate()
        upper = solution.duplicate()
        solution.set(0.5)
        right_hand_side.set(3.0)
        lower.set(0.0)
        upper.set(0.75)

        matrix = PETSc.Mat().createAIJ(
            size=(solution.getSizes(), solution.getSizes()),
            nnz=3,
            comm=comm,
        )
        assemble_coupled_operator(matrix)

        def form_function(_snes: Any, state: Any, target: Any) -> None:
            matrix.mult(state, target)
            target.axpy(-1.0, right_hand_side)

        def form_jacobian(
            _snes: Any, _state: Any, jacobian: Any, preconditioner: Any
        ) -> None:
            assemble_coupled_operator(jacobian)
            if preconditioner.handle != jacobian.handle:
                assemble_coupled_operator(preconditioner)

        solver = PETSc.SNES().create(comm)
        solver.setOptionsPrefix("preflight_vi_")
        solver.setType("vinewtonrsls")
        solver.setFunction(form_function, residual)
        solver.setJacobian(form_jacobian, matrix, matrix)
        solver.setVariableBounds(lower, upper)
        solver.setTolerances(rtol=0.0, atol=1.0e-12, stol=0.0, max_it=50)
        solver.getKSP().setType("gmres")
        solver.getKSP().setTolerances(rtol=1.0e-12, atol=1.0e-14, max_it=100)
        solver.getKSP().getPC().setType("jacobi")
        solver.setFromOptions()
        actual_type = str(solver.getType())
        if actual_type not in {"vinewtonrsls", "vinewtonssls"}:
            raise RuntimeError(
                "preflight SNES type was overridden with a non-VI solver: "
                f"{actual_type}"
            )
        solver.solve(None, solution)

        reason = int(solver.getConvergedReason())
        with solution.getBuffer(readonly=True) as local_buffer:
            values = np.asarray(local_buffer, dtype=float).copy()
        matrix.mult(solution, residual)
        residual.axpy(-1.0, right_hand_side)
        with residual.getBuffer(readonly=True) as local_buffer:
            gradient = np.asarray(local_buffer, dtype=float).copy()
        tolerance = 1.0e-10
        projected = gradient.copy()
        at_lower = values <= tolerance
        at_upper = values >= 0.75 - tolerance
        projected[at_lower & (gradient > 0.0)] = 0.0
        projected[at_upper & (gradient < 0.0)] = 0.0
        local_kkt = float(np.max(np.abs(projected), initial=0.0))
        local_feasibility = float(
            max(
                np.max(-values, initial=0.0),
                np.max(values - 0.75, initial=0.0),
            )
        )
        local_target_error = float(
            np.max(np.abs(values - 0.75), initial=0.0)
        )
        local_nonfinite = not (
            np.all(np.isfinite(values)) and np.all(np.isfinite(gradient))
        )
        kkt = float(comm.allreduce(local_kkt, op=MPI.MAX))
        feasibility = float(comm.allreduce(local_feasibility, op=MPI.MAX))
        target_error = float(comm.allreduce(local_target_error, op=MPI.MAX))
        nonfinite = bool(comm.allreduce(local_nonfinite, op=MPI.LOR))
        if reason <= 0:
            raise RuntimeError(f"distributed SNESVI diverged with reason {reason}.")
        if nonfinite or kkt > 1.0e-9 or feasibility > 1.0e-12:
            raise RuntimeError(
                "distributed SNESVI produced an invalid bound-constrained state: "
                f"KKT={kkt:.3e}, feasibility={feasibility:.3e}."
            )
        if target_error > 1.0e-10:
            raise RuntimeError(
                "distributed SNESVI did not reach the expected active upper bound: "
                f"error={target_error:.3e}."
            )
        completed = True
        return int(solver.getIterationNumber()), kkt, actual_type
    finally:
        if completed:
            for petsc_object in (
                solver,
                upper,
                lower,
                right_hand_side,
                residual,
                solution,
                matrix,
            ):
                if petsc_object is not None:
                    petsc_object.destroy()


def _xdmf_round_trip(
    comm: MPI.Intracomm,
    runtime: SimpleNamespace,
    output_path: Path,
    domain: Any,
    solution: Any,
    expected_cells: int,
) -> int:
    """Collectively write, close, reopen, and validate an XDMF mesh."""

    write_file = runtime.io.XDMFFile(
        comm, output_path / "preflight.xdmf", "w"
    )
    write_file.write_mesh(domain)
    write_file.write_function(solution, 0.0)
    write_file.close()

    read_file = runtime.io.XDMFFile(
        comm, output_path / "preflight.xdmf", "r"
    )
    reloaded_mesh = read_file.read_mesh(name=domain.name)
    read_file.close()
    reloaded_cells = int(
        reloaded_mesh.topology.index_map(reloaded_mesh.topology.dim).size_global
    )
    if reloaded_cells != expected_cells:
        raise RuntimeError(
            "parallel XDMF read-back changed the global cell count: "
            f"{reloaded_cells} != {expected_cells}"
        )
    return reloaded_cells


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    comm = MPI.COMM_WORLD
    host_system = platform.system()

    runtime = None
    runtime_error: str | None = None
    try:
        runtime = _load_runtime()
    except Exception as error:
        runtime_error = f"runtime import failed: {type(error).__name__}: {error}"
    _fail_collectively(comm, [] if runtime_error is None else [runtime_error])
    assert runtime is not None

    PETSc = runtime.PETSc
    dolfinx = runtime.dolfinx
    local_errors: list[str] = []
    if host_system != "Linux" and not args.allow_non_linux:
        local_errors.append(
            f"host is {host_system}, but the cluster backend requires Linux"
        )
    if args.expected_ranks < 0:
        local_errors.append("--expected-ranks cannot be negative")
    if args.expected_nodes < 0:
        local_errors.append("--expected-nodes cannot be negative")
    if comm.size == 1 and not args.allow_one_rank:
        local_errors.append(
            "preflight has only one MPI rank; launch with mpiexec/srun or pass "
            "--allow-one-rank for a developer-only check"
        )
    if args.expected_ranks and comm.size != args.expected_ranks:
        local_errors.append(
            f"MPI size is {comm.size}, expected {args.expected_ranks}"
        )

    try:
        petsc_mpi_comm = PETSc.COMM_WORLD.tompi4py()
        communicator_relation = MPI.Comm.Compare(comm, petsc_mpi_comm)
        if communicator_relation not in {MPI.IDENT, MPI.CONGRUENT}:
            local_errors.append(
                "PETSc.COMM_WORLD is not identical or congruent to mpi4py "
                f"MPI.COMM_WORLD (MPI comparison={communicator_relation})"
            )
    except Exception as error:
        local_errors.append(
            "PETSc/mpi4py communicator comparison failed: "
            f"{type(error).__name__}: {error}"
        )
    if np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating):
        local_errors.append("PETSc uses complex scalars; the solver requires real scalars")
    petsc_version = tuple(int(item) for item in PETSc.Sys.getVersion())
    try:
        petsc4py_version = tuple(
            int(item) for item in runtime.petsc4py.__version__.split(".")[:2]
        )
    except (AttributeError, ValueError):
        petsc4py_version = ()
    if petsc_version[:2] != (3, 25):
        local_errors.append(
            f"PETSc {'.'.join(map(str, petsc_version))} is installed; "
            "this bundle targets 3.25.x"
        )
    if petsc4py_version != petsc_version[:2]:
        local_errors.append(
            f"petsc4py {runtime.petsc4py.__version__} does not match "
            f"PETSc {'.'.join(map(str, petsc_version))} at major/minor level"
        )
    if not dolfinx.__version__.startswith("0.11."):
        local_errors.append(
            f"DOLFINx {dolfinx.__version__} is installed; this bundle targets 0.11.x"
        )
    processor_names = comm.allgather(MPI.Get_processor_name())
    ranks_per_node = dict(sorted(Counter(processor_names).items()))
    distinct_nodes = list(ranks_per_node)
    local_cpu_affinity, affinity_error = _read_cpu_affinity(
        host_system, allow_non_linux=args.allow_non_linux
    )
    if affinity_error is not None:
        local_errors.append(affinity_error)
    rank_cpu_affinity = comm.allgather(local_cpu_affinity)
    if args.expected_nodes and len(distinct_nodes) != args.expected_nodes:
        local_errors.append(
            f"MPI ranks occupy {len(distinct_nodes)} distinct host(s), "
            f"expected {args.expected_nodes}: {ranks_per_node}"
        )
    _fail_collectively(comm, local_errors)

    poisson_result = _abort_collective_stage(
        comm,
        "distributed Poisson/KSP check",
        lambda: _distributed_poisson(comm, runtime),
    )
    norm, cell_count, solver_type, domain, solution = poisson_result

    snesvi_result = _abort_collective_stage(
        comm,
        "distributed SNESVI check",
        lambda: _distributed_snesvi(comm, runtime),
    )
    snesvi_iterations, snesvi_kkt, snesvi_type = snesvi_result

    xdmf_reloaded_cells: int | None = None
    if args.check_xdmf is not None:
        output_payload: tuple[str | None, str | None]
        if comm.rank == 0:
            try:
                output_dir = args.check_xdmf.expanduser().resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                output_payload = (str(output_dir), None)
            except Exception as error:
                output_payload = (None, f"{type(error).__name__}: {error}")
        else:
            output_payload = (None, None)
        output_text, output_error = comm.bcast(output_payload, root=0)
        _fail_collectively(comm, [] if output_error is None else [output_error])
        assert output_text is not None
        output_path = Path(output_text)

        xdmf_reloaded_cells = _abort_collective_stage(
            comm,
            "parallel XDMF/HDF5 round trip",
            lambda: _xdmf_round_trip(
                comm,
                runtime,
                output_path,
                domain,
                solution,
                cell_count,
            ),
        )

        file_error: str | None = None
        if comm.rank == 0:
            try:
                for filename in ("preflight.xdmf", "preflight.h5"):
                    artifact = output_path / filename
                    if not artifact.is_file() or artifact.stat().st_size == 0:
                        raise RuntimeError(f"missing or empty artifact: {artifact}")
            except Exception as error:
                file_error = f"artifact check failed: {type(error).__name__}: {error}"
        _fail_collectively(comm, [] if file_error is None else [file_error])

    report: dict[str, Any] | None = None
    if comm.rank == 0:
        report = {
            "status": "ok",
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "dolfinx": dolfinx.__version__,
            "petsc": ".".join(str(item) for item in petsc_version),
            "petsc4py": runtime.petsc4py.__version__,
            "mpi4py": runtime.mpi4py.__version__,
            "mpi_ranks": comm.size,
            "mpi_nodes": len(distinct_nodes),
            "processor_names": distinct_nodes,
            "ranks_per_node": ranks_per_node,
            "rank_cpu_affinity": {
                str(rank): cpu_ids
                for rank, cpu_ids in enumerate(rank_cpu_affinity)
            },
            "rank_affinity_sizes": [
                len(cpu_ids) for cpu_ids in rank_cpu_affinity
            ],
            "mpi_library": MPI.Get_library_version().strip(),
            "test_problem_cells": cell_count,
            "test_solution_vector_2_norm": norm,
            "test_linear_solver": solver_type,
            "test_snesvi_solver": snesvi_type,
            "test_snesvi_iterations": snesvi_iterations,
            "test_snesvi_kkt_residual": snesvi_kkt,
            "xdmf_checked": args.check_xdmf is not None,
            "xdmf_reloaded_cells": xdmf_reloaded_cells,
        }

    report_error: str | None = None
    if comm.rank == 0 and args.report_json is not None:
        assert report is not None
        try:
            report_path = args.report_json.expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = report_path.with_name(report_path.name + ".tmp")
            temporary_path.write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            temporary_path.replace(report_path)
        except Exception as error:
            report_error = (
                "preflight report write failed: "
                f"{type(error).__name__}: {error}"
            )
    _fail_collectively(comm, [] if report_error is None else [report_error])

    if comm.rank == 0:
        assert report is not None
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
