"""Distributed PETSc/MPI backend for the corrected AT2 fracture benchmark.

This module is deliberately separate from the native-Windows SciPy backend.
It keeps the same governing equations and staggered load algorithm, but all
linear algebra, bound constraints, convergence decisions, diagnostics, and
field output are collective over the supplied MPI communicator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time
from typing import Any, Callable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
import mpi4py
import numpy as np
from petsc4py import PETSc
import petsc4py

import basix
import dolfinx
import ffcx
from dolfinx import fem, io, mesh as dmesh
from dolfinx.fem import petsc as fem_petsc
import ufl

from phasefield_crack import (
    CRACK_FRONT_THRESHOLD,
    MODEL_VERSION,
    NewtonReport,
    PhaseFieldSimulation,
    PhaseReport,
)
from phasefield_input import InputFileError, parse_input_file
from phasefield_crack_metrics import CrackMetrics, compute_crack_metrics
from linux_cluster.petsc_config import PetscSimulationConfig, quick_petsc_config


PETSC_BACKEND_VERSION = "1.1.0"


class RecoverableStepError(RuntimeError):
    """A synchronized nonlinear failure that may be retried by load cutback."""


def _abort_mpi_job(
    comm: MPI.Intracomm, label: str, error: Exception
) -> None:
    """Terminate peers that may be blocked after a rank-local collective error."""

    try:
        print(
            f"[rank {comm.rank}] fatal {label}: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        pass
    finally:
        comm.Abort(1)
    raise RuntimeError(f"fatal {label}") from error


class PetscPhaseFieldSimulation(PhaseFieldSimulation):
    """MPI-distributed staggered simulation using PETSc SNES and SNESVI."""

    def __init__(
        self,
        config: PetscSimulationConfig,
        comm: MPI.Intracomm = MPI.COMM_WORLD,
    ):
        config.validate()
        if not dolfinx.has_petsc4py:
            raise RuntimeError("DOLFINx was built without petsc4py support.")
        if np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating):
            raise RuntimeError("This implementation requires a real PETSc scalar build.")

        self.comm = comm
        self.rank = comm.rank
        self.size = comm.size
        self.config = self._prepare_output_directory(config)
        self._start_time = time.perf_counter()
        self._active_lower_visits = 0
        self._active_irreversibility_visits = 0
        self.records: list[dict[str, float | int]] = []

        self._load_material_spec()
        self._build_mesh_and_spaces()
        self._build_material_fields()
        self._build_boundary_conditions()
        self._build_forms()
        self._build_petsc_solvers()
        self._initialize_fields()
        self._write_material_fields()
        self._write_manifest()

    def _prepare_output_directory(
        self, config: PetscSimulationConfig
    ) -> PetscSimulationConfig:
        payload: tuple[str | None, str | None]
        if self.rank == 0:
            try:
                output_dir = Path(config.output_directory).expanduser().resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                payload = (str(output_dir), None)
            except Exception as error:  # propagated to every rank below
                payload = (
                    None,
                    f"{type(error).__name__}: cannot create output directory: {error}",
                )
        else:
            payload = (None, None)
        path_text, error_text = self.comm.bcast(payload, root=0)
        if error_text is not None or path_text is None:
            raise RuntimeError(error_text or "Output-directory setup failed.")
        self.output_dir = Path(path_text)
        return replace(config, output_directory=path_text)

    def _build_mesh_and_spaces(self) -> None:
        c = self.config
        self.domain = dmesh.create_rectangle(
            self.comm,
            [np.array([0.0, 0.0]), np.array([c.width, c.height])],
            [c.nx, c.ny],
            cell_type=dmesh.CellType.triangle,
        )
        self.V_u = fem.functionspace(self.domain, ("Lagrange", 1, (2,)))
        self.V_phi = fem.functionspace(self.domain, ("Lagrange", 1))
        self.V_H = fem.functionspace(self.domain, ("DG", 0))

        self.u = fem.Function(self.V_u, name="displacement")
        self.phi = fem.Function(self.V_phi, name="phase_field")
        self.history = fem.Function(self.V_H, name="history")
        self.history_trial = fem.Function(
            self.V_H, name="current_tensile_energy"
        )

        phase_map = self.V_phi.dofmap.index_map
        self._owned_phase_dofs = phase_map.size_local
        local_phase_size = phase_map.size_local + phase_map.num_ghosts
        local_numbers = np.arange(local_phase_size, dtype=np.int32)
        self._phase_global_ids = np.asarray(
            phase_map.local_to_global(local_numbers), dtype=np.int64
        )
        phase_coordinates = self.V_phi.tabulate_dof_coordinates()

        cell_count = self.domain.topology.index_map(
            self.domain.topology.dim
        ).size_local
        local_cells = np.asarray(
            [
                self._phase_global_ids[self.V_phi.dofmap.cell_dofs(cell)]
                for cell in range(cell_count)
            ],
            dtype=np.int64,
        ).reshape((-1, 3))
        local_payload = (
            self._phase_global_ids[: self._owned_phase_dofs].copy(),
            phase_coordinates[: self._owned_phase_dofs].copy(),
            local_cells,
        )
        gathered = self.comm.gather(local_payload, root=0)

        self._root_phase_ids: np.ndarray | None = None
        self._root_phase_coordinates: np.ndarray | None = None
        self._root_phase_cells: np.ndarray | None = None
        self._root_phase_neighbors: tuple[np.ndarray, ...] | None = None
        self._root_phase_position: dict[int, int] | None = None
        def build_root_topology() -> None:
            assert gathered is not None
            global_ids = np.concatenate([part[0] for part in gathered])
            global_coordinates = np.vstack([part[1] for part in gathered])
            order = np.argsort(global_ids)
            global_ids = global_ids[order]
            global_coordinates = global_coordinates[order]
            if np.unique(global_ids).size != global_ids.size:
                raise RuntimeError("Owned phase-field global IDs are not unique.")
            position = {
                int(global_id): index
                for index, global_id in enumerate(global_ids.tolist())
            }
            cells_by_id = np.vstack([part[2] for part in gathered])
            cells = np.asarray(
                [
                    [position[int(global_id)] for global_id in cell]
                    for cell in cells_by_id
                ],
                dtype=np.int32,
            )
            neighbor_sets: list[set[int]] = [set() for _ in range(global_ids.size)]
            for cell in cells:
                for dof in cell:
                    neighbor_sets[int(dof)].update(
                        int(other) for other in cell if other != dof
                    )
            self._root_phase_ids = global_ids
            self._root_phase_coordinates = global_coordinates
            self._root_phase_cells = cells
            self._root_phase_neighbors = tuple(
                np.asarray(sorted(neighbors), dtype=np.int32)
                for neighbors in neighbor_sets
            )
            self._root_phase_position = position

        self._root_compute("global phase-topology construction", build_root_topology)

        self._owned_u_entries = (
            self.V_u.dofmap.index_map.size_local * self.V_u.dofmap.index_map_bs
        )
        self._owned_history_dofs = self.V_H.dofmap.index_map.size_local

    def _build_boundary_conditions(self) -> None:
        # The shared implementation constructs the selected legacy/relative/
        # symmetric scheme, all load Constants, boundary tags, and the notch.
        super()._build_boundary_conditions()

        owned_notch = self.notch_dofs[
            self.notch_dofs < self._owned_phase_dofs
        ]
        local_notch_ids = self._phase_global_ids[owned_notch]
        gathered_notch_ids = self.comm.gather(local_notch_ids, root=0)
        self._root_notch_positions: np.ndarray | None = None

        def build_root_notch() -> None:
            assert gathered_notch_ids is not None
            assert self._root_phase_position is not None
            notch_ids = np.unique(np.concatenate(gathered_notch_ids))
            self._root_notch_positions = np.asarray(
                [self._root_phase_position[int(global_id)] for global_id in notch_ids],
                dtype=np.int32,
            )

        self._root_compute("global notch construction", build_root_notch)

    def _build_petsc_solvers(self) -> None:
        c = self.config
        self._u_matrix = fem_petsc.create_matrix(self.form_u_jacobian)
        self._u_residual = fem_petsc.create_vector(self.V_u)
        self._u_snes = PETSc.SNES().create(self.comm)
        u_context = {
            "u": self.u,
            "residual": self.form_u_residual,
            "jacobian": self.form_u_jacobian,
            "bcs": self.displacement_bcs,
        }
        self._u_snes.setFunction(
            fem_petsc.assemble_residual, self._u_residual, kargs=u_context
        )
        self._u_snes.setJacobian(
            fem_petsc.assemble_jacobian,
            self._u_matrix,
            self._u_matrix,
            kargs={
                "u": self.u,
                "jacobian": self.form_u_jacobian,
                "preconditioner": None,
                "bcs": self.displacement_bcs,
            },
        )
        self._configure_snes(
            self._u_snes,
            prefix="pf_u_",
            snes_type=c.displacement_snes_type,
            ksp_type=c.displacement_ksp_type,
            pc_type=c.displacement_pc_type,
            rtol=c.newton_relative_tolerance,
            atol=c.newton_absolute_tolerance,
            stol=0.0,
            max_iterations=c.max_newton_iterations,
        )

        self._phase_matrix = fem_petsc.create_matrix(self.form_phi_bilinear)
        self._phase_rhs_ghosted = fem_petsc.create_vector(self.V_phi)
        self._phase_rhs = self._phase_matrix.createVecLeft()
        self._phase_residual = fem_petsc.create_vector(self.V_phi)
        self._phase_lower = fem.Function(self.V_phi, name="phase_lower_bound")
        self._phase_upper = fem.Function(self.V_phi, name="phase_upper_bound")
        self._phase_snes = PETSc.SNES().create(self.comm)
        self._phase_snes.setFunction(self._phase_function, self._phase_residual)
        self._phase_snes.setJacobian(
            self._phase_jacobian, self._phase_matrix, self._phase_matrix
        )
        self._configure_snes(
            self._phase_snes,
            prefix="pf_phi_",
            snes_type=c.phase_snes_type,
            ksp_type=c.phase_ksp_type,
            pc_type=c.phase_pc_type,
            rtol=0.0,
            atol=0.1 * c.phase_kkt_tolerance,
            stol=0.0,
            max_iterations=c.phase_optimizer_max_iterations,
        )
        phase_type = self._phase_snes.getType().casefold()
        if phase_type not in {"vinewtonrsls", "vinewtonssls"}:
            raise RuntimeError(
                "PETSc options changed pf_phi_snes_type to a solver that does "
                f"not enforce bounds: {phase_type!r}."
            )
        if (
            phase_type == "vinewtonssls"
            and self._phase_snes.getKSP().getType().casefold() == "cg"
        ):
            raise RuntimeError(
                "pf_phi_ksp_type=cg is incompatible with vinewtonssls; use gmres."
            )
        # The independent KKT check is absolute. Do not allow PETSc options to
        # terminate this VI solve on relative residual or step size instead.
        self._phase_snes.setTolerances(
            rtol=0.0,
            atol=0.1 * c.phase_kkt_tolerance,
            stol=0.0,
            max_it=c.phase_optimizer_max_iterations,
        )

    def _configure_snes(
        self,
        snes: PETSc.SNES,
        *,
        prefix: str,
        snes_type: str,
        ksp_type: str,
        pc_type: str,
        rtol: float,
        atol: float,
        stol: float,
        max_iterations: int,
    ) -> None:
        c = self.config
        snes.setOptionsPrefix(prefix)
        snes.setType(snes_type)
        snes.setTolerances(
            rtol=rtol, atol=atol, stol=stol, max_it=max_iterations
        )
        ksp = snes.getKSP()
        ksp.setType(ksp_type)
        ksp.setTolerances(
            rtol=c.ksp_relative_tolerance,
            atol=c.ksp_absolute_tolerance,
            max_it=c.ksp_max_iterations,
        )
        ksp.getPC().setType(pc_type)
        if c.petsc_monitor:
            snes.setMonitor(
                lambda _solver, iteration, norm: self._monitor(
                    prefix, iteration, norm
                )
            )
        snes.setFromOptions()

    def _monitor(self, prefix: str, iteration: int, norm: float) -> None:
        self._root_log(
            f"[{prefix}SNES {iteration:03d}] residual={norm:.6e}"
        )

    def _root_log(self, message: str) -> None:
        """Write optional progress without letting a broken log pipe split ranks."""

        if self.rank == 0:
            try:
                print(message, flush=True)
            except Exception:
                pass

    def _phase_function(
        self, _snes: PETSc.SNES, x: PETSc.Vec, residual: PETSc.Vec
    ) -> None:
        self._phase_matrix.mult(x, residual)
        residual.axpy(-1.0, self._phase_rhs)

    def _phase_jacobian(
        self,
        _snes: PETSc.SNES,
        _x: PETSc.Vec,
        jacobian: PETSc.Mat,
        preconditioner: PETSc.Mat,
    ) -> None:
        jacobian.zeroEntries()
        fem_petsc.assemble_matrix(jacobian, self.form_phi_bilinear)
        jacobian.assemble()
        if preconditioner.handle != jacobian.handle:
            preconditioner.zeroEntries()
            fem_petsc.assemble_matrix(
                preconditioner, self.form_phi_bilinear
            )
            preconditioner.assemble()

    def _global_max(self, local_value: float) -> float:
        return float(self.comm.allreduce(float(local_value), op=MPI.MAX))

    def _global_min(self, local_value: float) -> float:
        return float(self.comm.allreduce(float(local_value), op=MPI.MIN))

    def _global_sum(self, local_value: float) -> float:
        return float(self.comm.allreduce(float(local_value), op=MPI.SUM))

    def _assert_finite_owned(
        self, label: str, values: np.ndarray, owned: int
    ) -> None:
        local_invalid = not bool(np.all(np.isfinite(values[:owned])))
        if self.comm.allreduce(local_invalid, op=MPI.LOR):
            raise RecoverableStepError(
                f"{label} contains non-finite owned values."
            )

    def _root_action(
        self, label: str, action: Callable[[], Any]
    ) -> Any:
        payload: tuple[Any, str | None]
        if self.rank == 0:
            try:
                payload = (action(), None)
            except Exception as error:
                payload = (
                    None,
                    f"{label} failed: {type(error).__name__}: {error}",
                )
        else:
            payload = (None, None)
        result, error_text = self.comm.bcast(payload, root=0)
        if error_text is not None:
            raise RuntimeError(error_text)
        return result

    def _root_compute(
        self, label: str, action: Callable[[], Any]
    ) -> Any:
        result: Any = None
        error_text: str | None = None
        if self.rank == 0:
            try:
                result = action()
            except Exception as error:
                error_text = f"{label} failed: {type(error).__name__}: {error}"
        error_text = self.comm.bcast(error_text, root=0)
        if error_text is not None:
            raise RuntimeError(error_text)
        return result

    def _boundary_violation(self) -> float:
        correction = np.zeros_like(self.u.x.array)
        for boundary_condition in self.displacement_bcs:
            boundary_condition.set(correction, self.u.x.array, alpha=1.0)
        constrained = self.constrained_u_dofs[
            self.constrained_u_dofs < self._owned_u_entries
        ]
        local = float(
            np.max(np.abs(correction[constrained]), initial=0.0)
        )
        result = self._global_max(local)
        if not math.isfinite(result):
            raise RecoverableStepError(
                "The displacement boundary violation is non-finite."
            )
        return result

    def _free_mechanical_residual_norm(self) -> float:
        residual = fem_petsc.assemble_vector(self.form_u_residual)
        try:
            residual.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            values = np.asarray(residual.array[: self._owned_u_entries]).copy()
            constrained = self.constrained_u_dofs[
                self.constrained_u_dofs < self._owned_u_entries
            ]
            values[constrained] = 0.0
            local_invalid = not bool(np.all(np.isfinite(values)))
            if self.comm.allreduce(local_invalid, op=MPI.LOR):
                raise RecoverableStepError(
                    "The free mechanical residual is non-finite."
                )
            local = float(np.max(np.abs(values), initial=0.0))
            return self._global_max(local)
        finally:
            residual.destroy()

    def _reaction_components(self) -> dict[str, float]:
        residual = fem_petsc.assemble_vector(self.form_u_residual)
        try:
            residual.ghostUpdate(
                addv=PETSc.InsertMode.ADD,
                mode=PETSc.ScatterMode.REVERSE,
            )
            owned_values = np.asarray(
                residual.array[: self._owned_u_entries], dtype=float
            )
            reactions: dict[str, float] = {}
            for name, dofs in self._boundary_u_dofs.items():
                owned_dofs = dofs[dofs < self._owned_u_entries]
                local_sum = float(
                    np.sum(owned_values[owned_dofs], dtype=float)
                )
                reactions[name] = self._global_sum(local_sum)
            if not all(math.isfinite(value) for value in reactions.values()):
                raise RuntimeError("A global grip reaction is non-finite.")
            return reactions
        finally:
            residual.destroy()

    def _solve_displacement(self) -> NewtonReport:
        c = self.config
        # Start from a BC-feasible iterate so the residual reference has the
        # same free-DOF infinity-norm meaning as the final acceptance check.
        for boundary_condition in self.displacement_bcs:
            boundary_condition.set(self.u.x.array)
        self.u.x.scatter_forward()
        reference = self._free_mechanical_residual_norm()
        residual_limit = c.newton_absolute_tolerance + (
            c.newton_relative_tolerance
            * max(reference, c.newton_absolute_tolerance)
        )
        # A global PETSc 2-norm below this value is a sufficient condition for
        # the desired infinity-norm bound; the explicit postcheck remains the
        # authoritative project criterion.
        self._u_snes.setTolerances(
            rtol=0.0,
            atol=residual_limit,
            stol=0.0,
            max_it=c.max_newton_iterations,
        )
        def execute_snes(label: str) -> None:
            try:
                self._u_snes.solve(None, self.u.x.petsc_vec)
            except Exception as error:
                _abort_mpi_job(self.comm, label, error)

        execute_snes("displacement PETSc solve")
        self.u.x.scatter_forward()
        self._assert_finite_owned(
            "displacement", self.u.x.array, self._owned_u_entries
        )
        reason = int(self._u_snes.getConvergedReason())
        if reason <= 0:
            raise RecoverableStepError(
                "PETSc displacement SNES failed: "
                f"reason={reason}, iterations={self._u_snes.getIterationNumber()}, "
                f"norm={self._u_snes.getFunctionNorm():.3e}."
            )
        iterations = int(self._u_snes.getIterationNumber())
        last_increment = (
            0.0
            if iterations == 0
            else float(
                self._u_snes.getSolutionUpdate().norm(
                    PETSc.NormType.NORM_INFINITY
                )
            )
        )
        residual = self._free_mechanical_residual_norm()
        boundary_violation = self._boundary_violation()
        # PETSc may correctly stop after one exact Newton update because the
        # new residual already satisfies atol. The serial contract checks the
        # next correction too. Re-entering SNES at this converged iterate gives
        # an iteration-zero, zero-change confirmation instead of a false
        # load-cutback caused by the size of the successful first update.
        if (
            math.isfinite(last_increment)
            and math.isfinite(residual)
            and math.isfinite(boundary_violation)
            and residual <= residual_limit
            and boundary_violation <= c.newton_increment_tolerance
            and last_increment > c.newton_increment_tolerance
        ):
            before_confirmation = self.u.x.array[
                : self._owned_u_entries
            ].copy()
            execute_snes("displacement PETSc confirmation solve")
            self.u.x.scatter_forward()
            confirmation_reason = int(self._u_snes.getConvergedReason())
            if confirmation_reason <= 0:
                raise RecoverableStepError(
                    "PETSc displacement confirmation failed: "
                    f"reason={confirmation_reason}."
                )
            confirmation_iterations = int(
                self._u_snes.getIterationNumber()
            )
            iterations += confirmation_iterations
            self._assert_finite_owned(
                "confirmed displacement",
                self.u.x.array,
                self._owned_u_entries,
            )
            confirmation_change = self._global_max(
                float(
                    np.max(
                        np.abs(
                            self.u.x.array[: self._owned_u_entries]
                            - before_confirmation
                        ),
                        initial=0.0,
                    )
                )
            )
            last_increment = (
                confirmation_change
                if confirmation_iterations == 0
                else float(
                    self._u_snes.getSolutionUpdate().norm(
                        PETSc.NormType.NORM_INFINITY
                    )
                )
            )
            residual = self._free_mechanical_residual_norm()
            boundary_violation = self._boundary_violation()
        if (
            not math.isfinite(last_increment)
            or not math.isfinite(residual)
            or not math.isfinite(boundary_violation)
            or residual > residual_limit
            or last_increment > c.newton_increment_tolerance
            or boundary_violation > c.newton_increment_tolerance
        ):
            raise RecoverableStepError(
                "PETSc displacement SNES did not satisfy the project's "
                "global infinity-norm checks: "
                f"residual={residual:.3e} (limit={residual_limit:.3e}), "
                f"last increment={last_increment:.3e}, "
                f"BC violation={boundary_violation:.3e}."
            )
        return NewtonReport(
            iterations, residual, last_increment, boundary_violation
        )

    def _update_trial_history(self, accepted_history: np.ndarray) -> float:
        self.history_trial.interpolate(self.history_expression)
        self.history_trial.x.scatter_forward()
        owned = self._owned_history_dofs
        candidate = self.history_trial.x.array[:owned]
        accepted = accepted_history[:owned]
        self._assert_finite_owned("trial history", candidate, owned)
        self._assert_finite_owned("accepted history", accepted, owned)
        self.history.x.array[:owned] = np.maximum(accepted, candidate)
        local_violation = bool(
            np.any(self.history.x.array[:owned] < accepted - 1.0e-13)
        )
        violation = self.comm.allreduce(local_violation, op=MPI.LOR)
        self.history.x.scatter_forward()
        if violation:
            raise RecoverableStepError("History irreversibility was violated.")
        local_increment = float(
            np.max(self.history.x.array[:owned] - accepted, initial=0.0)
        )
        return self._global_max(local_increment)

    def _assemble_phase_system(self) -> None:
        self._phase_matrix.zeroEntries()
        fem_petsc.assemble_matrix(
            self._phase_matrix,
            self.form_phi_bilinear,
        )
        self._phase_matrix.assemble()

        with self._phase_rhs_ghosted.localForm() as local:
            local.set(0.0)
        fem_petsc.assemble_vector(
            self._phase_rhs_ghosted, self.form_phi_linear
        )
        self._phase_rhs_ghosted.ghostUpdate(
            addv=PETSc.InsertMode.ADD,
            mode=PETSc.ScatterMode.REVERSE,
        )
        self._phase_rhs_ghosted.copy(self._phase_rhs)

    def _solve_phase_field(self, upper_bound: np.ndarray) -> PhaseReport:
        c = self.config
        owned = self._owned_phase_dofs
        upper = np.clip(np.asarray(upper_bound, dtype=float), 0.0, 1.0)
        self._assert_finite_owned("phase upper bound", upper, owned)
        local_notch = self.notch_dofs[self.notch_dofs < owned]
        upper[local_notch] = 0.0
        previous = self.phi.x.array[:owned].copy()
        self.phi.x.array[:] = np.clip(self.phi.x.array, 0.0, upper)
        self.phase_bc.set(self.phi.x.array)
        self.phi.x.scatter_forward()
        self._assert_finite_owned("phase field", self.phi.x.array, owned)

        self._phase_lower.x.array[:] = 0.0
        self._phase_upper.x.array[:] = upper
        self._phase_lower.x.scatter_forward()
        self._phase_upper.x.scatter_forward()

        try:
            self._assemble_phase_system()
        except Exception as error:
            _abort_mpi_job(self.comm, "phase PETSc assembly", error)
        self._phase_snes.setVariableBounds(
            self._phase_lower.x.petsc_vec, self._phase_upper.x.petsc_vec
        )
        try:
            self._phase_snes.solve(None, self.phi.x.petsc_vec)
        except Exception as error:
            _abort_mpi_job(self.comm, "phase PETSc SNESVI solve", error)
        self.phi.x.scatter_forward()
        reason = int(self._phase_snes.getConvergedReason())
        if reason <= 0:
            raise RecoverableStepError(
                "PETSc phase SNESVI failed: "
                f"reason={reason}, iterations={self._phase_snes.getIterationNumber()}, "
                f"norm={self._phase_snes.getFunctionNorm():.3e}."
            )

        values = np.asarray(self.phi.x.array[:owned], dtype=float)
        self._phase_matrix.mult(self.phi.x.petsc_vec, self._phase_residual)
        self._phase_residual.axpy(-1.0, self._phase_rhs)
        gradient = np.asarray(self._phase_residual.array[:owned], dtype=float)
        self._assert_finite_owned("phase gradient", gradient, owned)
        upper_owned = upper[:owned]
        projected = gradient.copy()
        scale = 100.0 * np.finfo(float).eps
        fixed = upper_owned <= scale
        at_lower = values <= scale
        at_upper = values >= upper_owned - scale
        projected[fixed] = 0.0
        projected[at_lower & (gradient > 0.0)] = 0.0
        projected[at_upper & (gradient < 0.0)] = 0.0
        kkt_residual = self._global_max(
            float(np.max(np.abs(projected), initial=0.0))
        )
        feasibility = self._global_max(
            float(
                max(
                    np.max(-values, initial=0.0),
                    np.max(values - upper_owned, initial=0.0),
                )
            )
        )
        kkt_residual = max(kkt_residual, feasibility)
        if kkt_residual > c.phase_kkt_tolerance:
            raise RecoverableStepError(
                "PETSc phase solve failed its global KKT check: "
                f"projected residual={kkt_residual:.3e}."
            )

        active_tolerance = 1.0e-10
        movable = upper_owned > active_tolerance
        local_active_lower = int(
            np.count_nonzero(movable & (values <= active_tolerance))
        )
        local_active_upper = int(
            np.count_nonzero(
                movable & (values >= upper_owned - active_tolerance)
            )
        )
        active_lower = int(
            self.comm.allreduce(local_active_lower, op=MPI.SUM)
        )
        active_upper = int(
            self.comm.allreduce(local_active_upper, op=MPI.SUM)
        )
        self._active_lower_visits += active_lower
        self._active_irreversibility_visits += active_upper
        increment = self._global_max(
            float(np.max(np.abs(values - previous), initial=0.0))
        )
        return PhaseReport(
            increment,
            kkt_residual,
            int(self._phase_snes.getIterationNumber()),
            active_lower,
            active_upper,
        )

    def _gather_global_phase_values(self) -> np.ndarray | None:
        local = (
            self._phase_global_ids[: self._owned_phase_dofs].copy(),
            self.phi.x.array[: self._owned_phase_dofs].copy(),
        )
        gathered = self.comm.gather(local, root=0)

        def assemble_root_values() -> np.ndarray:
            assert gathered is not None
            assert self._root_phase_ids is not None
            assert self._root_phase_position is not None
            values = np.empty(self._root_phase_ids.size, dtype=float)
            for global_ids, local_values in gathered:
                positions = np.asarray(
                    [self._root_phase_position[int(item)] for item in global_ids],
                    dtype=np.int64,
                )
                values[positions] = np.asarray(local_values, dtype=float)
            if not np.all(np.isfinite(values)):
                raise RuntimeError("Gathered phase values are non-finite.")
            return values

        return self._root_compute(
            "global phase-value assembly", assemble_root_values
        )

    def _crack_metrics(
        self, threshold: float = CRACK_FRONT_THRESHOLD
    ) -> CrackMetrics:
        global_values = self._gather_global_phase_values()

        def compute_root_metrics() -> CrackMetrics:
            assert global_values is not None
            assert self._root_phase_coordinates is not None
            assert self._root_phase_neighbors is not None
            assert self._root_notch_positions is not None
            return compute_crack_metrics(
                global_values,
                self._root_phase_coordinates,
                self._root_phase_neighbors,
                self._root_notch_positions,
                threshold,
                self.represented_notch_tip,
            )

        return self._root_action(
            "global connected-crack evaluation", compute_root_metrics
        )

    def _crack_front(
        self, threshold: float = CRACK_FRONT_THRESHOLD
    ) -> tuple[float, float]:
        metrics = self._crack_metrics(threshold)
        return metrics.front_x, metrics.extension_x

    def _diagnostics(
        self,
        step: int,
        pseudo_time: float,
        displacement: float,
        staggered_iterations: int,
        newton_report: NewtonReport,
        phase_report: PhaseReport,
        free_mechanical_residual: float,
        history_increment: float,
        load_increment: float = 0.0,
        cutbacks_before_step: int = 0,
    ) -> dict[str, float | int]:
        elastic = self._global_sum(
            float(fem.assemble_scalar(self.elastic_energy_form))
        )
        fracture = self._global_sum(
            float(fem.assemble_scalar(self.fracture_energy_form))
        )
        reactions = self._reaction_components()
        top_reaction_x = reactions["top_x"]
        top_reaction_y = reactions["top_y"]
        bottom_reaction_x = reactions["bottom_x"]
        bottom_reaction_y = reactions["bottom_y"]
        damage_integral = self._global_sum(
            float(fem.assemble_scalar(self.damage_integral_form))
        )
        if not all(
            math.isfinite(value)
            for value in (
                elastic,
                fracture,
                top_reaction_x,
                top_reaction_y,
                bottom_reaction_x,
                bottom_reaction_y,
                damage_integral,
            )
        ):
            raise RuntimeError("A global energy/reaction diagnostic is non-finite.")
        minimum_phase = self._global_min(
            float(
                np.min(
                    self.phi.x.array[: self._owned_phase_dofs],
                    initial=math.inf,
                )
            )
        )
        maximum_history = self._global_max(
            float(
                np.max(
                    self.history.x.array[: self._owned_history_dofs],
                    initial=-math.inf,
                )
            )
        )
        crack = self._crack_metrics()
        opening = self.config.max_displacement * pseudo_time
        sliding = self.config.max_sliding_displacement * pseudo_time
        return {
            "step": step,
            "pseudo_time": pseudo_time,
            "displacement": opening,
            "reaction_force": top_reaction_y,
            "opening_displacement": opening,
            "sliding_displacement": sliding,
            "top_reaction_x": top_reaction_x,
            "top_reaction_y": top_reaction_y,
            "bottom_reaction_x": bottom_reaction_x,
            "bottom_reaction_y": bottom_reaction_y,
            "force_balance_x": top_reaction_x + bottom_reaction_x,
            "force_balance_y": top_reaction_y + bottom_reaction_y,
            "elastic_energy": elastic,
            "fracture_energy": fracture,
            "instantaneous_internal_energy": elastic + fracture,
            "minimum_phase": minimum_phase,
            "maximum_history": maximum_history,
            "damage_integral": damage_integral,
            "crack_front_x": crack.front_x,
            "crack_extension": crack.extension_x,
            "crack_tip_x": crack.tip_x,
            "crack_tip_y": crack.tip_y,
            "crack_path_length": crack.path_length,
            "crack_kink_angle_degrees": crack.kink_angle_degrees,
            "connected_crack_node_count": crack.connected_node_count,
            "staggered_iterations": staggered_iterations,
            "newton_iterations_last": newton_report.iterations,
            "newton_residual_inf": newton_report.residual_norm,
            "newton_increment_inf": newton_report.increment_norm,
            "boundary_violation_inf": newton_report.boundary_violation,
            "phase_increment_inf": phase_report.increment_norm,
            "phase_kkt_residual_inf": phase_report.kkt_residual,
            "phase_optimizer_iterations": phase_report.optimizer_iterations,
            "active_lower_constraints": phase_report.active_lower_constraints,
            "active_irreversibility_constraints": (
                phase_report.active_irreversibility_constraints
            ),
            "free_mechanical_residual_inf": free_mechanical_residual,
            "history_increment_inf": history_increment,
            "load_increment": load_increment,
            "cutbacks_before_step": cutbacks_before_step,
            "cumulative_active_lower_visits": self._active_lower_visits,
            "cumulative_active_irreversibility_visits": (
                self._active_irreversibility_visits
            ),
        }

    def _write_manifest(self) -> None:
        manifest = {
            "model": "corrected AT2 spectral-split phase-field fracture",
            "model_version": MODEL_VERSION,
            "petsc_backend_version": PETSC_BACKEND_VERSION,
            "backend": "distributed PETSc SNES/SNESVI",
            "phase_convention": "phi=1 intact, phi=0 broken",
            "unit_system": "kN-mm with unit out-of-plane thickness",
            "mpi_ranks": self.size,
            "config": asdict(self.config),
            "material": {
                "specification": self.material_spec.to_dict(),
                "ranges": self.material_ranges,
                "region_cell_counts": self.material_region_cell_counts,
            },
            "petsc_runtime_options": {
                "displacement_snes_type": self._u_snes.getType(),
                "displacement_ksp_type": self._u_snes.getKSP().getType(),
                "displacement_pc_type": self._u_snes.getKSP().getPC().getType(),
                "phase_snes_type": self._phase_snes.getType(),
                "phase_ksp_type": self._phase_snes.getKSP().getType(),
                "phase_pc_type": self._phase_snes.getKSP().getPC().getType(),
            },
            "derived": {
                "lambda_range": self.material_ranges["lambda"],
                "mu_range": self.material_ranges["mu"],
                "grid_spacing": self.config.grid_spacing,
                "cell_diameter": self.config.cell_diameter,
                "grid_spacing_over_length_scale": (
                    self.config.grid_spacing
                    / self.material_ranges["length_scale"]["min"]
                ),
                "cell_diameter_over_length_scale": (
                    self.max_cell_diameter_over_length_scale
                ),
                "represented_notch_length": self.represented_notch_length,
                "crack_front_threshold": CRACK_FRONT_THRESHOLD,
                "relative_opening_at_full_load": self.config.max_displacement,
                "relative_sliding_at_full_load": (
                    self.config.max_sliding_displacement
                ),
                "cells": int(
                    self.domain.topology.index_map(
                        self.domain.topology.dim
                    ).size_global
                ),
                "displacement_dofs": int(
                    self.V_u.dofmap.index_map.size_global
                    * self.V_u.dofmap.index_map_bs
                ),
                "phase_dofs": int(self.V_phi.dofmap.index_map.size_global),
                "history_dofs": int(self.V_H.dofmap.index_map.size_global),
            },
            "software": {
                "dolfinx": dolfinx.__version__,
                "ffcx": ffcx.__version__,
                "basix": basix.__version__,
                "ufl": ufl.__version__,
                "petsc": ".".join(str(value) for value in PETSc.Sys.getVersion()),
                "petsc4py": petsc4py.__version__,
                "mpi4py": mpi4py.__version__,
                "mpi_library": MPI.Get_library_version().strip(),
                "numpy": np.__version__,
                "python": sys.version.split()[0],
            },
        }

        def write() -> None:
            (self.output_dir / "run_manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )

        self._root_action("run-manifest write", write)

    def _open_field_outputs(self) -> tuple[Any | None, Any | None]:
        if not self.config.write_xdmf:
            return None, None
        displacement_file = io.XDMFFile(
            self.comm, self.output_dir / "displacement.xdmf", "w"
        )
        phase_file = io.XDMFFile(
            self.comm, self.output_dir / "phase_field.xdmf", "w"
        )
        displacement_file.write_mesh(self.domain)
        phase_file.write_mesh(self.domain)
        return displacement_file, phase_file

    @staticmethod
    def _close_field_outputs(
        displacement_file: Any | None, phase_file: Any | None
    ) -> None:
        # Close phase first so a later displacement-close failure cannot leave
        # the second collective file open on the success path.
        if phase_file is not None:
            phase_file.close()
        if displacement_file is not None:
            displacement_file.close()

    def _collective_failure(self, error: Exception | None) -> str | None:
        local = (
            None
            if error is None
            else f"rank {self.rank}: {type(error).__name__}: {error}"
        )
        failures = self.comm.allgather(local)
        return next((item for item in failures if item is not None), None)

    def _collective_stage(
        self, label: str, action: Callable[[], Any]
    ) -> Any:
        result: Any = None
        local_error: str | None = None
        try:
            result = action()
        except RecoverableStepError as error:
            local_error = f"rank {self.rank}: {type(error).__name__}: {error}"
        except Exception as error:
            _abort_mpi_job(self.comm, label, error)
        failures = self.comm.allgather(local_error)
        failure = next((item for item in failures if item is not None), None)
        if failure is not None:
            raise RuntimeError(f"{label} failed collectively: {failure}")
        return result

    def run(self) -> dict[str, Any]:
        c = self.config
        displacement_file = None
        phase_file = None
        total_cutbacks = 0
        try:
            displacement_file, phase_file = self._collective_stage(
                "field-output setup", self._open_field_outputs
            )
            initial_newton = self._collective_stage(
                "initial boundary check",
                lambda: NewtonReport(
                    0, 0.0, 0.0, self._boundary_violation()
                ),
            )
            initial = self._collective_stage(
                "initial diagnostics",
                lambda: self._diagnostics(
                    0,
                    0.0,
                    0.0,
                    0,
                    initial_newton,
                    self.initial_phase_report,
                    self._free_mechanical_residual_norm(),
                    0.0,
                ),
            )
            self.records.append(initial)
            self._collective_stage(
                "initial field write",
                lambda: self._write_fields(
                    displacement_file, phase_file, 0.0
                ),
            )
            self._write_csv()
            self._write_summary(total_cutbacks, status="running")

            nominal_increment = 1.0 / c.load_steps
            load_increment = nominal_increment
            pseudo_time = 0.0
            step = 0
            consecutive_cutbacks = 0
            easy_step_streak = 0

            while pseudo_time < 1.0 - 1.0e-14:
                target_time = min(1.0, pseudo_time + load_increment)
                actual_increment = target_time - pseudo_time
                next_step = step + 1
                (
                    saved_u,
                    saved_phi,
                    saved_history,
                    saved_active_lower_visits,
                    saved_active_irreversibility_visits,
                ) = self._collective_stage(
                    "trial-state snapshot",
                    lambda: (
                        self.u.x.array.copy(),
                        self.phi.x.array.copy(),
                        self.history.x.array.copy(),
                        self._active_lower_visits,
                        self._active_irreversibility_visits,
                    ),
                )
                self._set_load_factor(target_time)

                last_newton = NewtonReport(0, math.inf, math.inf, math.inf)
                last_phase = PhaseReport(math.inf, math.inf, 0, 0, 0)
                last_mechanical_residual = math.inf
                last_history_increment = math.inf
                converged = False
                local_failure: Exception | None = None
                try:
                    for staggered_iteration in range(
                        1, c.max_staggered_iterations + 1
                    ):
                        last_newton = self._collective_stage(
                            "displacement solve", self._solve_displacement
                        )
                        last_history_increment = self._collective_stage(
                            "history update",
                            lambda: self._update_trial_history(saved_history),
                        )
                        last_phase = self._collective_stage(
                            "phase variational-inequality solve",
                            lambda: self._solve_phase_field(saved_phi),
                        )
                        last_mechanical_residual = self._collective_stage(
                            "mechanical residual check",
                            self._free_mechanical_residual_norm,
                        )
                        if (
                            last_phase.increment_norm <= c.staggered_tolerance
                            and last_phase.kkt_residual <= c.phase_kkt_tolerance
                            and last_mechanical_residual
                            <= c.staggered_mechanical_tolerance
                        ):
                            converged = True
                            break
                except Exception as error:
                    local_failure = error

                failure_text = self._collective_failure(local_failure)
                if failure_text is not None:
                    converged = False

                if not converged:
                    def restore_trial_state() -> None:
                        self.u.x.array[:] = saved_u
                        self.phi.x.array[:] = saved_phi
                        self.history.x.array[:] = saved_history
                        self.u.x.scatter_forward()
                        self.phi.x.scatter_forward()
                        self.history.x.scatter_forward()
                        self._active_lower_visits = saved_active_lower_visits
                        self._active_irreversibility_visits = (
                            saved_active_irreversibility_visits
                        )
                        self._set_load_factor(pseudo_time)

                    self._collective_stage(
                        "load-cutback rollback", restore_trial_state
                    )
                    consecutive_cutbacks += 1
                    total_cutbacks += 1
                    easy_step_streak = 0
                    if consecutive_cutbacks > c.max_load_cutbacks:
                        reason = failure_text or (
                            f"phase increment={last_phase.increment_norm:.3e}, "
                            f"free mechanical residual="
                            f"{last_mechanical_residual:.3e} after "
                            f"{c.max_staggered_iterations} staggered iterations"
                        )
                        raise RuntimeError(
                            "Load cutback limit reached near pseudo-time "
                            f"{target_time:.6f}: {reason}."
                        )
                    load_increment *= c.load_cutback_factor
                    if c.verbose:
                        self._root_log(
                            f"[cutback {consecutive_cutbacks}] retrying from "
                            f"t={pseudo_time:.6f} with dt={load_increment:.6e}"
                        )
                    continue

                step = next_step
                pseudo_time = target_time
                applied_displacement = c.max_displacement * pseudo_time
                record = self._collective_stage(
                    "accepted-step diagnostics",
                    lambda: self._diagnostics(
                        step,
                        pseudo_time,
                        applied_displacement,
                        staggered_iteration,
                        last_newton,
                        last_phase,
                        last_mechanical_residual,
                        last_history_increment,
                        actual_increment,
                        consecutive_cutbacks,
                    ),
                )
                self.records.append(record)
                cutbacks_for_step = consecutive_cutbacks
                consecutive_cutbacks = 0
                if (
                    cutbacks_for_step == 0
                    and staggered_iteration
                    <= c.load_growth_iteration_threshold
                    and load_increment < nominal_increment
                ):
                    easy_step_streak += 1
                else:
                    easy_step_streak = 0
                if easy_step_streak >= c.load_growth_patience:
                    load_increment = min(
                        nominal_increment,
                        load_increment / c.load_cutback_factor,
                    )
                    easy_step_streak = 0

                if (
                    step % c.write_every == 0
                    or pseudo_time >= 1.0 - 1.0e-14
                ):
                    self._collective_stage(
                        "accepted field write",
                        lambda: self._write_fields(
                            displacement_file, phase_file, pseudo_time
                        ),
                    )
                if c.verbose:
                    self._root_log(
                        f"[{step:03d}] t={pseudo_time:.5f}, "
                        f"opening={applied_displacement:.6e}, "
                        f"sliding={record['sliding_displacement']:.6e}, "
                        f"Rn={record['top_reaction_y']:.6e}, "
                        f"Rs={record['top_reaction_x']:.6e}, "
                        f"front={record['crack_front_x']:.4f}, "
                        f"alt={staggered_iteration}, "
                        f"mech={last_mechanical_residual:.2e}"
                    )
        except Exception as error:
            try:
                self._write_csv()
                self._write_summary(
                    total_cutbacks,
                    status="failed",
                    error=f"{type(error).__name__}: {error}",
                )
            except Exception as reporting_error:
                error.add_note(
                    "Failure reporting also failed: "
                    f"{type(reporting_error).__name__}: {reporting_error}"
                )
            raise
        finally:
            self._collective_stage(
                "field-output close",
                lambda: self._close_field_outputs(
                    displacement_file, phase_file
                ),
            )

        try:
            self._write_csv()
            if c.make_plots:
                self._make_plots()
            return self._write_summary(total_cutbacks, status="completed")
        except Exception as error:
            try:
                self._write_summary(
                    total_cutbacks,
                    status="failed",
                    error=f"Post-processing {type(error).__name__}: {error}",
                )
            except Exception as reporting_error:
                error.add_note(
                    "Post-processing failure reporting also failed: "
                    f"{type(reporting_error).__name__}: {reporting_error}"
                )
            raise

    def _write_csv(self) -> None:
        def write() -> None:
            with (self.output_dir / "load_response.csv").open(
                "w", newline="", encoding="utf-8"
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=self.CSV_COLUMNS)
                writer.writeheader()
                writer.writerows(self.records)

        self._root_action("load-response CSV write", write)

    def _make_plots(self) -> None:
        global_phase_values = self._gather_global_phase_values()

        def make() -> None:
            opening = np.asarray(
                [record["opening_displacement"] for record in self.records],
                dtype=float,
            )
            sliding = np.asarray(
                [record["sliding_displacement"] for record in self.records],
                dtype=float,
            )
            normal_reaction = np.asarray(
                [record["top_reaction_y"] for record in self.records],
                dtype=float,
            )
            sliding_reaction = np.asarray(
                [record["top_reaction_x"] for record in self.records],
                dtype=float,
            )
            plt.style.use("seaborn-v0_8-whitegrid")
            mixed = np.max(np.abs(sliding), initial=0.0) > 0.0
            fig, axes = plt.subplots(
                1,
                2 if mixed else 1,
                figsize=(11.8 if mixed else 7.2, 4.6),
                constrained_layout=True,
            )
            normal_axis = np.atleast_1d(axes)[0]
            normal_axis.plot(
                opening, normal_reaction, color="#0f766e", linewidth=2.3
            )
            normal_axis.fill_between(
                opening, normal_reaction, color="#14b8a6", alpha=0.12
            )
            normal_axis.set_xlabel("Relative opening [mm]")
            normal_axis.set_ylabel("Top normal reaction [kN]")
            normal_axis.set_title(f"Opening response ({self.size} MPI ranks)")
            normal_axis.spines[["top", "right"]].set_visible(False)
            if mixed:
                shear_axis = np.atleast_1d(axes)[1]
                shear_axis.plot(
                    sliding, sliding_reaction, color="#7c3aed", linewidth=2.3
                )
                shear_axis.fill_between(
                    sliding, sliding_reaction, color="#8b5cf6", alpha=0.12
                )
                shear_axis.set_xlabel("Relative sliding [mm]")
                shear_axis.set_ylabel("Top sliding reaction [kN]")
                shear_axis.set_title(f"Sliding response ({self.size} MPI ranks)")
                shear_axis.spines[["top", "right"]].set_visible(False)
            fig.savefig(
                self.output_dir / "load_displacement.png", dpi=180
            )
            plt.close(fig)

            assert global_phase_values is not None
            assert self._root_phase_coordinates is not None
            assert self._root_phase_cells is not None
            coordinates = self._root_phase_coordinates
            triangulation = mtri.Triangulation(
                coordinates[:, 0],
                coordinates[:, 1],
                self._root_phase_cells,
            )
            fig, ax = plt.subplots(
                figsize=(6.4, 5.4), constrained_layout=True
            )
            contour = ax.tricontourf(
                triangulation,
                global_phase_values,
                levels=np.linspace(0.0, 1.0, 21),
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            ax.set_aspect("equal")
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
            ax.set_title("Final intactness field, phi")
            colorbar = fig.colorbar(contour, ax=ax, shrink=0.86)
            colorbar.set_label("phi (0 broken, 1 intact)")
            fig.savefig(
                self.output_dir / "final_phase_field.png", dpi=180
            )
            plt.close(fig)

        self._root_action("plot generation", make)

    def _write_summary(
        self,
        total_cutbacks: int,
        status: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        if self.records:
            final: dict[str, float | int] | None = self.records[-1]
            reactions = np.asarray(
                [record["reaction_force"] for record in self.records],
                dtype=float,
            )
            peak_index = int(np.argmax(reactions))
            peak_reaction: float | None = float(reactions[peak_index])
            peak_step: int | None = int(self.records[peak_index]["step"])
            steps_completed = len(self.records) - 1
        else:
            final = None
            peak_reaction = None
            peak_step = None
            steps_completed = 0
        peak_sliding_reaction = (
            float(
                np.max(
                    np.abs(
                        np.asarray(
                            [record["top_reaction_x"] for record in self.records],
                            dtype=float,
                        )
                    ),
                    initial=0.0,
                )
            )
            if self.records
            else None
        )
        elapsed = self._global_max(time.perf_counter() - self._start_time)
        summary: dict[str, Any] = {
            "status": status,
            "backend": "distributed PETSc SNES/SNESVI",
            "mpi_ranks": self.size,
            "elapsed_seconds": elapsed,
            "output_directory": str(self.output_dir),
            "steps_completed": steps_completed,
            "nominal_load_steps": self.config.load_steps,
            "total_load_cutbacks": total_cutbacks,
            "peak_reaction_force": peak_reaction,
            "peak_reaction_step": peak_step,
            "peak_absolute_sliding_reaction": peak_sliding_reaction,
            "final": final,
            "material_ranges": self.material_ranges,
            "cell_diameter_over_length_scale": (
                self.max_cell_diameter_over_length_scale
            ),
            "h_over_length_scale": self.max_cell_diameter_over_length_scale,
            "cumulative_active_lower_visits": self._active_lower_visits,
            "cumulative_active_irreversibility_visits": (
                self._active_irreversibility_visits
            ),
        }
        if error is not None:
            summary["error"] = error

        def write() -> dict[str, Any]:
            temporary_path = self.output_dir / "summary.json.tmp"
            temporary_path.write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            temporary_path.replace(self.output_dir / "summary.json")
            return summary

        self._root_action("summary write", write)
        return summary

    def close(self) -> None:
        """Collectively release low-level PETSc objects owned by this backend."""

        if getattr(self, "_petsc_closed", False):
            return
        for name in ("_u_snes", "_phase_snes"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.destroy()
        for name in (
            "_u_residual",
            "_phase_rhs_ghosted",
            "_phase_rhs",
            "_phase_residual",
        ):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.destroy()
        for name in ("_u_matrix", "_phase_matrix"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.destroy()
        self._petsc_closed = True


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Distributed PETSc/MPI corrected AT2 phase-field fracture "
            "simulation for Linux clusters."
        )
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run a small MPI smoke case."
    )
    parser.add_argument(
        "-in",
        "--input-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Read LAMMPS-style keyword/value settings from PATH on rank 0.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--ny", type=int, default=None)
    parser.add_argument("--length-scale", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max-displacement", type=float, default=None)
    parser.add_argument("--max-sliding-displacement", type=float, default=None)
    parser.add_argument(
        "--mechanical-bc-scheme",
        choices=("legacy_roller_pin", "relative_clamped", "symmetric_clamped"),
        default=None,
    )
    parser.add_argument(
        "--material-mode",
        choices=("uniform", "linear_x", "linear_y", "file"),
        default=None,
    )
    parser.add_argument("--material-file", type=str, default=None)
    parser.add_argument("--max-staggered", type=int, default=None)
    parser.add_argument("--plane-stress", action="store_true")
    parser.add_argument("--no-xdmf", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--petsc-monitor", action="store_true")
    parser.add_argument("--u-ksp-type", type=str, default=None)
    parser.add_argument("--u-pc-type", type=str, default=None)
    parser.add_argument("--phase-ksp-type", type=str, default=None)
    parser.add_argument("--phase-pc-type", type=str, default=None)
    parser.add_argument(
        "--allow-underresolved-mesh", action="store_true"
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> PetscSimulationConfig:
    config = quick_petsc_config() if args.quick else PetscSimulationConfig()
    if args.input_file is not None:
        config = parse_input_file(args.input_file, config)

    updates: dict[str, Any] = {}
    option_map = {
        "output_dir": "output_directory",
        "nx": "nx",
        "ny": "ny",
        "length_scale": "length_scale",
        "steps": "load_steps",
        "max_displacement": "max_displacement",
        "max_sliding_displacement": "max_sliding_displacement",
        "mechanical_bc_scheme": "mechanical_bc_scheme",
        "material_mode": "material_mode",
        "material_file": "material_file",
        "max_staggered": "max_staggered_iterations",
        "u_ksp_type": "displacement_ksp_type",
        "u_pc_type": "displacement_pc_type",
        "phase_ksp_type": "phase_ksp_type",
        "phase_pc_type": "phase_pc_type",
    }
    for argument_name, field_name in option_map.items():
        value = getattr(args, argument_name)
        if value is not None:
            updates[field_name] = value
    if args.plane_stress:
        updates["plane_stress"] = True
    if args.no_xdmf:
        updates["write_xdmf"] = False
    if args.no_plots:
        updates["make_plots"] = False
    if args.quiet:
        updates["verbose"] = False
    if args.petsc_monitor:
        updates["petsc_monitor"] = True
    if args.allow_underresolved_mesh:
        updates["strict_mesh_resolution"] = False
    config = replace(config, **updates)
    if (
        config.material_mode.casefold() == "file"
        and config.material_file.casefold() not in {"", "none"}
    ):
        material_path = Path(config.material_file).expanduser()
        if not material_path.is_absolute():
            if args.input_file is not None and args.material_file is None:
                base_directory = Path(args.input_file).expanduser().resolve().parent
            else:
                base_directory = Path.cwd()
            material_path = base_directory / material_path
        config = replace(config, material_file=str(material_path.resolve()))
    try:
        config.validate()
    except ValueError as error:
        source = (
            f" after reading '{Path(args.input_file).expanduser().resolve()}'"
            if args.input_file is not None
            else ""
        )
        raise InputFileError(f"Invalid configuration{source}: {error}") from error
    return config


def _broadcast_config(
    args: argparse.Namespace, comm: MPI.Intracomm
) -> tuple[PetscSimulationConfig | None, str | None]:
    payload: tuple[dict[str, Any] | None, str | None]
    if comm.rank == 0:
        try:
            payload = (asdict(_config_from_args(args)), None)
        except Exception as error:
            payload = (None, f"{type(error).__name__}: {error}")
    else:
        payload = (None, None)
    data, error_text = comm.bcast(payload, root=0)
    if error_text is not None or data is None:
        return None, error_text or "Configuration broadcast failed."
    return PetscSimulationConfig(**data), None


def run_simulation(
    config: PetscSimulationConfig,
    comm: MPI.Intracomm = MPI.COMM_WORLD,
) -> dict[str, Any]:
    try:
        simulation = PetscPhaseFieldSimulation(config, comm=comm)
    except Exception as error:
        _abort_mpi_job(comm, "simulation construction", error)
    try:
        return simulation.run()
    finally:
        try:
            simulation.close()
        except Exception as error:
            _abort_mpi_job(comm, "PETSc object teardown", error)


def main(argv: Sequence[str] | None = None) -> int:
    comm = MPI.COMM_WORLD
    argument_parser = _parser()
    args = argument_parser.parse_args(argv)
    config, config_error = _broadcast_config(args, comm)
    if config_error is not None or config is None:
        if comm.rank == 0:
            print(f"error: {config_error}", file=sys.stderr)
            print(
                "Use --help for command-line syntax and consult "
                "linux_cluster/README.md for input keywords.",
                file=sys.stderr,
            )
        return 2
    try:
        summary = run_simulation(config, comm=comm)
    except Exception as error:
        failures = comm.allgather(
            f"rank {comm.rank}: {type(error).__name__}: {error}"
        )
        if comm.rank == 0:
            print(f"fatal: {failures[0]}", file=sys.stderr)
        return 1
    if comm.rank == 0:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
