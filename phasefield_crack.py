"""Corrected AT2 phase-field fracture benchmark implemented with FEniCSx.

The source PDF uses the intactness convention ``phi=1`` for intact material and
``phi=0`` for fully broken material.  This implementation follows its weak phase
equation (Eq. 5), completes the missing AT2 definitions, and adds convergence,
bounds, irreversibility, diagnostics, and a reproducible notched-tension test.

Native Windows DOLFINx has no PETSc backend.  Matrices are therefore assembled
with DOLFINx and solved in serial with SciPy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from mpi4py import MPI
import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import scipy
import ufl

import basix
import dolfinx
import ffcx
from dolfinx import fem, io, la, mesh as dmesh, plot

from phasefield_input import InputFileError, parse_input_file
from phasefield_crack_metrics import CrackMetrics, compute_crack_metrics
from phasefield_material import (
    MATERIAL_PROPERTIES,
    MaterialSpec,
    Profile,
    parse_material_file,
    uniform_material_spec,
)


MODEL_VERSION = "1.1.0"
CRACK_FRONT_THRESHOLD = 0.2


@dataclass(frozen=True)
class SimulationConfig:
    """All physical and numerical choices for the demonstration benchmark.

    The default units are kN and mm, with unit out-of-plane thickness.
    """

    # Geometry and mesh
    width: float = 1.0
    height: float = 1.0
    notch_length: float = 0.20
    nx: int = 100
    ny: int = 100

    # Isotropic elastic and AT2 material data (kN-mm unit system)
    young_modulus: float = 210.0
    poisson_ratio: float = 0.30
    fracture_toughness: float = 2.7e-3
    length_scale: float = 0.030
    residual_stiffness: float = 1.0e-8
    plane_stress: bool = False

    # Spatial material coefficients. The scalar values above are the uniform
    # values, or the x=0/y=0 endpoints for the built-in linear profiles.
    material_mode: str = "uniform"
    young_modulus_end: float = 210.0
    poisson_ratio_end: float = 0.30
    fracture_toughness_end: float = 2.7e-3
    length_scale_end: float = 0.030
    material_file: str = "none"

    # Displacement-controlled loading
    mechanical_bc_scheme: str = "legacy_roller_pin"
    max_displacement: float = 0.015
    max_sliding_displacement: float = 0.0
    load_steps: int = 60

    # Nonlinear and staggered solution controls
    max_newton_iterations: int = 25
    newton_relative_tolerance: float = 1.0e-8
    newton_absolute_tolerance: float = 1.0e-10
    newton_increment_tolerance: float = 1.0e-10
    max_staggered_iterations: int = 250
    staggered_tolerance: float = 1.0e-5
    staggered_mechanical_tolerance: float = 1.0e-6
    phase_kkt_tolerance: float = 1.0e-8
    phase_optimizer_max_iterations: int = 200
    spectral_smoothing: float = 1.0e-10
    max_load_cutbacks: int = 8
    load_cutback_factor: float = 0.5
    load_growth_patience: int = 3
    load_growth_iteration_threshold: int = 10

    # Output and safeguards
    output_directory: str = "results/default"
    write_every: int = 1
    write_xdmf: bool = True
    write_material_fields: bool = True
    make_plots: bool = True
    strict_mesh_resolution: bool = True
    verbose: bool = True

    def validate(self) -> None:
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Specimen width and height must be positive.")
        if self.nx < 2 or self.ny < 2:
            raise ValueError("nx and ny must both be at least 2.")
        if self.ny % 2:
            raise ValueError("ny must be even so the horizontal starter notch aligns with vertices.")
        if not 0.0 < self.notch_length < self.width:
            raise ValueError("notch_length must lie strictly inside the specimen width.")
        if (
            self.young_modulus <= 0.0
            or self.young_modulus_end <= 0.0
            or self.fracture_toughness <= 0.0
            or self.fracture_toughness_end <= 0.0
        ):
            raise ValueError(
                "Young's modulus and fracture toughness endpoints must be positive."
            )
        if not (
            -0.99 < self.poisson_ratio < 0.5
            and -0.99 < self.poisson_ratio_end < 0.5
        ):
            raise ValueError(
                "Poisson-ratio endpoints are outside the admissible constitutive range."
            )
        if self.length_scale <= 0.0 or self.length_scale_end <= 0.0:
            raise ValueError("length_scale endpoints must be positive.")
        material_mode = self.material_mode.casefold()
        if material_mode not in {"uniform", "linear_x", "linear_y", "file"}:
            raise ValueError(
                "material_mode must be uniform, linear_x, linear_y, or file."
            )
        if material_mode == "file" and self.material_file.casefold() in {
            "",
            "none",
        }:
            raise ValueError("material_file must name a file when material_mode=file.")
        if not 0.0 < self.residual_stiffness < 1.0:
            raise ValueError("residual_stiffness must lie strictly between 0 and 1.")
        mechanical_scheme = self.mechanical_bc_scheme.casefold()
        if mechanical_scheme not in {
            "legacy_roller_pin",
            "relative_clamped",
            "symmetric_clamped",
        }:
            raise ValueError(
                "mechanical_bc_scheme must be legacy_roller_pin, "
                "relative_clamped, or symmetric_clamped."
            )
        if (
            mechanical_scheme == "legacy_roller_pin"
            and self.max_sliding_displacement != 0.0
        ):
            raise ValueError(
                "legacy_roller_pin supports opening only; use relative_clamped "
                "or symmetric_clamped for nonzero sliding."
            )
        if not math.isfinite(self.max_sliding_displacement):
            raise ValueError("max_sliding_displacement must be finite.")
        if not math.isfinite(self.max_displacement):
            raise ValueError("max_displacement must be finite.")
        if self.max_displacement < 0.0 or self.load_steps < 1:
            raise ValueError("Loading requires nonnegative max_displacement and at least one step.")
        if self.max_newton_iterations < 1 or self.max_staggered_iterations < 1:
            raise ValueError("Iteration limits must be positive.")
        if self.max_load_cutbacks < 0:
            raise ValueError("max_load_cutbacks cannot be negative.")
        if not 0.0 < self.load_cutback_factor < 1.0:
            raise ValueError("load_cutback_factor must lie strictly between 0 and 1.")
        if self.load_growth_patience < 1 or self.load_growth_iteration_threshold < 1:
            raise ValueError("Load-growth patience and iteration threshold must be positive.")
        if (
            self.newton_relative_tolerance <= 0.0
            or self.newton_absolute_tolerance <= 0.0
            or self.newton_increment_tolerance <= 0.0
        ):
            raise ValueError("Newton tolerances must be positive.")
        if (
            self.staggered_tolerance <= 0.0
            or self.staggered_mechanical_tolerance <= 0.0
            or self.phase_kkt_tolerance <= 0.0
            or self.spectral_smoothing <= 0.0
        ):
            raise ValueError("Staggered, KKT, and smoothing tolerances must be positive.")
        if self.phase_optimizer_max_iterations < 1:
            raise ValueError("phase_optimizer_max_iterations must be positive.")
        if self.write_every < 1:
            raise ValueError("write_every must be at least one.")

        notch_cells = self.notch_length / (self.width / self.nx)
        if not math.isclose(notch_cells, round(notch_cells), abs_tol=1.0e-10):
            raise ValueError(
                "notch_length must align with an x-direction mesh vertex; "
                f"notch_length/dx={notch_cells:.6f}."
            )

        # A file may replace every fallback value, so its actual minimum ell is
        # checked after the DG0 material fields have been constructed.
        minimum_configured_length_scale = (
            min(self.length_scale, self.length_scale_end)
            if material_mode in {"linear_x", "linear_y"}
            else self.length_scale
        )
        ratio = self.cell_diameter / minimum_configured_length_scale
        if (
            material_mode != "file"
            and self.strict_mesh_resolution
            and ratio > 0.5 + 1.0e-12
        ):
            raise ValueError(
                f"Phase field is under-resolved: cell-diameter/ell={ratio:.3f} "
                "exceeds 0.5. "
                "Increase nx/ny, increase ell, or explicitly allow an exploratory mesh."
            )

    @property
    def grid_spacing(self) -> float:
        return max(self.width / self.nx, self.height / self.ny)

    @property
    def cell_diameter(self) -> float:
        """Largest edge of the structured right-triangle mesh."""

        return math.hypot(self.width / self.nx, self.height / self.ny)

    @property
    def mesh_size(self) -> float:
        """Backward-compatible alias for the actual triangular cell diameter."""

        return self.cell_diameter


@dataclass
class NewtonReport:
    iterations: int
    residual_norm: float
    increment_norm: float
    boundary_violation: float


@dataclass
class PhaseReport:
    increment_norm: float
    kkt_residual: float
    optimizer_iterations: int
    active_lower_constraints: int
    active_irreversibility_constraints: int


def lame_parameters(young_modulus: float, poisson_ratio: float, plane_stress: bool) -> tuple[float, float]:
    """Return the effective 2D Lamé parameters."""

    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    if plane_stress:
        lam = young_modulus * poisson_ratio / (1.0 - poisson_ratio**2)
    else:
        lam = young_modulus * poisson_ratio / (
            (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio)
        )
    return lam, mu


def degradation(phi: np.ndarray | float, residual_stiffness: float) -> np.ndarray | float:
    """Normalized degradation with exact intact and residual stiffness limits."""

    return residual_stiffness + (1.0 - residual_stiffness) * np.asarray(phi) ** 2


def spectral_energy_numpy(
    strain: np.ndarray, lam: float, mu: float
) -> tuple[float, float]:
    """Reference spectral split used by constitutive unit tests."""

    eigenvalues = np.linalg.eigvalsh(np.asarray(strain, dtype=float))
    trace = float(np.trace(strain))
    positive = np.maximum(eigenvalues, 0.0)
    negative = np.minimum(eigenvalues, 0.0)
    trace_positive = max(trace, 0.0)
    trace_negative = min(trace, 0.0)
    psi_plus = 0.5 * lam * trace_positive**2 + mu * float(positive @ positive)
    psi_minus = 0.5 * lam * trace_negative**2 + mu * float(negative @ negative)
    return psi_plus, psi_minus


def _macaulay_smooth(value: Any, smoothing: float) -> tuple[Any, Any]:
    root = ufl.sqrt(value * value + smoothing * smoothing)
    return 0.5 * (value + root), 0.5 * (value - root)


def _spectral_energy_ufl(strain: Any, lam: Any, mu: Any, smoothing: float) -> tuple[Any, Any]:
    """Differentiable 2D Miehe-type spectral strain-energy split."""

    trace = ufl.tr(strain)
    discriminant = ufl.sqrt(
        (strain[0, 0] - strain[1, 1]) ** 2 + 4.0 * strain[0, 1] ** 2 + smoothing**2
    )
    eigenvalue_1 = 0.5 * (trace + discriminant)
    eigenvalue_2 = 0.5 * (trace - discriminant)

    trace_positive, trace_negative = _macaulay_smooth(trace, smoothing)
    eig1_positive, eig1_negative = _macaulay_smooth(eigenvalue_1, smoothing)
    eig2_positive, eig2_negative = _macaulay_smooth(eigenvalue_2, smoothing)

    psi_plus = 0.5 * lam * trace_positive**2 + mu * (
        eig1_positive**2 + eig2_positive**2
    )
    psi_minus = 0.5 * lam * trace_negative**2 + mu * (
        eig1_negative**2 + eig2_negative**2
    )
    return psi_plus, psi_minus


class PhaseFieldSimulation:
    """Serial staggered phase-field fracture simulation."""

    CSV_COLUMNS = (
        "step",
        "pseudo_time",
        "displacement",
        "reaction_force",
        "opening_displacement",
        "sliding_displacement",
        "top_reaction_x",
        "top_reaction_y",
        "bottom_reaction_x",
        "bottom_reaction_y",
        "force_balance_x",
        "force_balance_y",
        "elastic_energy",
        "fracture_energy",
        "instantaneous_internal_energy",
        "minimum_phase",
        "maximum_history",
        "damage_integral",
        "crack_front_x",
        "crack_extension",
        "crack_tip_x",
        "crack_tip_y",
        "crack_path_length",
        "crack_kink_angle_degrees",
        "connected_crack_node_count",
        "staggered_iterations",
        "newton_iterations_last",
        "newton_residual_inf",
        "newton_increment_inf",
        "boundary_violation_inf",
        "phase_increment_inf",
        "phase_kkt_residual_inf",
        "phase_optimizer_iterations",
        "active_lower_constraints",
        "active_irreversibility_constraints",
        "free_mechanical_residual_inf",
        "history_increment_inf",
        "load_increment",
        "cutbacks_before_step",
        "cumulative_active_lower_visits",
        "cumulative_active_irreversibility_visits",
    )

    def __init__(self, config: SimulationConfig):
        config.validate()
        if MPI.COMM_WORLD.size != 1:
            raise RuntimeError("The native SciPy backend is serial; run with exactly one MPI rank.")

        self.config = config
        self.output_dir = Path(config.output_directory).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._start_time = time.perf_counter()
        self._active_lower_visits = 0
        self._active_irreversibility_visits = 0
        self.records: list[dict[str, float | int]] = []

        self._load_material_spec()
        self._build_mesh_and_spaces()
        self._build_material_fields()
        self._build_boundary_conditions()
        self._build_forms()
        self._initialize_fields()
        self._write_material_fields()
        self._write_manifest()

    def _build_mesh_and_spaces(self) -> None:
        c = self.config
        self.domain = dmesh.create_rectangle(
            MPI.COMM_WORLD,
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
        self.history_trial = fem.Function(self.V_H, name="current_tensile_energy")

        phase_dof_count = self.V_phi.tabulate_dof_coordinates().shape[0]
        phase_neighbors: list[set[int]] = [set() for _ in range(phase_dof_count)]
        cell_count = self.domain.topology.index_map(self.domain.topology.dim).size_local
        for cell in range(cell_count):
            cell_dofs = [int(dof) for dof in self.V_phi.dofmap.cell_dofs(cell)]
            for dof in cell_dofs:
                phase_neighbors[dof].update(other for other in cell_dofs if other != dof)
        self.phase_neighbors = tuple(
            np.asarray(sorted(neighbors), dtype=np.int32)
            for neighbors in phase_neighbors
        )

    def _load_material_spec(self) -> None:
        """Parse once on rank zero and broadcast a normalized definition."""

        c = self.config
        comm = getattr(self, "comm", MPI.COMM_WORLD)
        defaults = {
            "young_modulus": c.young_modulus,
            "poisson_ratio": c.poisson_ratio,
            "fracture_toughness": c.fracture_toughness,
            "length_scale": c.length_scale,
        }

        payload: tuple[dict[str, Any] | None, str | None]
        if comm.rank == 0:
            try:
                mode = c.material_mode.casefold()
                if mode == "uniform":
                    material_spec = uniform_material_spec(defaults)
                elif mode in {"linear_x", "linear_y"}:
                    end_values = {
                        "young_modulus": c.young_modulus_end,
                        "poisson_ratio": c.poisson_ratio_end,
                        "fracture_toughness": c.fracture_toughness_end,
                        "length_scale": c.length_scale_end,
                    }
                    material_spec = MaterialSpec(
                        profiles={
                            name: Profile(mode, (defaults[name], end_values[name]))
                            for name in MATERIAL_PROPERTIES
                        }
                    )
                else:
                    material_spec = parse_material_file(c.material_file, defaults)
                payload = (material_spec.to_dict(), None)
            except Exception as error:
                payload = (
                    None,
                    f"{type(error).__name__}: material definition is invalid: {error}",
                )
        else:
            payload = (None, None)
        normalized, error_text = comm.bcast(payload, root=0)
        if error_text is not None or normalized is None:
            raise ValueError(error_text or "Material-definition broadcast failed.")
        self.material_spec = MaterialSpec.from_dict(normalized)

    def _build_material_fields(self) -> None:
        """Create partition-independent DG0 material coefficient fields."""

        c = self.config
        comm = getattr(self, "comm", MPI.COMM_WORLD)
        if not hasattr(self, "material_spec"):
            self._load_material_spec()

        coordinates = self.V_H.tabulate_dof_coordinates()
        # Pass the unambiguous DOLFINx callback layout (gdim, point_count),
        # including on partitions that happen to own exactly two cells.
        material_coordinates = coordinates[:, :2].T
        evaluated: dict[str, np.ndarray] | None = None
        local_evaluation_error: str | None = None
        try:
            evaluated = self.material_spec.evaluate(
                material_coordinates, c.width, c.height
            )
        except Exception as error:
            local_evaluation_error = (
                f"rank {comm.rank}: {type(error).__name__}: {error}"
            )
        evaluation_errors = comm.allgather(local_evaluation_error)
        evaluation_failure = next(
            (error for error in evaluation_errors if error is not None), None
        )
        if evaluation_failure is not None or evaluated is None:
            raise ValueError(
                "Material evaluation failed collectively: "
                f"{evaluation_failure or 'unknown rank-local failure'}"
            )
        field_names = {
            "young_modulus": "young_modulus_field",
            "poisson_ratio": "poisson_ratio_field",
            "fracture_toughness": "fracture_toughness_field",
            "length_scale": "length_scale_field",
        }
        owned = self.V_H.dofmap.index_map.size_local
        local_values: dict[str, np.ndarray] = {}
        for property_name, attribute_name in field_names.items():
            values = np.asarray(evaluated[property_name], dtype=float).reshape(-1)
            if values.size != coordinates.shape[0]:
                raise RuntimeError(
                    f"Material profile '{property_name}' produced {values.size} values "
                    f"for {coordinates.shape[0]} DG0 degrees of freedom."
                )
            coefficient = fem.Function(self.V_H, name=property_name)
            coefficient.x.array[:] = values
            coefficient.x.scatter_forward()
            setattr(self, attribute_name, coefficient)
            local_values[property_name] = coefficient.x.array[:owned]

        young = local_values["young_modulus"]
        poisson = local_values["poisson_ratio"]
        toughness = local_values["fracture_toughness"]
        length = local_values["length_scale"]
        local_nonfinite = not all(
            np.all(np.isfinite(values)) for values in local_values.values()
        )
        if comm.allreduce(local_nonfinite, op=MPI.LOR):
            raise ValueError("Material coefficients must be finite on every cell.")
        local_nonpositive = bool(
            np.any(young <= 0.0)
            or np.any(toughness <= 0.0)
            or np.any(length <= 0.0)
        )
        if comm.allreduce(local_nonpositive, op=MPI.LOR):
            raise ValueError("E, Gc, and ell must be positive on every cell.")
        local_bad_poisson = bool(
            np.any((poisson <= -0.99) | (poisson >= 0.5))
        )
        if comm.allreduce(local_bad_poisson, op=MPI.LOR):
            raise ValueError("Poisson ratio must lie in (-0.99, 0.5) on every cell.")

        mu_values = young / (2.0 * (1.0 + poisson))
        if c.plane_stress:
            lambda_values = young * poisson / (1.0 - poisson**2)
        else:
            lambda_values = young * poisson / (
                (1.0 + poisson) * (1.0 - 2.0 * poisson)
            )
        range_values = dict(local_values)
        range_values["lambda"] = lambda_values
        range_values["mu"] = mu_values
        self.material_ranges: dict[str, dict[str, float]] = {}
        for name, values in range_values.items():
            local_minimum = float(np.min(values, initial=math.inf))
            local_maximum = float(np.max(values, initial=-math.inf))
            self.material_ranges[name] = {
                "min": float(comm.allreduce(local_minimum, op=MPI.MIN)),
                "max": float(comm.allreduce(local_maximum, op=MPI.MAX)),
            }

        self.material_region_cell_counts: dict[str, int] = {}
        for name, mask in self.material_spec.region_masks(
            material_coordinates
        ).items():
            local_count = int(np.count_nonzero(np.asarray(mask, dtype=bool)[:owned]))
            global_count = int(comm.allreduce(local_count, op=MPI.SUM))
            if global_count == 0:
                raise ValueError(f"Material region '{name}' contains no mesh cells.")
            self.material_region_cell_counts[name] = global_count

        minimum_length_scale = self.material_ranges["length_scale"]["min"]
        self.max_cell_diameter_over_length_scale = (
            c.cell_diameter / minimum_length_scale
        )
        if (
            c.strict_mesh_resolution
            and self.max_cell_diameter_over_length_scale > 0.5 + 1.0e-12
        ):
            raise ValueError(
                "Phase field is under-resolved for the spatial material: "
                "cell-diameter/min(ell)="
                f"{self.max_cell_diameter_over_length_scale:.3f} exceeds 0.5. "
                "Increase nx/ny, increase the minimum ell, or explicitly allow "
                "an exploratory mesh."
            )

    def _write_material_fields(self) -> None:
        if not self.config.write_material_fields:
            return
        comm = getattr(self, "comm", MPI.COMM_WORLD)
        with io.XDMFFile(
            comm, self.output_dir / "material_fields.xdmf", "w"
        ) as material_file:
            material_file.write_mesh(self.domain)
            for coefficient in (
                self.young_modulus_field,
                self.poisson_ratio_field,
                self.fracture_toughness_field,
                self.length_scale_field,
            ):
                material_file.write_function(coefficient, 0.0)

    def _build_boundary_conditions(self) -> None:
        c = self.config
        fdim = self.domain.topology.dim - 1

        bottom_facets = dmesh.locate_entities_boundary(
            self.domain, fdim, lambda x: np.isclose(x[1], 0.0)
        )
        top_facets = dmesh.locate_entities_boundary(
            self.domain, fdim, lambda x: np.isclose(x[1], c.height)
        )
        left_facets = dmesh.locate_entities_boundary(
            self.domain, fdim, lambda x: np.isclose(x[0], 0.0)
        )
        right_facets = dmesh.locate_entities_boundary(
            self.domain, fdim, lambda x: np.isclose(x[0], c.width)
        )
        pin_vertices = dmesh.locate_entities_boundary(
            self.domain,
            0,
            lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0),
        )

        bottom_x_dofs = fem.locate_dofs_topological(
            self.V_u.sub(0), fdim, bottom_facets
        )
        bottom_y_dofs = fem.locate_dofs_topological(
            self.V_u.sub(1), fdim, bottom_facets
        )
        top_x_dofs = fem.locate_dofs_topological(
            self.V_u.sub(0), fdim, top_facets
        )
        top_y_dofs = fem.locate_dofs_topological(
            self.V_u.sub(1), fdim, top_facets
        )
        pin_x_dofs = fem.locate_dofs_topological(self.V_u.sub(0), 0, pin_vertices)

        comm = getattr(self, "comm", MPI.COMM_WORLD)
        for boundary_name, dofs in (
            ("bottom-x", bottom_x_dofs),
            ("bottom-y", bottom_y_dofs),
            ("top-x", top_x_dofs),
            ("top-y", top_y_dofs),
        ):
            if comm.allreduce(int(dofs.size), op=MPI.SUM) == 0:
                raise RuntimeError(
                    f"No displacement degrees of freedom were found on {boundary_name}."
                )

        self.bottom_x_displacement = fem.Constant(self.domain, np.float64(0.0))
        self.bottom_y_displacement = fem.Constant(self.domain, np.float64(0.0))
        self.top_x_displacement = fem.Constant(self.domain, np.float64(0.0))
        self.top_y_displacement = fem.Constant(self.domain, np.float64(0.0))
        # Compatibility name retained for callers that previously inspected
        # the single top-y loading Constant.
        self.top_displacement = self.top_y_displacement

        scheme = c.mechanical_bc_scheme.casefold()
        if scheme == "legacy_roller_pin":
            constrained_sets = (bottom_y_dofs, top_y_dofs, pin_x_dofs)
            self.displacement_bcs = [
                fem.dirichletbc(
                    np.float64(0.0), bottom_y_dofs, self.V_u.sub(1)
                ),
                fem.dirichletbc(
                    self.top_y_displacement, top_y_dofs, self.V_u.sub(1)
                ),
                fem.dirichletbc(
                    np.float64(0.0), pin_x_dofs, self.V_u.sub(0)
                ),
            ]
            self._load_targets = (
                (self.top_y_displacement, c.max_displacement),
            )
            self._boundary_u_dofs = {
                "bottom_x": np.asarray(pin_x_dofs, dtype=np.int32),
                "bottom_y": np.asarray(bottom_y_dofs, dtype=np.int32),
                "top_x": np.empty(0, dtype=np.int32),
                "top_y": np.asarray(top_y_dofs, dtype=np.int32),
            }
        elif scheme == "relative_clamped":
            constrained_sets = (
                bottom_x_dofs,
                bottom_y_dofs,
                top_x_dofs,
                top_y_dofs,
            )
            self.displacement_bcs = [
                fem.dirichletbc(
                    np.float64(0.0), bottom_x_dofs, self.V_u.sub(0)
                ),
                fem.dirichletbc(
                    np.float64(0.0), bottom_y_dofs, self.V_u.sub(1)
                ),
                fem.dirichletbc(
                    self.top_x_displacement, top_x_dofs, self.V_u.sub(0)
                ),
                fem.dirichletbc(
                    self.top_y_displacement, top_y_dofs, self.V_u.sub(1)
                ),
            ]
            self._load_targets = (
                (self.top_x_displacement, c.max_sliding_displacement),
                (self.top_y_displacement, c.max_displacement),
            )
            self._boundary_u_dofs = {
                "bottom_x": np.asarray(bottom_x_dofs, dtype=np.int32),
                "bottom_y": np.asarray(bottom_y_dofs, dtype=np.int32),
                "top_x": np.asarray(top_x_dofs, dtype=np.int32),
                "top_y": np.asarray(top_y_dofs, dtype=np.int32),
            }
        else:
            constrained_sets = (
                bottom_x_dofs,
                bottom_y_dofs,
                top_x_dofs,
                top_y_dofs,
            )
            self.displacement_bcs = [
                fem.dirichletbc(
                    self.bottom_x_displacement, bottom_x_dofs, self.V_u.sub(0)
                ),
                fem.dirichletbc(
                    self.bottom_y_displacement, bottom_y_dofs, self.V_u.sub(1)
                ),
                fem.dirichletbc(
                    self.top_x_displacement, top_x_dofs, self.V_u.sub(0)
                ),
                fem.dirichletbc(
                    self.top_y_displacement, top_y_dofs, self.V_u.sub(1)
                ),
            ]
            self._load_targets = (
                (self.bottom_x_displacement, -0.5 * c.max_sliding_displacement),
                (self.bottom_y_displacement, -0.5 * c.max_displacement),
                (self.top_x_displacement, 0.5 * c.max_sliding_displacement),
                (self.top_y_displacement, 0.5 * c.max_displacement),
            )
            self._boundary_u_dofs = {
                "bottom_x": np.asarray(bottom_x_dofs, dtype=np.int32),
                "bottom_y": np.asarray(bottom_y_dofs, dtype=np.int32),
                "top_x": np.asarray(top_x_dofs, dtype=np.int32),
                "top_y": np.asarray(top_y_dofs, dtype=np.int32),
            }
        self.constrained_u_dofs = np.unique(
            np.concatenate(constrained_sets)
        ).astype(np.int32)
        self._set_load_factor(0.0)

        notch_vertices = dmesh.locate_entities(
            self.domain,
            0,
            lambda x: np.isclose(x[1], 0.5 * c.height)
            & (x[0] <= c.notch_length + 1.0e-12),
        )
        notch_dofs = fem.locate_dofs_topological(self.V_phi, 0, notch_vertices)
        if comm.allreduce(int(notch_dofs.size), op=MPI.SUM) == 0:
            raise RuntimeError("No phase-field degrees of freedom were found on the starter notch.")
        self.notch_dofs = np.asarray(notch_dofs, dtype=np.int32)
        phase_coordinates = self.V_phi.tabulate_dof_coordinates()
        local_notch_max = (
            float(np.max(phase_coordinates[self.notch_dofs, 0]))
            if self.notch_dofs.size
            else -math.inf
        )
        self.represented_notch_length = float(
            comm.allreduce(local_notch_max, op=MPI.MAX)
        )
        self.represented_notch_tip = (
            self.represented_notch_length,
            0.5 * c.height,
        )
        self.phase_bc = fem.dirichletbc(np.float64(0.0), notch_dofs, self.V_phi)

        facets = np.concatenate(
            (top_facets, bottom_facets, left_facets, right_facets)
        ).astype(np.int32)
        values = np.concatenate(
            (
                np.full(top_facets.size, 1, dtype=np.int32),
                np.full(bottom_facets.size, 2, dtype=np.int32),
                np.full(left_facets.size, 3, dtype=np.int32),
                np.full(right_facets.size, 4, dtype=np.int32),
            )
        )
        order = np.argsort(facets)
        self.facet_tags = dmesh.meshtags(
            self.domain, fdim, facets[order], values[order]
        )
        self.boundary_markers = {"top": 1, "bottom": 2, "left": 3, "right": 4}

    def _set_load_factor(self, load_factor: float) -> None:
        if not 0.0 <= load_factor <= 1.0 + 1.0e-12:
            raise ValueError("load_factor must lie in [0, 1].")
        for constant, final_value in self._load_targets:
            constant.value = np.float64(load_factor * final_value)
        self.current_load_factor = float(load_factor)

    def _build_forms(self) -> None:
        c = self.config
        young = self.young_modulus_field
        poisson = self.poisson_ratio_field
        self.mu = young / (2.0 * (1.0 + poisson))
        if c.plane_stress:
            self.lam = young * poisson / (1.0 - poisson**2)
        else:
            self.lam = young * poisson / (
                (1.0 + poisson) * (1.0 - 2.0 * poisson)
            )

        test_u = ufl.TestFunction(self.V_u)
        trial_du = ufl.TrialFunction(self.V_u)
        strain = ufl.variable(ufl.sym(ufl.grad(self.u)))
        psi_plus, psi_minus = _spectral_energy_ufl(
            strain, self.lam, self.mu, c.spectral_smoothing
        )
        degradation_ufl = c.residual_stiffness + (
            1.0 - c.residual_stiffness
        ) * self.phi**2
        elastic_density = degradation_ufl * psi_plus + psi_minus
        stress = ufl.diff(elastic_density, strain)

        displacement_residual = ufl.inner(stress, ufl.sym(ufl.grad(test_u))) * ufl.dx
        displacement_jacobian = ufl.derivative(
            displacement_residual, self.u, trial_du
        )
        self.form_u_residual = fem.form(displacement_residual)
        self.form_u_jacobian = fem.form(displacement_jacobian)

        trial_phi = ufl.TrialFunction(self.V_phi)
        test_phi = ufl.TestFunction(self.V_phi)
        driving_factor = 2.0 * (1.0 - c.residual_stiffness) * self.history
        toughness = self.fracture_toughness_field
        length_scale = self.length_scale_field
        phase_bilinear = (
            toughness
            * length_scale
            * ufl.inner(ufl.grad(trial_phi), ufl.grad(test_phi))
            + (
                toughness / length_scale + driving_factor
            )
            * trial_phi
            * test_phi
        ) * ufl.dx
        phase_linear = (
            toughness / length_scale * test_phi * ufl.dx
        )
        self.form_phi_bilinear = fem.form(phase_bilinear)
        self.form_phi_linear = fem.form(phase_linear)

        interpolation_points = self.V_H.element.interpolation_points
        self.history_expression = fem.Expression(psi_plus, interpolation_points)

        fracture_density = toughness * (
            0.5 / length_scale * (1.0 - self.phi) ** 2
            + 0.5
            * length_scale
            * ufl.inner(ufl.grad(self.phi), ufl.grad(self.phi))
        )
        self.elastic_energy_form = fem.form(elastic_density * ufl.dx)
        self.fracture_energy_form = fem.form(fracture_density * ufl.dx)
        self.damage_integral_form = fem.form((1.0 - self.phi) * ufl.dx)

        ds = ufl.Measure("ds", domain=self.domain, subdomain_data=self.facet_tags)
        normal = ufl.FacetNormal(self.domain)
        traction = ufl.dot(stress, normal)
        self.top_reaction_x_form = fem.form(traction[0] * ds(1))
        self.top_reaction_y_form = fem.form(traction[1] * ds(1))
        self.bottom_reaction_x_form = fem.form(traction[0] * ds(2))
        self.bottom_reaction_y_form = fem.form(traction[1] * ds(2))
        # Backward-compatible name: the historical response was top-y only.
        self.reaction_form = self.top_reaction_y_form

    def _initialize_fields(self) -> None:
        self.u.x.array[:] = 0.0
        self.phi.x.array[:] = 1.0
        self.history.x.array[:] = 0.0
        self.phase_bc.set(self.phi.x.array)
        self.u.x.scatter_forward()
        self.phi.x.scatter_forward()
        self.history.x.scatter_forward()

        # Equilibrate a diffuse starter crack at zero load.  The internal phi=0
        # line is retained as an essential phase constraint throughout the run.
        initial_upper_bound = self.phi.x.array.copy()
        self.initial_phase_report = self._solve_phase_field(initial_upper_bound)

    def _write_manifest(self) -> None:
        manifest = {
            "model": "corrected AT2 spectral-split phase-field fracture",
            "model_version": MODEL_VERSION,
            "phase_convention": "phi=1 intact, phi=0 broken",
            "unit_system": "kN-mm with unit out-of-plane thickness",
            "config": asdict(self.config),
            "material": {
                "specification": self.material_spec.to_dict(),
                "ranges": self.material_ranges,
                "region_cell_counts": self.material_region_cell_counts,
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
                "h_over_length_scale": self.max_cell_diameter_over_length_scale,
                "represented_notch_length": self.represented_notch_length,
                "crack_front_threshold": CRACK_FRONT_THRESHOLD,
                "relative_opening_at_full_load": self.config.max_displacement,
                "relative_sliding_at_full_load": (
                    self.config.max_sliding_displacement
                ),
                "cells": int(
                    self.domain.topology.index_map(self.domain.topology.dim).size_global
                ),
                "displacement_dofs": int(
                    self.V_u.dofmap.index_map.size_global * self.V_u.dofmap.index_map_bs
                ),
                "phase_dofs": int(self.V_phi.dofmap.index_map.size_global),
                "history_dofs": int(self.V_H.dofmap.index_map.size_global),
            },
            "software": {
                "dolfinx": dolfinx.__version__,
                "ffcx": ffcx.__version__,
                "basix": basix.__version__,
                "ufl": ufl.__version__,
                "scipy": scipy.__version__,
                "numpy": np.__version__,
                "python": sys.version.split()[0],
            },
        }
        (self.output_dir / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _direct_solve(matrix: Any, right_hand_side: np.ndarray) -> np.ndarray:
        # Vector-valued spaces are exposed as BSR matrices by DOLFINx.  SciPy's
        # direct solver accepts them after an explicit CSR conversion.
        matrix = matrix.tocsr()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                solution = spsolve(matrix, right_hand_side)
        except MatrixRankWarning as error:
            raise RuntimeError("Sparse matrix is singular or rank deficient.") from error
        solution = np.asarray(solution, dtype=float)
        if not np.all(np.isfinite(solution)):
            raise RuntimeError("Sparse solve returned non-finite values.")
        return solution

    def _boundary_violation(self) -> float:
        correction = np.zeros_like(self.u.x.array)
        for boundary_condition in self.displacement_bcs:
            boundary_condition.set(correction, self.u.x.array, alpha=1.0)
        return float(
            np.max(np.abs(correction[self.constrained_u_dofs]), initial=0.0)
        )

    def _free_mechanical_residual_norm(self) -> float:
        residual = fem.assemble_vector(self.form_u_residual)
        residual.scatter_reverse(la.InsertMode.add)
        values = residual.array.copy()
        values[self.constrained_u_dofs] = 0.0
        return float(np.max(np.abs(values), initial=0.0))

    def _reaction_components(self) -> dict[str, float]:
        """Sum the assembled nodal residual on each grip and component."""

        residual = fem.assemble_vector(self.form_u_residual)
        residual.scatter_reverse(la.InsertMode.add)
        values = residual.array
        reactions = {
            name: float(np.sum(values[dofs], dtype=float))
            for name, dofs in self._boundary_u_dofs.items()
        }
        if not all(math.isfinite(value) for value in reactions.values()):
            raise RuntimeError("A grip reaction is non-finite.")
        return reactions

    def _solve_displacement(self) -> NewtonReport:
        c = self.config
        residual_reference: float | None = None
        last_increment = math.inf
        last_residual = math.inf
        last_boundary_violation = math.inf

        for linear_solves in range(c.max_newton_iterations + 1):
            residual_vector = fem.assemble_vector(self.form_u_residual)
            residual_vector.scatter_reverse(la.InsertMode.add)
            free_residual = residual_vector.array.copy()
            free_residual[self.constrained_u_dofs] = 0.0
            last_residual = float(np.max(np.abs(free_residual), initial=0.0))
            last_boundary_violation = self._boundary_violation()
            if (
                residual_reference is None
                and last_boundary_violation <= c.newton_increment_tolerance
            ):
                residual_reference = max(
                    last_residual, c.newton_absolute_tolerance
                )
            residual_scale = (
                residual_reference
                if residual_reference is not None
                else max(last_residual, c.newton_absolute_tolerance)
            )
            residual_limit = (
                c.newton_absolute_tolerance
                + c.newton_relative_tolerance * residual_scale
            )

            increment_is_small = (
                linear_solves == 0 or last_increment <= c.newton_increment_tolerance
            )
            if (
                last_residual <= residual_limit
                and last_boundary_violation <= c.newton_increment_tolerance
                and increment_is_small
            ):
                return NewtonReport(
                    linear_solves,
                    last_residual,
                    0.0 if linear_solves == 0 else last_increment,
                    last_boundary_violation,
                )

            if linear_solves == c.max_newton_iterations:
                break

            native_matrix = fem.assemble_matrix(
                self.form_u_jacobian, bcs=self.displacement_bcs
            )
            native_matrix.scatter_reverse()
            matrix = native_matrix.to_scipy()

            residual_vector.array[:] *= -1.0
            fem.apply_lifting(
                residual_vector.array,
                [self.form_u_jacobian],
                bcs=[self.displacement_bcs],
                x0=[self.u.x.array],
                alpha=1.0,
            )
            residual_vector.scatter_reverse(la.InsertMode.add)
            for boundary_condition in self.displacement_bcs:
                boundary_condition.set(
                    residual_vector.array, self.u.x.array, alpha=1.0
                )

            increment = self._direct_solve(matrix, residual_vector.array)
            last_increment = float(np.linalg.norm(increment, ord=np.inf))
            self.u.x.array[:] += increment
            self.u.x.scatter_forward()

        raise RuntimeError(
            "Displacement Newton solve failed to converge: "
            f"residual={last_residual:.3e}, increment={last_increment:.3e}, "
            f"BC violation={last_boundary_violation:.3e}, "
            f"limit={c.max_newton_iterations}."
        )

    def _update_trial_history(self, accepted_history: np.ndarray) -> float:
        """Rebuild trial history from the immutable accepted load state."""

        self.history_trial.interpolate(self.history_expression)
        candidate = self.history_trial.x.array
        self.history.x.array[:] = np.maximum(accepted_history, candidate)
        self.history.x.scatter_forward()
        if np.any(self.history.x.array < accepted_history - 1.0e-13):
            raise RuntimeError("History irreversibility was violated.")
        return float(
            np.max(self.history.x.array - accepted_history, initial=0.0)
        )

    @staticmethod
    def _projected_gradient(
        matrix: Any,
        right_hand_side: np.ndarray,
        values: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        gradient = np.asarray(matrix @ values - right_hand_side, dtype=float)
        projected = gradient.copy()
        scale = 100.0 * np.finfo(float).eps
        fixed = upper - lower <= scale
        at_lower = values <= lower + scale
        at_upper = values >= upper - scale
        projected[fixed] = 0.0
        projected[at_lower & (gradient > 0.0)] = 0.0
        projected[at_upper & (gradient < 0.0)] = 0.0
        return gradient, float(np.max(np.abs(projected), initial=0.0))

    def _solve_phase_field(self, upper_bound: np.ndarray) -> PhaseReport:
        """Solve the convex phase subproblem with exact box constraints."""

        c = self.config
        native_matrix = fem.assemble_matrix(
            self.form_phi_bilinear, bcs=[self.phase_bc]
        )
        native_matrix.scatter_reverse()
        matrix = native_matrix.to_scipy().tocsr()
        right_hand_side = fem.assemble_vector(self.form_phi_linear)
        fem.apply_lifting(
            right_hand_side.array,
            [self.form_phi_bilinear],
            bcs=[[self.phase_bc]],
        )
        right_hand_side.scatter_reverse(la.InsertMode.add)
        self.phase_bc.set(right_hand_side.array)

        lower = np.zeros_like(self.phi.x.array)
        upper = np.clip(np.asarray(upper_bound, dtype=float), 0.0, 1.0)
        upper[self.notch_dofs] = 0.0
        unconstrained = self._direct_solve(matrix, right_hand_side.array)
        candidate = np.clip(unconstrained, lower, upper)

        _, kkt_residual = self._projected_gradient(
            matrix, right_hand_side.array, candidate, lower, upper
        )
        optimizer_iterations = 0
        if kkt_residual > c.phase_kkt_tolerance:
            def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
                matrix_values = matrix @ values
                value = 0.5 * float(values @ matrix_values) - float(
                    right_hand_side.array @ values
                )
                gradient = np.asarray(matrix_values - right_hand_side.array, dtype=float)
                return value, gradient

            result = minimize(
                objective,
                candidate,
                method="L-BFGS-B",
                jac=True,
                bounds=Bounds(lower, upper),
                options={
                    "maxiter": c.phase_optimizer_max_iterations,
                    "ftol": 1.0e-15,
                    "gtol": min(0.1 * c.phase_kkt_tolerance, 1.0e-10),
                    "maxls": 40,
                },
            )
            candidate = np.clip(np.asarray(result.x, dtype=float), lower, upper)
            optimizer_iterations = int(result.nit)
            _, kkt_residual = self._projected_gradient(
                matrix, right_hand_side.array, candidate, lower, upper
            )
            if kkt_residual > c.phase_kkt_tolerance:
                raise RuntimeError(
                    "Bound-constrained phase solve failed its KKT check: "
                    f"projected residual={kkt_residual:.3e}, "
                    f"optimizer status={result.message!s}."
                )

        active_tolerance = 1.0e-10
        movable = upper - lower > active_tolerance
        active_lower = int(
            np.count_nonzero(movable & (candidate <= lower + active_tolerance))
        )
        active_upper = int(
            np.count_nonzero(movable & (candidate >= upper - active_tolerance))
        )
        self._active_lower_visits += active_lower
        self._active_irreversibility_visits += active_upper

        increment = float(
            np.max(np.abs(candidate - self.phi.x.array), initial=0.0)
        )
        self.phi.x.array[:] = candidate
        self.phase_bc.set(self.phi.x.array)
        self.phi.x.scatter_forward()
        return PhaseReport(
            increment,
            kkt_residual,
            optimizer_iterations,
            active_lower,
            active_upper,
        )

    def _crack_metrics(
        self, threshold: float = CRACK_FRONT_THRESHOLD
    ) -> CrackMetrics:
        coordinates = self.V_phi.tabulate_dof_coordinates()
        values = self.phi.x.array[: coordinates.shape[0]]
        return compute_crack_metrics(
            values,
            coordinates,
            self.phase_neighbors,
            self.notch_dofs,
            threshold,
            self.represented_notch_tip,
        )

    def _crack_front(
        self, threshold: float = CRACK_FRONT_THRESHOLD
    ) -> tuple[float, float]:
        """Backward-compatible x-projected crack-front diagnostic."""

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
        elastic = float(fem.assemble_scalar(self.elastic_energy_form))
        fracture = float(fem.assemble_scalar(self.fracture_energy_form))
        reactions = self._reaction_components()
        top_reaction_x = reactions["top_x"]
        top_reaction_y = reactions["top_y"]
        bottom_reaction_x = reactions["bottom_x"]
        bottom_reaction_y = reactions["bottom_y"]
        damage_integral = float(fem.assemble_scalar(self.damage_integral_form))
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
            # This is a physical snapshot based on the instantaneous strain.
            # It is not the history-driven phase subproblem's KKT potential.
            "instantaneous_internal_energy": elastic + fracture,
            "minimum_phase": float(np.min(self.phi.x.array)),
            "maximum_history": float(np.max(self.history.x.array)),
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

    def _open_field_outputs(self) -> tuple[Any | None, Any | None]:
        if not self.config.write_xdmf:
            return None, None
        displacement_file = io.XDMFFile(
            MPI.COMM_WORLD, self.output_dir / "displacement.xdmf", "w"
        )
        phase_file = io.XDMFFile(
            MPI.COMM_WORLD, self.output_dir / "phase_field.xdmf", "w"
        )
        displacement_file.write_mesh(self.domain)
        phase_file.write_mesh(self.domain)
        return displacement_file, phase_file

    def _write_fields(self, displacement_file: Any, phase_file: Any, pseudo_time: float) -> None:
        if displacement_file is not None:
            displacement_file.write_function(self.u, pseudo_time)
            phase_file.write_function(self.phi, pseudo_time)

    def run(self) -> dict[str, Any]:
        c = self.config
        displacement_file = None
        phase_file = None
        total_cutbacks = 0
        try:
            displacement_file, phase_file = self._open_field_outputs()
            initial_newton = NewtonReport(0, 0.0, 0.0, self._boundary_violation())
            initial = self._diagnostics(
                0,
                0.0,
                0.0,
                0,
                initial_newton,
                self.initial_phase_report,
                self._free_mechanical_residual_norm(),
                0.0,
            )
            self.records.append(initial)
            self._write_fields(displacement_file, phase_file, 0.0)
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
                applied_displacement = c.max_displacement * target_time

                saved_u = self.u.x.array.copy()
                saved_phi = self.phi.x.array.copy()
                saved_history = self.history.x.array.copy()
                saved_active_lower_visits = self._active_lower_visits
                saved_active_irreversibility_visits = (
                    self._active_irreversibility_visits
                )
                self._set_load_factor(target_time)

                last_newton = NewtonReport(0, math.inf, math.inf, math.inf)
                last_phase = PhaseReport(math.inf, math.inf, 0, 0, 0)
                last_mechanical_residual = math.inf
                last_history_increment = math.inf
                converged = False
                failure: RuntimeError | None = None
                try:
                    for staggered_iteration in range(1, c.max_staggered_iterations + 1):
                        last_newton = self._solve_displacement()
                        last_history_increment = self._update_trial_history(
                            saved_history
                        )
                        last_phase = self._solve_phase_field(saved_phi)
                        last_mechanical_residual = (
                            self._free_mechanical_residual_norm()
                        )
                        if (
                            last_phase.increment_norm <= c.staggered_tolerance
                            and last_phase.kkt_residual <= c.phase_kkt_tolerance
                            and last_mechanical_residual
                            <= c.staggered_mechanical_tolerance
                        ):
                            converged = True
                            break
                except RuntimeError as error:
                    failure = error

                if not converged:
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

                    consecutive_cutbacks += 1
                    total_cutbacks += 1
                    easy_step_streak = 0
                    if consecutive_cutbacks > c.max_load_cutbacks:
                        reason = str(failure) if failure is not None else (
                            f"phase increment={last_phase.increment_norm:.3e}, "
                            f"free mechanical residual="
                            f"{last_mechanical_residual:.3e} after "
                            f"{c.max_staggered_iterations} staggered iterations"
                        )
                        raise RuntimeError(
                            f"Load cutback limit reached near pseudo-time {target_time:.6f}: "
                            f"{reason}."
                        ) from failure

                    load_increment *= c.load_cutback_factor
                    if c.verbose:
                        print(
                            f"[cutback {consecutive_cutbacks}] retrying from "
                            f"t={pseudo_time:.6f} with dt={load_increment:.6e}"
                        )
                    continue

                step = next_step
                pseudo_time = target_time
                applied_displacement = c.max_displacement * pseudo_time

                record = self._diagnostics(
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
                )
                self.records.append(record)
                cutbacks_for_step = consecutive_cutbacks
                consecutive_cutbacks = 0
                # Hold a reduced increment through the difficult crack-growth
                # regime. Recover one level only after several demonstrably
                # easy accepted solves, avoiding cutback/re-growth chatter.
                if (
                    cutbacks_for_step == 0
                    and staggered_iteration <= c.load_growth_iteration_threshold
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

                if step % c.write_every == 0 or pseudo_time >= 1.0 - 1.0e-14:
                    self._write_fields(
                        displacement_file, phase_file, pseudo_time
                    )

                if c.verbose:
                    print(
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
            if displacement_file is not None:
                displacement_file.close()
                phase_file.close()

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
        with (self.output_dir / "load_response.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(self.records)

    def _make_plots(self) -> None:
        opening = np.asarray(
            [r["opening_displacement"] for r in self.records], dtype=float
        )
        sliding = np.asarray(
            [r["sliding_displacement"] for r in self.records], dtype=float
        )
        normal_reaction = np.asarray(
            [r["top_reaction_y"] for r in self.records], dtype=float
        )
        sliding_reaction = np.asarray(
            [r["top_reaction_x"] for r in self.records], dtype=float
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
        normal_axis.plot(opening, normal_reaction, color="#0f766e", linewidth=2.3)
        normal_axis.fill_between(
            opening, normal_reaction, color="#14b8a6", alpha=0.12
        )
        normal_axis.set_xlabel("Relative opening [mm]")
        normal_axis.set_ylabel("Top normal reaction [kN]")
        normal_axis.set_title("Opening response")
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
            shear_axis.set_title("Sliding response")
            shear_axis.spines[["top", "right"]].set_visible(False)
        fig.savefig(self.output_dir / "load_displacement.png", dpi=180)
        plt.close(fig)

        topology, _cell_types, geometry = plot.vtk_mesh(self.V_phi)
        cells = topology.reshape(-1, 4)[:, 1:]
        triangulation = mtri.Triangulation(geometry[:, 0], geometry[:, 1], cells)
        values = self.phi.x.array[: geometry.shape[0]]

        fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
        contour = ax.tricontourf(
            triangulation,
            values,
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
        fig.savefig(self.output_dir / "final_phase_field.png", dpi=180)
        plt.close(fig)

    def _write_summary(
        self,
        total_cutbacks: int,
        status: str,
        error: str | None = None,
    ) -> dict[str, Any]:
        if self.records:
            final: dict[str, float | int] | None = self.records[-1]
            reactions = np.asarray(
                [r["reaction_force"] for r in self.records], dtype=float
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
                            [r["top_reaction_x"] for r in self.records],
                            dtype=float,
                        )
                    ),
                    initial=0.0,
                )
            )
            if self.records
            else None
        )
        summary: dict[str, Any] = {
            "status": status,
            "elapsed_seconds": time.perf_counter() - self._start_time,
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
        temporary_path = self.output_dir / "summary.json.tmp"
        temporary_path.write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        temporary_path.replace(self.output_dir / "summary.json")
        return summary


def quick_config(output_directory: str = "results/quick") -> SimulationConfig:
    """Small, resolved configuration for installation and CI smoke tests."""

    return SimulationConfig(
        nx=20,
        ny=20,
        length_scale=0.150,
        length_scale_end=0.150,
        max_displacement=5.0e-4,
        load_steps=3,
        max_staggered_iterations=12,
        output_directory=output_directory,
        write_every=1,
    )


def run_simulation(config: SimulationConfig) -> dict[str, Any]:
    return PhaseFieldSimulation(config).run()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Corrected FEniCSx AT2 phase-field fracture simulation."
    )
    parser.add_argument("--quick", action="store_true", help="Run a small installation smoke test.")
    parser.add_argument(
        "-in",
        "--input-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Read LAMMPS-style keyword/value settings from PATH.",
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
    parser.add_argument(
        "--allow-underresolved-mesh",
        action="store_true",
        help="Permit h/ell > 0.5 for exploratory runs.",
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> SimulationConfig:
    config = quick_config() if args.quick else SimulationConfig()
    if args.input_file is not None:
        config = parse_input_file(args.input_file, config)

    updates: dict[str, Any] = {}
    if args.output_dir is not None:
        updates["output_directory"] = args.output_dir
    if args.nx is not None:
        updates["nx"] = args.nx
    if args.ny is not None:
        updates["ny"] = args.ny
    if args.length_scale is not None:
        updates["length_scale"] = args.length_scale
    if args.steps is not None:
        updates["load_steps"] = args.steps
    if args.max_displacement is not None:
        updates["max_displacement"] = args.max_displacement
    if args.max_sliding_displacement is not None:
        updates["max_sliding_displacement"] = args.max_sliding_displacement
    if args.mechanical_bc_scheme is not None:
        updates["mechanical_bc_scheme"] = args.mechanical_bc_scheme
    if args.material_mode is not None:
        updates["material_mode"] = args.material_mode
    if args.material_file is not None:
        updates["material_file"] = args.material_file
    if args.max_staggered is not None:
        updates["max_staggered_iterations"] = args.max_staggered
    if args.plane_stress:
        updates["plane_stress"] = True
    if args.no_xdmf:
        updates["write_xdmf"] = False
    if args.no_plots:
        updates["make_plots"] = False
    if args.quiet:
        updates["verbose"] = False
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
        if args.input_file is not None:
            source = Path(args.input_file).expanduser().resolve()
            raise InputFileError(
                f"Configuration after applying input file '{source}' and "
                f"command-line overrides is invalid: {error}"
            ) from error
        raise InputFileError(
            "Configuration after applying command-line overrides is invalid: "
            f"{error}"
        ) from error
    return config


def main(argv: Sequence[str] | None = None) -> int:
    argument_parser = _parser()
    args = argument_parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except InputFileError as error:
        argument_parser.error(str(error))
    summary = run_simulation(config)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
