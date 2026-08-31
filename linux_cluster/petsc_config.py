"""Configuration extensions for the Linux PETSc/MPI backend."""

from __future__ import annotations

from dataclasses import dataclass

from phasefield_crack import SimulationConfig


@dataclass(frozen=True)
class PetscSimulationConfig(SimulationConfig):
    """Physical, nonlinear, and PETSc choices for a distributed run.

    The inherited fields are intentionally identical to the Windows/SciPy
    implementation.  The fields below select PETSc algorithms and are also
    accepted by the LAMMPS-style input parser.
    """

    displacement_snes_type: str = "newtonls"
    displacement_ksp_type: str = "gmres"
    displacement_pc_type: str = "gamg"
    phase_snes_type: str = "vinewtonrsls"
    phase_ksp_type: str = "cg"
    phase_pc_type: str = "gamg"
    ksp_relative_tolerance: float = 1.0e-10
    ksp_absolute_tolerance: float = 1.0e-14
    ksp_max_iterations: int = 1000
    petsc_monitor: bool = False

    def validate(self) -> None:
        super().validate()
        solver_names = {
            "displacement_snes_type": self.displacement_snes_type,
            "displacement_ksp_type": self.displacement_ksp_type,
            "displacement_pc_type": self.displacement_pc_type,
            "phase_snes_type": self.phase_snes_type,
            "phase_ksp_type": self.phase_ksp_type,
            "phase_pc_type": self.phase_pc_type,
        }
        for name, value in solver_names.items():
            if not value.strip():
                raise ValueError(f"{name} must be a nonempty PETSc type name.")
        if self.phase_snes_type.casefold() not in {
            "vinewtonrsls",
            "vinewtonssls",
        }:
            raise ValueError(
                "phase_snes_type must be 'vinewtonrsls' or "
                "'vinewtonssls' so the irreversibility bounds are enforced."
            )
        if (
            self.phase_snes_type.casefold() == "vinewtonssls"
            and self.phase_ksp_type.casefold() == "cg"
        ):
            raise ValueError(
                "phase_ksp_type cannot be 'cg' with vinewtonssls; use gmres "
                "because the semismooth Jacobian is not guaranteed SPD."
            )
        if self.ksp_relative_tolerance <= 0.0:
            raise ValueError("ksp_relative_tolerance must be positive.")
        if self.ksp_absolute_tolerance <= 0.0:
            raise ValueError("ksp_absolute_tolerance must be positive.")
        if self.ksp_max_iterations < 1:
            raise ValueError("ksp_max_iterations must be positive.")


def quick_petsc_config(
    output_directory: str = "results/linux_petsc_quick",
) -> PetscSimulationConfig:
    """Small distributed smoke configuration."""

    return PetscSimulationConfig(
        nx=20,
        ny=20,
        length_scale=0.150,
        length_scale_end=0.150,
        max_displacement=5.0e-4,
        load_steps=3,
        max_staggered_iterations=12,
        output_directory=output_directory,
        write_every=1,
        make_plots=False,
    )
