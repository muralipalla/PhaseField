import csv
import json
from dataclasses import replace

import numpy as np
import pytest
from mpi4py import MPI
from scipy.sparse import csr_matrix

from dolfinx import io

from phasefield_crack import (
    PhaseFieldSimulation,
    SimulationConfig,
    degradation,
    lame_parameters,
    quick_config,
    run_simulation,
    spectral_energy_numpy,
)


def test_degradation_has_exact_endpoints_and_is_monotone():
    residual = 1.0e-8
    phi = np.linspace(0.0, 1.0, 101)
    values = degradation(phi, residual)
    assert values[0] == pytest.approx(residual)
    assert values[-1] == pytest.approx(1.0)
    assert np.all(np.diff(values) >= 0.0)


def test_lame_parameters_and_spectral_split():
    lam_strain, mu_strain = lame_parameters(210.0, 0.3, plane_stress=False)
    lam_stress, mu_stress = lame_parameters(210.0, 0.3, plane_stress=True)
    assert mu_strain == pytest.approx(mu_stress)
    assert lam_strain > lam_stress > 0.0

    tension = np.array([[1.0e-3, 0.0], [0.0, 0.0]])
    compression = -tension
    tension_plus, tension_minus = spectral_energy_numpy(tension, lam_strain, mu_strain)
    compression_plus, compression_minus = spectral_energy_numpy(
        compression, lam_strain, mu_strain
    )
    assert tension_plus > 0.0
    assert tension_minus == pytest.approx(0.0)
    assert compression_plus == pytest.approx(0.0)
    assert compression_minus > 0.0


def test_underresolved_mesh_is_rejected():
    config = SimulationConfig(nx=10, ny=10, length_scale=0.1)
    with pytest.raises(ValueError, match="under-resolved"):
        config.validate()


def test_notch_must_align_with_mesh_vertices():
    config = SimulationConfig(nx=8, ny=8, length_scale=0.5)
    with pytest.raises(ValueError, match="notch_length must align"):
        config.validate()


@pytest.mark.parametrize("sliding", (1.0e-3, 1.0e-16, -1.0e-16))
def test_legacy_boundary_scheme_rejects_every_nonzero_sliding(sliding):
    config = SimulationConfig(max_sliding_displacement=sliding)
    with pytest.raises(ValueError, match="supports opening only"):
        config.validate()


@pytest.mark.parametrize("opening", (float("nan"), float("inf")))
def test_nonfinite_opening_displacement_is_rejected(opening):
    config = SimulationConfig(max_displacement=opening)
    with pytest.raises(ValueError, match="max_displacement must be finite"):
        config.validate()


def test_symmetric_mixed_mode_updates_all_grip_constants(tmp_path):
    config = replace(
        quick_config(str(tmp_path / "mixed_constants")),
        nx=10,
        ny=10,
        length_scale=0.30,
        mechanical_bc_scheme="symmetric_clamped",
        max_displacement=2.0e-3,
        max_sliding_displacement=-8.0e-4,
        write_xdmf=False,
        write_material_fields=False,
        make_plots=False,
        verbose=False,
    )
    simulation = PhaseFieldSimulation(config)

    simulation._set_load_factor(0.25)

    assert float(simulation.bottom_x_displacement.value) == pytest.approx(1.0e-4)
    assert float(simulation.top_x_displacement.value) == pytest.approx(-1.0e-4)
    assert float(simulation.bottom_y_displacement.value) == pytest.approx(-2.5e-4)
    assert float(simulation.top_y_displacement.value) == pytest.approx(2.5e-4)
    assert len(simulation.displacement_bcs) == 4


def test_relative_mixed_mode_run_has_balanced_vector_reactions(tmp_path):
    config = replace(
        quick_config(str(tmp_path / "mixed_run")),
        nx=10,
        ny=10,
        length_scale=0.30,
        mechanical_bc_scheme="relative_clamped",
        max_displacement=1.0e-4,
        max_sliding_displacement=5.0e-5,
        load_steps=1,
        write_xdmf=False,
        write_material_fields=False,
        make_plots=False,
        verbose=False,
    )

    summary = run_simulation(config)
    final = summary["final"]

    assert summary["status"] == "completed"
    assert abs(final["top_reaction_x"]) > 0.0
    assert abs(final["top_reaction_y"]) > 0.0
    assert abs(final["force_balance_x"]) < 1.0e-7
    assert abs(final["force_balance_y"]) < 1.0e-7


def test_linear_material_mode_populates_dg0_coefficients(tmp_path):
    config = replace(
        quick_config(str(tmp_path / "graded")),
        nx=10,
        ny=10,
        length_scale=0.30,
        material_mode="linear_x",
        young_modulus=100.0,
        young_modulus_end=300.0,
        poisson_ratio_end=0.30,
        fracture_toughness_end=2.7e-3,
        length_scale_end=0.30,
        write_xdmf=False,
        write_material_fields=False,
        make_plots=False,
        verbose=False,
    )
    simulation = PhaseFieldSimulation(config)
    coordinates = simulation.V_H.tabulate_dof_coordinates()
    expected = 100.0 + 200.0 * coordinates[:, 0] / config.width

    np.testing.assert_allclose(
        simulation.young_modulus_field.x.array, expected, rtol=0.0, atol=1.0e-12
    )
    assert simulation.material_ranges["young_modulus"]["min"] > 100.0
    assert simulation.material_ranges["young_modulus"]["max"] < 300.0


def test_material_file_regions_are_applied_and_recorded(tmp_path):
    material_file = tmp_path / "inclusion.material"
    material_file.write_text(
        "\n".join(
            (
                "profile young_modulus constant 200.0",
                "profile length_scale constant 0.30",
                "region soft circle 0.5 0.5 0.25 10",
                "override soft young_modulus 80.0",
            )
        ),
        encoding="utf-8",
    )
    config = replace(
        quick_config(str(tmp_path / "file_material")),
        nx=10,
        ny=10,
        length_scale=0.30,
        material_mode="file",
        material_file=str(material_file),
        write_xdmf=False,
        write_material_fields=False,
        make_plots=False,
        verbose=False,
    )

    simulation = PhaseFieldSimulation(config)

    assert simulation.material_ranges["young_modulus"]["min"] == pytest.approx(80.0)
    assert simulation.material_ranges["young_modulus"]["max"] == pytest.approx(200.0)
    assert simulation.material_region_cell_counts["soft"] > 0
    assert simulation.material_spec.source_path == str(material_file.resolve())


def test_rank_deficiency_is_promoted_to_retryable_runtime_error():
    with pytest.raises(RuntimeError, match="singular or rank deficient"):
        PhaseFieldSimulation._direct_solve(csr_matrix((2, 2)), np.ones(2))


def test_phase_box_solver_exercises_optimizer_and_active_upper_bound(tmp_path):
    config = replace(
        quick_config(str(tmp_path / "phase_qp")),
        nx=10,
        ny=10,
        length_scale=0.30,
        write_xdmf=False,
        make_plots=False,
        verbose=False,
    )
    simulation = PhaseFieldSimulation(config)
    assert simulation._boundary_u_dofs["top_x"].size == 0
    assert simulation._boundary_u_dofs["bottom_x"].size == 1
    accepted_upper_bound = np.full_like(simulation.phi.x.array, 0.65)
    accepted_upper_bound[simulation.notch_dofs] = 0.0

    report = simulation._solve_phase_field(accepted_upper_bound)

    assert report.optimizer_iterations > 0
    assert report.active_irreversibility_constraints > 0
    assert report.kkt_residual <= config.phase_kkt_tolerance
    assert np.all(simulation.phi.x.array <= accepted_upper_bound + 1.0e-12)
    assert np.all(simulation.phi.x.array >= -1.0e-12)


def test_end_to_end_smoke_run_writes_auditable_outputs(tmp_path):
    output_dir = tmp_path / "smoke"
    config = replace(
        quick_config(str(output_dir)),
        nx=10,
        ny=10,
        length_scale=0.30,
        load_steps=2,
        max_displacement=1.0e-4,
        make_plots=False,
        verbose=False,
    )
    summary = run_simulation(config)

    assert summary["status"] == "completed"
    assert summary["steps_completed"] == 2
    assert np.isfinite(summary["peak_reaction_force"])
    assert 0.0 <= summary["final"]["minimum_phase"] <= 1.0
    assert summary["final"]["maximum_history"] >= 0.0
    assert summary["final"]["phase_increment_inf"] <= config.staggered_tolerance
    assert (
        summary["final"]["free_mechanical_residual_inf"]
        <= config.staggered_mechanical_tolerance
    )
    assert summary["final"]["phase_kkt_residual_inf"] <= config.phase_kkt_tolerance
    assert summary["final"]["boundary_violation_inf"] <= config.newton_increment_tolerance

    expected = {
        "run_manifest.json",
        "summary.json",
        "load_response.csv",
        "displacement.xdmf",
        "displacement.h5",
        "phase_field.xdmf",
        "phase_field.h5",
        "material_fields.xdmf",
        "material_fields.h5",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})

    with (output_dir / "load_response.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == config.load_steps + 1
    assert [int(row["step"]) for row in rows] == [0, 1, 2]
    assert float(rows[-1]["pseudo_time"]) == pytest.approx(1.0)
    pseudo_times = np.asarray([float(row["pseudo_time"]) for row in rows])
    assert np.all(np.diff(pseudo_times) > 0.0)
    reactions = np.asarray([float(row["reaction_force"]) for row in rows])
    assert np.all(np.diff(reactions) >= -1.0e-10)
    histories = np.asarray([float(row["maximum_history"]) for row in rows])
    damages = np.asarray([float(row["damage_integral"]) for row in rows])
    assert np.all(np.diff(histories) >= -1.0e-13)
    assert np.all(np.diff(damages) >= -1.0e-12)

    with io.XDMFFile(MPI.COMM_WORLD, output_dir / "displacement.xdmf", "r") as xdmf:
        restored_mesh = xdmf.read_mesh(name="mesh")
    assert restored_mesh.topology.index_map(restored_mesh.topology.dim).size_global > 0

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["software"]["dolfinx"] == "0.11.0"
    assert manifest["derived"]["cell_diameter_over_length_scale"] <= 0.5
    assert manifest["derived"]["history_dofs"] == 2 * config.nx * config.ny
    assert manifest["material"]["ranges"]["young_modulus"] == {
        "min": pytest.approx(config.young_modulus),
        "max": pytest.approx(config.young_modulus),
    }
    assert manifest["material"]["specification"]["source_path"] is None


def test_cutback_rolls_back_trial_fields_and_recovers_increment(tmp_path, monkeypatch):
    config = replace(
        quick_config(str(tmp_path / "cutback")),
        nx=10,
        ny=10,
        length_scale=0.30,
        load_steps=4,
        max_displacement=1.0e-4,
        max_sliding_displacement=5.0e-5,
        mechanical_bc_scheme="symmetric_clamped",
        write_xdmf=False,
        make_plots=False,
        verbose=False,
    )
    simulation = PhaseFieldSimulation(config)
    original_solve = simulation._solve_displacement
    original_set_load_factor = simulation._set_load_factor
    calls = 0
    attempted_load_factors = []

    def track_load_factor(load_factor):
        attempted_load_factors.append(load_factor)
        original_set_load_factor(load_factor)

    def fail_first_trial():
        nonlocal calls
        calls += 1
        if calls == 1:
            simulation.u.x.array[:] = 99.0
            simulation.phi.x.array[:] = 0.123
            simulation.history.x.array[:] = 99.0
            raise RuntimeError("synthetic rejected trial")
        return original_solve()

    monkeypatch.setattr(simulation, "_solve_displacement", fail_first_trial)
    monkeypatch.setattr(simulation, "_set_load_factor", track_load_factor)
    summary = simulation.run()

    assert summary["status"] == "completed"
    assert summary["total_load_cutbacks"] == 1
    assert summary["steps_completed"] == 6
    pseudo_times = [record["pseudo_time"] for record in simulation.records]
    assert pseudo_times == pytest.approx(
        [0.0, 0.125, 0.250, 0.375, 0.500, 0.750, 1.0]
    )
    accepted_increments = np.diff(pseudo_times)
    assert accepted_increments[-1] > accepted_increments[1]
    assert np.max(simulation.u.x.array) < 1.0
    assert np.max(simulation.history.x.array) < 1.0
    assert attempted_load_factors[:3] == pytest.approx([0.25, 0.0, 0.125])


def test_terminal_failure_writes_auditable_partial_summary(tmp_path, monkeypatch):
    output_dir = tmp_path / "failure"
    config = replace(
        quick_config(str(output_dir)),
        nx=10,
        ny=10,
        length_scale=0.30,
        load_steps=1,
        max_load_cutbacks=0,
        write_xdmf=False,
        make_plots=False,
        verbose=False,
    )
    simulation = PhaseFieldSimulation(config)

    def always_fail():
        raise RuntimeError("synthetic terminal failure")

    monkeypatch.setattr(simulation, "_solve_displacement", always_fail)
    with pytest.raises(RuntimeError, match="Load cutback limit reached"):
        simulation.run()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["steps_completed"] == 0
    assert "synthetic terminal failure" in summary["error"]
    assert (output_dir / "load_response.csv").is_file()
