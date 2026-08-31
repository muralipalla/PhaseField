from __future__ import annotations

import csv
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest

from linux_cluster.petsc_config import PetscSimulationConfig
from linux_cluster.summarize_xeon16_suite import (
    REGRESSION_FIELDS,
    RunAnalysis,
    analyze_run,
    analyze_scaling,
    main as summarize_main,
)
from linux_cluster.xeon16_hardware import (
    detect_hardware,
    parse_rank_sweep,
    scheduler_capacities,
)
from phasefield_input import parse_input_file
from phasefield_material import parse_material_file


ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "linux_cluster"
INPUTS = LINUX / "inputs" / "xeon16"


def test_all_xeon16_inputs_parse_validate_and_resolve_materials() -> None:
    names = {
        "xeon16_smoke.in",
        "xeon16_mode_i.in",
        "xeon16_mixed_mode.in",
        "xeon16_mixed_symmetric.in",
        "xeon16_graded_linear.in",
        "xeon16_graded_inclusion.in",
        "xeon16_scaling.in",
    }
    assert {path.name for path in INPUTS.glob("*.in")} == names
    for name in sorted(names):
        config = parse_input_file(INPUTS / name, PetscSimulationConfig())
        config.validate()
        minimum_configured_length = (
            min(config.length_scale, config.length_scale_end)
            if config.material_mode in {"linear_x", "linear_y"}
            else config.length_scale
        )
        assert config.cell_diameter <= 0.5 * minimum_configured_length

    relative = parse_input_file(
        INPUTS / "xeon16_mixed_mode.in", PetscSimulationConfig()
    )
    symmetric = parse_input_file(
        INPUTS / "xeon16_mixed_symmetric.in", PetscSimulationConfig()
    )
    assert relative.mechanical_bc_scheme == "relative_clamped"
    assert symmetric.mechanical_bc_scheme == "symmetric_clamped"
    assert relative.max_displacement == symmetric.max_displacement
    assert relative.max_sliding_displacement == symmetric.max_sliding_displacement
    assert relative.nx == symmetric.nx and relative.ny == symmetric.ny
    relative_payload = asdict(relative)
    symmetric_payload = asdict(symmetric)
    for payload in (relative_payload, symmetric_payload):
        payload.pop("mechanical_bc_scheme")
        payload.pop("output_directory")
    assert relative_payload == symmetric_payload

    inclusion = parse_input_file(
        INPUTS / "xeon16_graded_inclusion.in", PetscSimulationConfig()
    )
    material_path = INPUTS / inclusion.material_file
    spec = parse_material_file(
        material_path,
        {
            "young_modulus": inclusion.young_modulus,
            "poisson_ratio": inclusion.poisson_ratio,
            "fracture_toughness": inclusion.fracture_toughness,
            "length_scale": inclusion.length_scale,
        },
    )
    assert Path(spec.source_path) == material_path.resolve()
    assert spec.source_sha256 is not None and len(spec.source_sha256) == 64
    assert [region.name for region in spec.regions] == ["stiff_tough_inclusion"]
    centroids = np.asarray(
        [
            point
            for j in range(80)
            for i in range(80)
            for point in (
                ((i + 1.0 / 3.0) / 80.0, (j + 1.0 / 3.0) / 80.0),
                ((i + 2.0 / 3.0) / 80.0, (j + 2.0 / 3.0) / 80.0),
            )
        ]
    ).T
    values = spec.evaluate(centroids, width=1.0, height=1.0)
    masks = spec.region_masks(centroids)
    assert int(np.count_nonzero(masks["stiff_tough_inclusion"])) == 576
    assert np.min(values["young_modulus"]) == pytest.approx(168.175)
    assert np.max(values["young_modulus"]) == pytest.approx(315.0)


def test_scaling_case_has_fixed_medium_workload_and_no_heavy_output() -> None:
    config = parse_input_file(
        INPUTS / "xeon16_scaling.in", PetscSimulationConfig()
    )
    assert config.nx == config.ny == 320
    assert 2 * config.nx * config.ny == 204_800
    assert config.load_steps == 10
    assert not config.write_xdmf
    assert not config.write_material_fields
    assert not config.make_plots
    assert not config.verbose


def test_rank_sweep_validation() -> None:
    assert parse_rank_sweep("1 2,4 8,16") == [1, 2, 4, 8, 16]
    with pytest.raises(ValueError, match="strictly increasing"):
        parse_rank_sweep("1 4 4 16")
    with pytest.raises(ValueError, match="positive"):
        parse_rank_sweep("0 1 16")


def test_physical_core_detection_deduplicates_hyperthreads(tmp_path: Path) -> None:
    sysfs = tmp_path / "cpu"
    for cpu, package, core in ((0, 0, 0), (1, 0, 0), (2, 0, 1)):
        topology = sysfs / f"cpu{cpu}" / "topology"
        topology.mkdir(parents=True)
        (topology / "physical_package_id").write_text(
            str(package), encoding="utf-8"
        )
        (topology / "core_id").write_text(str(core), encoding="utf-8")
    report = detect_hardware(
        environment={"SLURM_NTASKS": "4"},
        affinity_cpu_ids=[0, 1, 2],
        sysfs_root=sysfs,
    )
    assert report["affinity_logical_cpus"] == 3
    assert report["physical_core_capacity"] == 2
    assert report["effective_rank_capacity"] == 2
    assert report["physical_topology_complete"] is True


def test_scheduler_capacity_reads_pbs_nodefile(tmp_path: Path) -> None:
    nodefile = tmp_path / "nodes"
    nodefile.write_text("node1\nnode1\nnode2\n", encoding="utf-8")
    capacities = scheduler_capacities(
        {"PBS_NP": "3", "PBS_NODEFILE": str(nodefile)}
    )
    assert capacities == {"PBS_NP": 3, "PBS_NODEFILE": 3}


def _response_row(
    pseudo_time: float,
    balance: float = 0.0,
    recorded_balance: float | None = None,
) -> dict[str, float]:
    loaded = pseudo_time > 0.0
    return {
        "pseudo_time": pseudo_time,
        "opening_displacement": 1.0e-3 * pseudo_time,
        "sliding_displacement": 0.0,
        "top_reaction_x": 0.0,
        "top_reaction_y": 1.0 if loaded else 0.0,
        "bottom_reaction_x": 0.0,
        "bottom_reaction_y": -1.0 + balance if loaded else 0.0,
        "force_balance_x": 0.0,
        "force_balance_y": (
            balance if recorded_balance is None else recorded_balance
        )
        if loaded
        else 0.0,
        "elastic_energy": 0.1 * pseudo_time,
        "fracture_energy": 0.01 * pseudo_time,
        "minimum_phase": 1.0 - 0.1 * pseudo_time,
        "maximum_history": 0.02 * pseudo_time,
        "damage_integral": 0.1 * pseudo_time,
        "crack_extension": 0.0,
        "crack_path_length": 0.2,
        "crack_kink_angle_degrees": 0.0,
        "boundary_violation_inf": 1.0e-12 if loaded else 0.0,
        "phase_increment_inf": 1.0e-6 if loaded else 0.0,
        "phase_kkt_residual_inf": 1.0e-10 if loaded else 0.0,
        "free_mechanical_residual_inf": 1.0e-8 if loaded else 0.0,
        "staggered_iterations": 2.0 if loaded else 0.0,
        "newton_iterations_last": 3.0 if loaded else 0.0,
        "phase_optimizer_iterations": 1.0 if loaded else 0.0,
    }


def _synthetic_suite(
    tmp_path: Path,
    balance: float = 0.0,
    recorded_balance: float | None = None,
    suite_mode: str = "smoke",
) -> tuple[Path, dict[str, str]]:
    suite = tmp_path / "suite"
    output = suite / "runs" / "01_smoke_16r_rep1"
    log = suite / "logs" / "01_smoke_16r_rep1.log"
    output.mkdir(parents=True)
    log.parent.mkdir(parents=True)
    log.write_text("synthetic log\n", encoding="utf-8")
    (log.parent / "preflight.log").write_text(
        "synthetic preflight log\n", encoding="utf-8"
    )
    rows = [
        _response_row(0.0),
        _response_row(1.0, balance, recorded_balance),
    ]
    config = {
        "material_mode": "uniform",
        "young_modulus": 210.0,
        "poisson_ratio": 0.30,
        "fracture_toughness": 0.0027,
        "length_scale": 0.10,
        "max_displacement": 1.0e-3,
        "max_sliding_displacement": 0.0,
        "write_xdmf": False,
        "write_material_fields": False,
        "make_plots": False,
        "phase_kkt_tolerance": 1.0e-8,
        "staggered_mechanical_tolerance": 1.0e-6,
        "newton_increment_tolerance": 1.0e-10,
        "staggered_tolerance": 1.0e-5,
    }
    final = dict(rows[-1])
    summary = {
        "status": "completed",
        "mpi_ranks": 16,
        "elapsed_seconds": 2.5,
        "steps_completed": 1,
        "total_load_cutbacks": 0,
        "peak_reaction_force": 1.0,
        "peak_absolute_sliding_reaction": 0.0,
        "final": final,
    }
    material_ranges = {
        "young_modulus": {"min": 210.0, "max": 210.0},
        "poisson_ratio": {"min": 0.3, "max": 0.3},
        "fracture_toughness": {"min": 0.0027, "max": 0.0027},
        "length_scale": {"min": 0.1, "max": 0.1},
    }
    manifest = {
        "mpi_ranks": 16,
        "config": config,
        "material": {
            "ranges": material_ranges,
            "specification": {"source_sha256": None},
            "region_cell_counts": {},
        },
        "derived": {
            "cells": 2048,
            "displacement_dofs": 2178,
            "phase_dofs": 1089,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (output / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with (output / "load_response.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    entry = {
        "order": "1",
        "case": "smoke",
        "category": "smoke",
        "ranks": "16",
        "repeat": "1",
        "input_sha256": "a" * 64,
        "input_path": str(INPUTS / "xeon16_smoke.in"),
        "output_path": str(output),
        "log_path": str(log),
        "start_utc": "2026-01-01T00:00:00Z",
        "end_utc": "2026-01-01T00:00:03Z",
        "wall_seconds": "3",
        "launcher_rc": "0",
    }
    with (suite / "cases.tsv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entry), delimiter="\t")
        writer.writeheader()
        writer.writerow(entry)
    with (suite / "suite_settings.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["key", "value"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"key": "project_root", "value": str(ROOT)})
        writer.writerow({"key": "suite_mode", "value": suite_mode})
        writer.writerow({"key": "max_ranks", "value": "16"})
        writer.writerow({"key": "expected_nodes", "value": "1"})
    (suite / "hardware.json").write_text(
        json.dumps(
            {"capacity_check": "passed", "effective_rank_capacity": 16}
        ),
        encoding="utf-8",
    )
    (suite / "preflight.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "mpi_ranks": 16,
                "mpi_nodes": 1,
                "xdmf_checked": True,
                "xdmf_reloaded_cells": 256,
                "rank_cpu_affinity": {
                    str(rank): [rank] for rank in range(16)
                },
            }
        ),
        encoding="utf-8",
    )
    return suite, entry


def test_summarizer_accepts_a_complete_synthetic_run(tmp_path: Path) -> None:
    suite, _entry = _synthetic_suite(tmp_path)
    assert summarize_main([str(suite)]) == 0
    summary = json.loads((suite / "suite_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "passed"
    assert summary["passed_runs"] == 1
    assert (suite / "suite_manifest.json").is_file()
    assert (suite / "suite_summary.csv").is_file()
    assert "Overall status: **PASSED**" in (suite / "suite_report.md").read_text(
        encoding="utf-8"
    )


def test_summarizer_rejects_force_imbalance(tmp_path: Path) -> None:
    suite, entry = _synthetic_suite(tmp_path, balance=1.0e-3)
    analysis = analyze_run(entry, balance_rtol=1.0e-5, balance_atol=1.0e-8)
    assert not analysis.passed
    assert any("force balance y" in error for error in analysis.errors)


def test_summarizer_recomputes_balance_instead_of_trusting_csv(tmp_path: Path) -> None:
    suite, entry = _synthetic_suite(
        tmp_path, balance=1.0e-3, recorded_balance=0.0
    )
    analysis = analyze_run(entry, balance_rtol=1.0e-5, balance_atol=1.0e-8)
    assert not analysis.passed
    assert any("does not equal top_reaction" in error for error in analysis.errors)
    assert analysis.statistics["max_force_balance_y"] == pytest.approx(1.0e-3)


def test_summarizer_requires_nonempty_per_run_log(tmp_path: Path) -> None:
    _suite, entry = _synthetic_suite(tmp_path)
    Path(entry["log_path"]).unlink()
    analysis = analyze_run(entry, balance_rtol=1.0e-5, balance_atol=1.0e-8)
    assert not analysis.passed
    assert "console log is missing or empty" in analysis.errors


def test_scaling_mode_cannot_pass_without_scaling_cases(tmp_path: Path) -> None:
    suite, _entry = _synthetic_suite(tmp_path, suite_mode="scaling")
    assert summarize_main([str(suite)]) == 1
    summary = json.loads((suite / "suite_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert any("case scaling" in error for error in summary["suite_errors"])


def test_summarizer_rejects_duplicate_single_cpu_affinity(tmp_path: Path) -> None:
    suite, _entry = _synthetic_suite(tmp_path)
    preflight_path = suite / "preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    preflight["rank_cpu_affinity"] = {str(rank): [0] for rank in range(16)}
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    assert summarize_main([str(suite)]) == 1
    summary = json.loads((suite / "suite_summary.json").read_text(encoding="utf-8"))
    assert any("same affinity mask" in error for error in summary["suite_errors"])


def test_scaling_analysis_requires_every_rank_and_warmup() -> None:
    final = {name: 1.0 for name in REGRESSION_FIELDS}
    baseline = RunAnalysis(
        entry={
            "case": "scaling",
            "category": "scaling",
            "ranks": "1",
            "repeat": "1",
            "input_sha256": "b" * 64,
            "wall_seconds": "11",
        },
        summary={
            "elapsed_seconds": 10.0,
            "steps_completed": 10,
            "total_load_cutbacks": 0,
            "final": final,
        },
        statistics={
            "sum_staggered_iterations": 20,
            "sum_newton_iterations": 30,
            "sum_phase_iterations": 10,
        },
    )
    result = analyze_scaling(
        [baseline],
        rtol=5.0e-5,
        atol=1.0e-8,
        expected_ranks=[1, 2],
        expected_repetitions=1,
        expected_max_ranks=2,
        require_complete=True,
    )
    assert result is not None
    assert result["status"] == "failed"
    assert any("expected [1, 2]" in error for error in result["errors"])
    assert any("warm-ups" in error for error in result["errors"])


def test_scaling_warmup_must_use_timed_input_hash() -> None:
    final = {name: 1.0 for name in REGRESSION_FIELDS}
    baseline = RunAnalysis(
        entry={
            "case": "scaling",
            "category": "scaling",
            "ranks": "1",
            "repeat": "1",
            "input_sha256": "b" * 64,
            "wall_seconds": "11",
        },
        summary={
            "elapsed_seconds": 10.0,
            "steps_completed": 10,
            "total_load_cutbacks": 0,
            "final": final,
        },
        statistics={
            "sum_staggered_iterations": 20,
            "sum_newton_iterations": 30,
            "sum_phase_iterations": 10,
        },
    )
    warmup = RunAnalysis(
        entry={
            "case": "scaling_warmup",
            "category": "warmup",
            "ranks": "1",
            "repeat": "0",
            "input_sha256": "c" * 64,
            "wall_seconds": "12",
        }
    )
    result = analyze_scaling(
        [warmup, baseline],
        rtol=5.0e-5,
        atol=1.0e-8,
        expected_ranks=[1],
        expected_repetitions=1,
        expected_max_ranks=1,
        require_complete=True,
    )
    assert result is not None
    assert result["status"] == "failed"
    assert any("timed scaling input" in error for error in result["errors"])


def test_suite_scripts_and_documentation_are_present() -> None:
    launcher = (LINUX / "run_xeon16_suite.sh").read_text(encoding="utf-8")
    checker = (LINUX / "summarize_xeon16_suite.py").read_text(encoding="utf-8")
    hardware = (LINUX / "xeon16_hardware.py").read_text(encoding="utf-8")
    plan = (LINUX / "XEON16_TEST_PLAN.md").read_text(encoding="utf-8")
    for token in (
        "1 2 4 8 16",
        "PREFLIGHT_ONLY=1",
        "SKIP_PREFLIGHT=1",
        "SCALING_REPEATS",
        "PIPESTATUS[0]",
        "scaling_warmup",
        "SKIP_PREFLIGHT=0",
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "ALLOW_PETSC_OPTIONS",
    ):
        assert token in launcher
    assert "speedup" in checker
    assert "sched_getaffinity" in hardware
    assert "204,800" in plan
