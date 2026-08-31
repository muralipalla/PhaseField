"""Validate and summarize Xeon16 functional and strong-scaling runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Iterable, Mapping


REQUIRED_RESPONSE_COLUMNS = {
    "pseudo_time",
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
    "minimum_phase",
    "damage_integral",
    "boundary_violation_inf",
    "phase_increment_inf",
    "phase_kkt_residual_inf",
    "free_mechanical_residual_inf",
    "staggered_iterations",
    "newton_iterations_last",
    "phase_optimizer_iterations",
}

REGRESSION_FIELDS = (
    "top_reaction_x",
    "top_reaction_y",
    "bottom_reaction_x",
    "bottom_reaction_y",
    "elastic_energy",
    "fracture_energy",
    "damage_integral",
    "minimum_phase",
    "crack_extension",
    "crack_path_length",
)


@dataclass
class RunAnalysis:
    entry: dict[str, str]
    summary: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    response: list[dict[str, float]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.errors


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value is not an object")
    return value


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def _read_settings(path: Path) -> dict[str, str]:
    rows = _read_tsv(path)
    return {row["key"]: row["value"] for row in rows}


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_write_json(path: Path, value: object) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2) + "\n")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _close(left: float, right: float, *, rtol: float, atol: float) -> bool:
    return abs(left - right) <= atol + rtol * max(abs(left), abs(right))


def _required_artifacts(config: Mapping[str, Any]) -> list[str]:
    artifacts = ["summary.json", "run_manifest.json", "load_response.csv"]
    if bool(config.get("write_xdmf")):
        artifacts.extend(
            [
                "displacement.xdmf",
                "displacement.h5",
                "phase_field.xdmf",
                "phase_field.h5",
            ]
        )
    if bool(config.get("write_material_fields")):
        artifacts.extend(["material_fields.xdmf", "material_fields.h5"])
    if bool(config.get("make_plots")):
        artifacts.extend(["load_displacement.png", "final_phase_field.png"])
    return artifacts


def _response_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        columns = set(reader.fieldnames or ())
        missing = sorted(REQUIRED_RESPONSE_COLUMNS - columns)
        if missing:
            raise ValueError(f"missing response columns: {', '.join(missing)}")
        rows: list[dict[str, float]] = []
        for row_number, row in enumerate(reader, start=2):
            converted: dict[str, float] = {}
            for key, text in row.items():
                try:
                    converted[key] = float(text)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"row {row_number}, column {key!r} is not numeric"
                    ) from error
            if not all(math.isfinite(value) for value in converted.values()):
                raise ValueError(f"row {row_number} contains a non-finite value")
            rows.append(converted)
    if not rows:
        raise ValueError("response CSV has no rows")
    return rows


def _check_material_manifest(analysis: RunAnalysis) -> None:
    assert analysis.manifest is not None
    manifest = analysis.manifest
    config = manifest.get("config", {})
    material = manifest.get("material", {})
    ranges = material.get("ranges", {})
    for property_name in (
        "young_modulus",
        "poisson_ratio",
        "fracture_toughness",
        "length_scale",
    ):
        limits = ranges.get(property_name)
        if not isinstance(limits, dict):
            analysis.errors.append(f"material range missing for {property_name}")
            continue
        minimum = limits.get("min")
        maximum = limits.get("max")
        if not (_finite_number(minimum) and _finite_number(maximum)):
            analysis.errors.append(f"material range is non-finite for {property_name}")
        elif float(minimum) > float(maximum):
            analysis.errors.append(f"material range is reversed for {property_name}")
    length_limits = ranges.get("length_scale", {})
    if _finite_number(length_limits.get("min")) and float(length_limits["min"]) <= 0:
        analysis.errors.append("minimum material length_scale is not positive")

    mode = str(config.get("material_mode", "uniform")).casefold()
    if mode in {"linear_x", "linear_y"}:
        pairs = (
            ("young_modulus", "young_modulus_end"),
            ("poisson_ratio", "poisson_ratio_end"),
            ("fracture_toughness", "fracture_toughness_end"),
            ("length_scale", "length_scale_end"),
        )
        for start_name, end_name in pairs:
            start = config.get(start_name)
            end = config.get(end_name)
            limits = ranges.get(start_name, {})
            if _finite_number(start) and _finite_number(end) and float(start) != float(end):
                span = float(limits.get("max", math.nan)) - float(
                    limits.get("min", math.nan)
                )
                if not math.isfinite(span) or span <= 0.0:
                    analysis.errors.append(
                        f"{mode} did not produce a spatial {start_name} range"
                    )
    elif mode == "file":
        specification = material.get("specification", {})
        recorded_sha = specification.get("source_sha256")
        if not recorded_sha:
            analysis.errors.append("file material provenance has no SHA-256")
        source_path = specification.get("source_path")
        if not source_path:
            analysis.errors.append("file material provenance has no resolved source path")
        else:
            try:
                actual_sha = sha256(Path(source_path).read_bytes()).hexdigest()
            except OSError as error:
                analysis.errors.append(f"cannot re-read material source: {error}")
            else:
                if recorded_sha != actual_sha:
                    analysis.errors.append(
                        "material source SHA-256 does not match the current source file"
                    )
        region_counts = material.get("region_cell_counts", {})
        if not region_counts or not any(int(value) > 0 for value in region_counts.values()):
            analysis.errors.append("file material regions selected no mesh cells")

    expected_ranges: dict[str, tuple[float, float]] = {}
    expected_region: tuple[str, int] | None = None
    case_name = analysis.entry.get("case")
    if mode == "uniform":
        for property_name in (
            "young_modulus",
            "poisson_ratio",
            "fracture_toughness",
            "length_scale",
        ):
            configured = config.get(property_name)
            if not _finite_number(configured):
                analysis.errors.append(
                    f"uniform config is missing {property_name}"
                )
            else:
                expected_ranges[property_name] = (
                    float(configured),
                    float(configured),
                )
    if case_name == "graded_linear":
        expected_ranges = {
            "young_modulus": (105.4375, 209.5625),
            "poisson_ratio": (0.30, 0.30),
            "fracture_toughness": (0.00180375, 0.00269625),
            "length_scale": (0.040, 0.040),
        }
    elif case_name == "graded_inclusion":
        expected_ranges = {
            "young_modulus": (168.175, 315.0),
            "poisson_ratio": (0.28, 0.30),
            "fracture_toughness": (0.0027, 0.0032),
            "length_scale": (0.040, 0.040),
        }
        expected_region = ("stiff_tough_inclusion", 576)
    for property_name, (expected_minimum, expected_maximum) in expected_ranges.items():
        limits = ranges.get(property_name, {})
        actual_minimum = float(limits.get("min", math.nan))
        actual_maximum = float(limits.get("max", math.nan))
        if not _close(
            actual_minimum, expected_minimum, rtol=1.0e-10, atol=1.0e-12
        ) or not _close(
            actual_maximum, expected_maximum, rtol=1.0e-10, atol=1.0e-12
        ):
            analysis.errors.append(
                f"unexpected {property_name} range: "
                f"[{actual_minimum:.12g}, {actual_maximum:.12g}]"
            )
    if expected_region is not None:
        region_name, expected_count = expected_region
        actual_count = int(material.get("region_cell_counts", {}).get(region_name, -1))
        if actual_count != expected_count:
            analysis.errors.append(
                f"region {region_name} selected {actual_count} cells; "
                f"expected {expected_count}"
            )


def analyze_run(
    entry: dict[str, str],
    *,
    balance_rtol: float,
    balance_atol: float,
) -> RunAnalysis:
    analysis = RunAnalysis(entry=entry)
    log_value = entry.get("log_path")
    if not log_value:
        analysis.errors.append("ledger log path is missing")
    else:
        log_path = Path(log_value)
        if not log_path.is_file() or log_path.stat().st_size == 0:
            analysis.errors.append("console log is missing or empty")
    output = Path(entry["output_path"])
    try:
        launcher_rc = int(entry["launcher_rc"])
    except (KeyError, ValueError):
        launcher_rc = -1
        analysis.errors.append("ledger launcher status is invalid")
    if launcher_rc != 0:
        analysis.errors.append(f"launcher exited with status {launcher_rc}")
    if not output.is_dir():
        analysis.errors.append("result directory is missing")
        return analysis

    for filename, attribute in (
        ("summary.json", "summary"),
        ("run_manifest.json", "manifest"),
    ):
        path = output / filename
        try:
            setattr(analysis, attribute, _read_json(path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            analysis.errors.append(f"cannot read {filename}: {error}")
    response_path = output / "load_response.csv"
    try:
        analysis.response = _response_rows(response_path)
    except (OSError, ValueError) as error:
        analysis.errors.append(f"cannot read load_response.csv: {error}")

    if analysis.summary is None or analysis.manifest is None:
        return analysis
    summary = analysis.summary
    manifest = analysis.manifest
    config = manifest.get("config", {})
    requested_ranks = int(entry["ranks"])

    if summary.get("status") != "completed":
        analysis.errors.append(
            f"summary status is {summary.get('status')!r}, not 'completed'"
        )
    for source_name, payload in (("summary", summary), ("manifest", manifest)):
        if payload.get("mpi_ranks") != requested_ranks:
            analysis.errors.append(
                f"{source_name} mpi_ranks={payload.get('mpi_ranks')!r}; "
                f"requested {requested_ranks}"
            )

    for filename in _required_artifacts(config):
        artifact = output / filename
        if not artifact.is_file() or artifact.stat().st_size == 0:
            analysis.errors.append(f"required artifact is missing or empty: {filename}")

    final = summary.get("final")
    if not isinstance(final, dict):
        analysis.errors.append("summary has no final accepted state")
    else:
        selected = set(REGRESSION_FIELDS) | {
            "pseudo_time",
            "opening_displacement",
            "sliding_displacement",
            "force_balance_x",
            "force_balance_y",
            "phase_kkt_residual_inf",
            "free_mechanical_residual_inf",
            "boundary_violation_inf",
        }
        for name in selected:
            if name in final and not _finite_number(final[name]):
                analysis.errors.append(f"final diagnostic {name} is non-finite")
        if not _close(
            float(final.get("pseudo_time", math.nan)), 1.0, rtol=0.0, atol=1.0e-12
        ):
            analysis.errors.append("final pseudo_time is not 1")
        expected_opening = float(config.get("max_displacement", math.nan))
        expected_sliding = float(config.get("max_sliding_displacement", math.nan))
        for name, expected in (
            ("opening_displacement", expected_opening),
            ("sliding_displacement", expected_sliding),
        ):
            actual = float(final.get(name, math.nan))
            if not _close(actual, expected, rtol=0.0, atol=1.0e-12):
                analysis.errors.append(f"final {name} does not reach the configured load")

    if analysis.response:
        response = analysis.response
        expected_rows = int(summary.get("steps_completed", -1)) + 1
        if len(response) != expected_rows:
            analysis.errors.append(
                f"CSV has {len(response)} rows; expected {expected_rows}"
            )
        times = [row["pseudo_time"] for row in response]
        if not _close(times[0], 0.0, rtol=0.0, atol=1.0e-12):
            analysis.errors.append("response does not begin at pseudo_time 0")
        if any(right <= left for left, right in zip(times, times[1:])):
            analysis.errors.append("accepted pseudo_time values are not strictly increasing")
        damages = [row["damage_integral"] for row in response]
        if any(right + 1.0e-10 < left for left, right in zip(damages, damages[1:])):
            analysis.errors.append("damage_integral decreases despite irreversibility")

        phase_tolerance = 1.05 * float(config.get("phase_kkt_tolerance", 1.0e-8))
        mechanical_tolerance = 1.05 * float(
            config.get("staggered_mechanical_tolerance", 1.0e-6)
        )
        boundary_tolerance = 1.05 * float(
            config.get("newton_increment_tolerance", 1.0e-10)
        )
        staggered_tolerance = 1.05 * float(
            config.get("staggered_tolerance", 1.0e-5)
        )
        for index, row in enumerate(response):
            if not -1.0e-10 <= row["minimum_phase"] <= 1.0 + 1.0e-10:
                analysis.errors.append(f"row {index} phase lies outside [0,1]")
            checks = (
                ("phase_kkt_residual_inf", phase_tolerance),
                ("free_mechanical_residual_inf", mechanical_tolerance),
                ("boundary_violation_inf", boundary_tolerance),
                ("phase_increment_inf", staggered_tolerance),
            )
            for name, tolerance in checks:
                if row[name] > tolerance:
                    analysis.errors.append(
                        f"row {index} {name}={row[name]:.3e} exceeds {tolerance:.3e}"
                    )
            for component in ("x", "y"):
                computed_balance = (
                    row[f"top_reaction_{component}"]
                    + row[f"bottom_reaction_{component}"]
                )
                recorded_balance = row[f"force_balance_{component}"]
                if not _close(
                    recorded_balance,
                    computed_balance,
                    rtol=1.0e-12,
                    atol=1.0e-14,
                ):
                    analysis.errors.append(
                        f"row {index} recorded force_balance_{component} does not "
                        "equal top_reaction + bottom_reaction"
                    )
                balance = abs(computed_balance)
                scale = max(
                    abs(row[f"top_reaction_{component}"]),
                    abs(row[f"bottom_reaction_{component}"]),
                )
                tolerance = max(balance_atol, balance_rtol * scale)
                if balance > tolerance:
                    analysis.errors.append(
                        f"row {index} force balance {component}={balance:.3e} "
                        f"exceeds {tolerance:.3e}"
                    )

        analysis.statistics = {
            "max_force_balance_x": max(
                abs(row["top_reaction_x"] + row["bottom_reaction_x"])
                for row in response
            ),
            "max_force_balance_y": max(
                abs(row["top_reaction_y"] + row["bottom_reaction_y"])
                for row in response
            ),
            "max_phase_kkt_residual": max(row["phase_kkt_residual_inf"] for row in response),
            "max_free_mechanical_residual": max(
                row["free_mechanical_residual_inf"] for row in response
            ),
            "sum_staggered_iterations": int(
                sum(row["staggered_iterations"] for row in response[1:])
            ),
            "sum_newton_iterations": int(
                sum(row["newton_iterations_last"] for row in response[1:])
            ),
            "sum_phase_iterations": int(
                sum(row["phase_optimizer_iterations"] for row in response[1:])
            ),
        }
    elapsed = summary.get("elapsed_seconds")
    if not _finite_number(elapsed) or float(elapsed) <= 0.0:
        analysis.errors.append("summary elapsed_seconds is not positive and finite")
    if int(summary.get("total_load_cutbacks", 0)) > 0:
        analysis.warnings.append(
            f"run used {summary['total_load_cutbacks']} adaptive load cutback(s)"
        )
    case_name = entry.get("case")
    if case_name in {"mixed_mode", "mixed_symmetric", "graded_inclusion"}:
        sliding_peak = summary.get("peak_absolute_sliding_reaction")
        if not _finite_number(sliding_peak) or float(sliding_peak) <= 1.0e-8:
            analysis.errors.append("mixed-mode case produced no measurable sliding reaction")
    if case_name == "mode_i" and analysis.response:
        initial_damage = analysis.response[0]["damage_integral"]
        final_damage = analysis.response[-1]["damage_integral"]
        if final_damage <= initial_damage + 1.0e-10:
            analysis.errors.append("Mode-I fracture case produced no additional damage")
        final_state = summary.get("final") or {}
        minimum_extension = float(config.get("width", 1.0)) / int(
            config.get("nx", 1)
        )
        if float(final_state.get("crack_extension", 0.0)) < (
            minimum_extension - 1.0e-12
        ):
            analysis.errors.append(
                "Mode-I case did not propagate the thresholded crack by at "
                f"least one mesh interval ({minimum_extension:.6g})"
            )
        if summary.get("peak_reaction_step") == summary.get("steps_completed"):
            analysis.warnings.append(
                "Mode-I peak reaction occurs at the final state; no post-peak branch was sampled"
            )
    _check_material_manifest(analysis)
    return analysis


def _compare_values(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    fields: Iterable[str],
    *,
    rtol: float,
    atol: float,
) -> list[str]:
    differences: list[str] = []
    for name in fields:
        if name not in left or name not in right:
            differences.append(f"missing comparison field {name}")
            continue
        first = float(left[name])
        second = float(right[name])
        if not _close(first, second, rtol=rtol, atol=atol):
            differences.append(f"{name}: {first:.12g} versus {second:.12g}")
    return differences


def compare_mixed_schemes(
    analyses: list[RunAnalysis], *, rtol: float, atol: float
) -> dict[str, Any] | None:
    by_name = {analysis.entry["case"]: analysis for analysis in analyses}
    relative = by_name.get("mixed_mode")
    symmetric = by_name.get("mixed_symmetric")
    if relative is None or symmetric is None:
        return None
    errors: list[str] = []
    if relative.passed and symmetric.passed:
        if len(relative.response) != len(symmetric.response):
            errors.append("mixed schemes accepted different numbers of load states")
        else:
            curve_fields = (
                "pseudo_time",
                "top_reaction_x",
                "top_reaction_y",
                "elastic_energy",
                "fracture_energy",
                "damage_integral",
                "minimum_phase",
            )
            for index, (left, right) in enumerate(
                zip(relative.response, symmetric.response)
            ):
                errors.extend(
                    f"row {index} {message}"
                    for message in _compare_values(
                        left, right, curve_fields, rtol=rtol, atol=atol
                    )
                )
    else:
        errors.append("one or both mixed-scheme runs failed individual validation")
    return {
        "status": "passed" if not errors else "failed",
        "relative_case": relative.entry["output_path"],
        "symmetric_case": symmetric.entry["output_path"],
        "rtol": rtol,
        "atol": atol,
        "differences": errors,
    }


def analyze_scaling(
    analyses: list[RunAnalysis],
    *,
    rtol: float,
    atol: float,
    expected_ranks: list[int] | None = None,
    expected_repetitions: int | None = None,
    expected_max_ranks: int | None = None,
    require_complete: bool = True,
) -> dict[str, Any] | None:
    scaling = [
        analysis for analysis in analyses if analysis.entry.get("category") == "scaling"
    ]
    if not scaling:
        return None
    by_rank: dict[int, list[RunAnalysis]] = {}
    for analysis in scaling:
        by_rank.setdefault(int(analysis.entry["ranks"]), []).append(analysis)
    errors: list[str] = []
    warnings: list[str] = []
    warmups = [
        analysis
        for analysis in analyses
        if analysis.entry.get("case") == "scaling_warmup"
    ]
    if require_complete and expected_ranks is not None:
        observed_ranks = sorted(by_rank)
        if observed_ranks != expected_ranks:
            errors.append(
                f"scaling ranks are {observed_ranks}; expected {expected_ranks}"
            )
        if expected_repetitions is not None:
            for rank in expected_ranks:
                actual_repetitions = len(by_rank.get(rank, []))
                if actual_repetitions != expected_repetitions:
                    errors.append(
                        f"rank {rank} has {actual_repetitions} repetition(s); "
                        f"expected {expected_repetitions}"
                    )
        if len(warmups) != 1:
            errors.append(f"found {len(warmups)} scaling warm-ups; expected 1")
        elif not warmups[0].passed:
            errors.append("the exact-input scaling warm-up failed validation")
        elif expected_max_ranks is not None and int(
            warmups[0].entry["ranks"]
        ) != expected_max_ranks:
            errors.append("scaling warm-up did not use max_ranks")
    hashes = {analysis.entry.get("input_sha256") for analysis in scaling}
    if len(hashes) != 1:
        errors.append("scaling runs did not use byte-identical input files")
    elif len(warmups) == 1 and warmups[0].entry.get("input_sha256") not in hashes:
        errors.append("scaling warm-up did not use the timed scaling input file")
    if 1 not in by_rank:
        errors.append("scaling sweep has no one-rank baseline")
        return {
            "status": "failed",
            "errors": errors,
            "warnings": warnings,
            "ranks": [],
        }

    valid_baselines = [analysis for analysis in by_rank[1] if analysis.passed]
    if not valid_baselines:
        errors.append("one-rank baseline failed validation")
        baseline = None
    else:
        baseline = valid_baselines[0]
    timing_rows: list[dict[str, Any]] = []
    baseline_elapsed = (
        statistics.median(
            float(analysis.summary["elapsed_seconds"])
            for analysis in valid_baselines
            if analysis.summary is not None
        )
        if valid_baselines
        else math.nan
    )
    baseline_wall = (
        statistics.median(float(analysis.entry["wall_seconds"]) for analysis in valid_baselines)
        if valid_baselines
        else math.nan
    )
    signatures: set[tuple[int, ...]] = set()
    for rank in sorted(by_rank):
        group = by_rank[rank]
        valid = [analysis for analysis in group if analysis.passed]
        if not valid:
            errors.append(f"all {rank}-rank scaling repetitions failed")
            continue
        elapsed = statistics.median(
            float(analysis.summary["elapsed_seconds"])
            for analysis in valid
            if analysis.summary is not None
        )
        wall = statistics.median(float(analysis.entry["wall_seconds"]) for analysis in valid)
        speedup = baseline_elapsed / elapsed if math.isfinite(baseline_elapsed) else math.nan
        efficiency = speedup / rank if math.isfinite(speedup) else math.nan
        timing_rows.append(
            {
                "ranks": rank,
                "repetitions": len(group),
                "valid_repetitions": len(valid),
                "median_solver_seconds": elapsed,
                "median_external_wall_seconds": wall,
                "speedup": speedup,
                "parallel_efficiency": efficiency,
            }
        )
        for analysis in valid:
            summary = analysis.summary or {}
            stats = analysis.statistics
            signatures.add(
                (
                    int(summary.get("steps_completed", -1)),
                    int(summary.get("total_load_cutbacks", -1)),
                    int(stats.get("sum_staggered_iterations", -1)),
                    int(stats.get("sum_newton_iterations", -1)),
                    int(stats.get("sum_phase_iterations", -1)),
                )
            )
            if baseline is not None and analysis is not baseline:
                left = baseline.summary.get("final", {}) if baseline.summary else {}
                right = summary.get("final", {})
                differences = _compare_values(
                    left, right, REGRESSION_FIELDS, rtol=rtol, atol=atol
                )
                errors.extend(
                    f"rank {rank}, repeat {analysis.entry['repeat']} {message}"
                    for message in differences
                )
    timing_comparable = len(signatures) == 1
    if not timing_comparable:
        warnings.append(
            "accepted-step/cutback/iteration workloads differ; speedup is not an equal-work comparison"
        )
    if timing_rows and timing_rows[-1]["parallel_efficiency"] < 0.25:
        warnings.append(
            "highest-rank parallel efficiency is below 25%; the fixed problem is likely too small or communication-bound"
        )
    return {
        "status": "passed" if not errors else "failed",
        "input_sha256": next(iter(hashes)) if len(hashes) == 1 else None,
        "regression_rtol": rtol,
        "regression_atol": atol,
        "timing_comparable": timing_comparable,
        "workload_signatures": [list(signature) for signature in sorted(signatures)],
        "ranks": timing_rows,
        "errors": errors,
        "warnings": warnings,
        "baseline_external_wall_seconds": baseline_wall,
    }


def _git_metadata(project_root: Path) -> dict[str, Any]:
    def command(*arguments: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *arguments],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = command("rev-parse", "HEAD")
    status = command("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def _run_record(analysis: RunAnalysis) -> dict[str, Any]:
    summary = analysis.summary or {}
    manifest = analysis.manifest or {}
    final = summary.get("final") or {}
    derived = manifest.get("derived") or {}
    return {
        "order": int(analysis.entry["order"]),
        "case": analysis.entry["case"],
        "category": analysis.entry["category"],
        "ranks": int(analysis.entry["ranks"]),
        "repeat": int(analysis.entry["repeat"]),
        "passed": analysis.passed,
        "launcher_rc": int(analysis.entry["launcher_rc"]),
        "status": summary.get("status"),
        "external_wall_seconds": float(analysis.entry["wall_seconds"]),
        "solver_elapsed_seconds": summary.get("elapsed_seconds"),
        "steps_completed": summary.get("steps_completed"),
        "cutbacks": summary.get("total_load_cutbacks"),
        "cells": derived.get("cells"),
        "displacement_dofs": derived.get("displacement_dofs"),
        "phase_dofs": derived.get("phase_dofs"),
        "peak_reaction_force": summary.get("peak_reaction_force"),
        "peak_absolute_sliding_reaction": summary.get(
            "peak_absolute_sliding_reaction"
        ),
        "final_top_reaction_x": final.get("top_reaction_x"),
        "final_top_reaction_y": final.get("top_reaction_y"),
        "final_elastic_energy": final.get("elastic_energy"),
        "final_fracture_energy": final.get("fracture_energy"),
        "final_minimum_phase": final.get("minimum_phase"),
        "final_crack_extension": final.get("crack_extension"),
        "final_crack_path_length": final.get("crack_path_length"),
        "final_crack_kink_angle_degrees": final.get("crack_kink_angle_degrees"),
        **analysis.statistics,
        "input_sha256": analysis.entry["input_sha256"],
        "input_path": analysis.entry["input_path"],
        "output_path": analysis.entry["output_path"],
        "log_path": analysis.entry["log_path"],
        "errors": analysis.errors,
        "warnings": analysis.warnings,
    }


def _write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
    columns = (
        "order",
        "case",
        "category",
        "ranks",
        "repeat",
        "passed",
        "launcher_rc",
        "status",
        "external_wall_seconds",
        "solver_elapsed_seconds",
        "steps_completed",
        "cutbacks",
        "cells",
        "displacement_dofs",
        "phase_dofs",
        "peak_reaction_force",
        "peak_absolute_sliding_reaction",
        "final_top_reaction_x",
        "final_top_reaction_y",
        "final_elastic_energy",
        "final_fracture_energy",
        "final_minimum_phase",
        "final_crack_extension",
        "final_crack_path_length",
        "final_crack_kink_angle_degrees",
        "max_force_balance_x",
        "max_force_balance_y",
        "max_phase_kkt_residual",
        "max_free_mechanical_residual",
        "sum_staggered_iterations",
        "sum_newton_iterations",
        "sum_phase_iterations",
        "errors",
        "warnings",
        "output_path",
        "log_path",
    )
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            flattened = dict(record)
            flattened["errors"] = " | ".join(record["errors"])
            flattened["warnings"] = " | ".join(record["warnings"])
            writer.writerow(flattened)
    temporary.replace(path)


def _relative_link(root: Path, value: str) -> str:
    try:
        return Path(value).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return value


def _markdown_report(
    root: Path,
    status: str,
    records: list[dict[str, Any]],
    scaling: dict[str, Any] | None,
    mixed: dict[str, Any] | None,
    suite_errors: list[str],
    suite_warnings: list[str],
) -> str:
    lines = [
        "# Xeon16 phase-field test report",
        "",
        f"Overall status: **{status.upper()}**",
        "",
    ]
    if suite_errors:
        lines.extend(["## Suite-level failures", ""])
        lines.extend(f"- {message}" for message in suite_errors)
        lines.append("")
    if suite_warnings:
        lines.extend(["## Suite-level warnings", ""])
        lines.extend(f"- {message}" for message in suite_warnings)
        lines.append("")
    lines.extend(
        [
            "## Functional and solver checks",
            "",
            "| Case | Ranks | Result | Solver s | Steps | Cutbacks | Max abs(Fx balance) | Max abs(Fy balance) | Artifacts |",
            "|---|---:|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for record in records:
        result = "PASS" if record["passed"] else "FAIL"
        solver_seconds = record.get("solver_elapsed_seconds")
        solver_text = f"{solver_seconds:.3f}" if _finite_number(solver_seconds) else "—"
        fx = record.get("max_force_balance_x")
        fy = record.get("max_force_balance_y")
        fx_text = f"{fx:.3e}" if _finite_number(fx) else "—"
        fy_text = f"{fy:.3e}" if _finite_number(fy) else "—"
        output_link = _relative_link(root, record["output_path"])
        log_link = _relative_link(root, record["log_path"])
        lines.append(
            f"| {record['case']} (rep {record['repeat']}) | {record['ranks']} | "
            f"{result} | {solver_text} | {record.get('steps_completed', '—')} | "
            f"{record.get('cutbacks', '—')} | {fx_text} | {fy_text} | "
            f"[result]({output_link}) · [log]({log_link}) |"
        )
    failures = [record for record in records if record["errors"]]
    warnings = [record for record in records if record["warnings"]]
    if failures:
        lines.extend(["", "### Failures", ""])
        for record in failures:
            for message in record["errors"]:
                lines.append(f"- `{record['case']}` ({record['ranks']} ranks): {message}")
    if warnings:
        lines.extend(["", "### Warnings", ""])
        for record in warnings:
            for message in record["warnings"]:
                lines.append(f"- `{record['case']}` ({record['ranks']} ranks): {message}")

    if mixed is not None:
        lines.extend(
            [
                "",
                "## Relative versus symmetric mixed loading",
                "",
                f"Comparison status: **{mixed['status'].upper()}**. The schemes are "
                "expected to agree because they prescribe the same relative edge motion.",
            ]
        )
        for difference in mixed["differences"]:
            lines.append(f"- {difference}")

    if scaling is not None:
        lines.extend(
            [
                "",
                "## Fixed-work strong scaling",
                "",
                f"Numerical regression status: **{scaling['status'].upper()}**. "
                f"Equal-work timing: **{'yes' if scaling['timing_comparable'] else 'no'}**.",
                "",
                "| Ranks | Repetitions | Median solver s | Speedup | Efficiency |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in scaling["ranks"]:
            lines.append(
                f"| {row['ranks']} | {row['valid_repetitions']}/{row['repetitions']} | "
                f"{row['median_solver_seconds']:.3f} | {row['speedup']:.3f} | "
                f"{100.0 * row['parallel_efficiency']:.1f}% |"
            )
        for message in scaling["errors"]:
            lines.append(f"- ERROR: {message}")
        for message in scaling["warnings"]:
            lines.append(f"- WARNING: {message}")
        lines.extend(
            [
                "",
                "Speedup is T1/Tp and efficiency is speedup/p. These are diagnostic "
                "workstation measurements, not publication-grade benchmarks unless CPU "
                "binding and repeated-run variability are also controlled.",
            ]
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `suite_manifest.json`: hardware, launcher, software preflight, and provenance.",
            "- `suite_summary.json`: complete machine-readable validation and scaling report.",
            "- `suite_summary.csv`: one row per simulation run.",
            "- `cases.tsv`: launcher ledger with input hashes and external wall times.",
            "- `logs/`: complete preflight and per-run console logs.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite_root", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--balance-rtol", type=float, default=1.0e-5)
    parser.add_argument("--balance-atol", type=float, default=1.0e-8)
    parser.add_argument("--mixed-rtol", type=float, default=1.0e-5)
    parser.add_argument("--mixed-atol", type=float, default=1.0e-8)
    parser.add_argument("--scaling-rtol", type=float, default=5.0e-5)
    parser.add_argument("--scaling-atol", type=float, default=1.0e-8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.suite_root.expanduser().resolve()
    ledger_path = root / "cases.tsv"
    settings_path = root / "suite_settings.tsv"
    try:
        entries = _read_tsv(ledger_path)
        settings = _read_settings(settings_path)
    except OSError as error:
        raise SystemExit(f"cannot read suite metadata: {error}") from error
    if not entries:
        raise SystemExit("suite ledger contains no simulation runs")

    analyses = [
        analyze_run(
            entry,
            balance_rtol=args.balance_rtol,
            balance_atol=args.balance_atol,
        )
        for entry in entries
    ]
    mixed = compare_mixed_schemes(
        analyses, rtol=args.mixed_rtol, atol=args.mixed_atol
    )
    suite_mode = settings.get("suite_mode", "")
    expects_scaling = suite_mode in {"scaling", "all"}
    expected_scaling_ranks = [
        int(token)
        for token in settings.get("rank_sweep", "1 2 4 8 16")
        .replace(",", " ")
        .split()
    ]
    expected_scaling_repetitions = int(settings.get("scaling_repeats", "1"))
    scaling = analyze_scaling(
        analyses,
        rtol=args.scaling_rtol,
        atol=args.scaling_atol,
        expected_ranks=expected_scaling_ranks if expects_scaling else None,
        expected_repetitions=(
            expected_scaling_repetitions if expects_scaling else None
        ),
        expected_max_ranks=(
            int(settings.get("max_ranks", "16")) if expects_scaling else None
        ),
        require_complete=expects_scaling and not args.allow_incomplete,
    )
    records = [_run_record(analysis) for analysis in analyses]

    hardware = None
    preflight = None
    for filename, target in (("hardware.json", "hardware"), ("preflight.json", "preflight")):
        try:
            value = _read_json(root / filename)
        except (OSError, ValueError, json.JSONDecodeError):
            value = None
        if target == "hardware":
            hardware = value
        else:
            preflight = value
    suite_errors: list[str] = []
    suite_warnings: list[str] = []
    expected_ranks = int(settings.get("max_ranks", "16"))
    expected_nodes = int(settings.get("expected_nodes", "1"))
    if not args.allow_incomplete:
        functional_cases = {
            "mode_i": 1,
            "mixed_mode": 1,
            "mixed_symmetric": 1,
            "graded_linear": 1,
            "graded_inclusion": 1,
        }
        expected_case_counts: dict[str, int] = {"smoke": 1}
        if suite_mode in {"validation", "all"}:
            expected_case_counts.update(functional_cases)
        if suite_mode in {"scaling", "all"}:
            expected_case_counts["scaling_warmup"] = 1
            expected_case_counts["scaling"] = (
                len(expected_scaling_ranks) * expected_scaling_repetitions
            )
        if suite_mode not in {"smoke", "validation", "scaling", "all"}:
            suite_errors.append(f"unknown suite_mode {suite_mode!r}")
        else:
            actual_case_counts = Counter(
                analysis.entry.get("case", "") for analysis in analyses
            )
            for case_name, expected_count in expected_case_counts.items():
                actual_count = actual_case_counts.get(case_name, 0)
                if actual_count != expected_count:
                    suite_errors.append(
                        f"case {case_name} has {actual_count} run(s); "
                        f"expected {expected_count} for {suite_mode} mode"
                    )
            unexpected_cases = sorted(
                set(actual_case_counts).difference(expected_case_counts)
            )
            if unexpected_cases:
                suite_errors.append(
                    "unexpected case(s) for "
                    f"{suite_mode} mode: {', '.join(unexpected_cases)}"
                )

    preflight_log = root / "logs" / "preflight.log"
    if not preflight_log.is_file() or preflight_log.stat().st_size == 0:
        suite_errors.append("preflight console log is missing or empty")
    if hardware is None:
        suite_errors.append("hardware.json is missing or invalid")
    else:
        if hardware.get("capacity_check") != "passed":
            suite_errors.append("hardware capacity check is not recorded as passed")
        if int(hardware.get("effective_rank_capacity", 0)) < expected_ranks:
            suite_errors.append("hardware report has insufficient effective rank capacity")
    if preflight is None:
        suite_errors.append("preflight.json is missing or invalid")
    else:
        if preflight.get("status") != "ok":
            suite_errors.append("PETSc/MPI preflight status is not ok")
        if preflight.get("mpi_ranks") != expected_ranks:
            suite_errors.append("preflight rank count does not match max_ranks")
        if preflight.get("mpi_nodes") != expected_nodes:
            suite_errors.append("preflight node count does not match expected_nodes")
        if not preflight.get("xdmf_checked"):
            suite_errors.append("preflight did not perform the XDMF/HDF5 round trip")
        reloaded_cells = preflight.get("xdmf_reloaded_cells")
        if not isinstance(reloaded_cells, int) or reloaded_cells <= 0:
            suite_errors.append("preflight has no valid XDMF read-back cell count")
        affinity = preflight.get("rank_cpu_affinity")
        if not isinstance(affinity, dict) or len(affinity) != expected_ranks:
            suite_errors.append("preflight affinity did not report every MPI rank")
        else:
            affinity_sets: list[tuple[int, ...]] = []
            for rank, value in affinity.items():
                if (
                    not isinstance(value, list)
                    or not value
                    or any(not isinstance(cpu, int) for cpu in value)
                ):
                    suite_errors.append(
                        f"preflight affinity mask for rank {rank} is invalid"
                    )
                    continue
                affinity_sets.append(tuple(sorted(set(value))))
            if len(affinity_sets) == expected_ranks:
                unique_masks = set(affinity_sets)
                if len(unique_masks) == 1:
                    shared_width = len(affinity_sets[0])
                    if shared_width < expected_ranks:
                        suite_errors.append(
                            "all MPI ranks are confined to the same affinity mask "
                            f"containing only {shared_width} CPU(s)"
                        )
                    else:
                        suite_warnings.append(
                            "MPI ranks share one broad CPU affinity mask; scaling "
                            "timings may be noisier than explicitly core-bound runs"
                        )
                singleton_cpus = [mask[0] for mask in affinity_sets if len(mask) == 1]
                duplicate_singletons = sorted(
                    cpu
                    for cpu, count in Counter(singleton_cpus).items()
                    if count > 1
                )
                if duplicate_singletons:
                    suite_errors.append(
                        "multiple MPI ranks are pinned to the same CPU(s): "
                        + ", ".join(str(cpu) for cpu in duplicate_singletons)
                    )
                if all(len(mask) <= 2 for mask in affinity_sets):
                    affinity_union = set().union(
                        *(set(mask) for mask in affinity_sets)
                    )
                    if len(affinity_union) < expected_ranks:
                        suite_errors.append(
                            "narrow MPI affinity masks cover only "
                            f"{len(affinity_union)} CPU(s) for {expected_ranks} ranks"
                        )
    project_root = Path(settings.get("project_root", root))
    manifest = {
        "suite": "Xeon16 phase-field PETSc/MPI validation",
        "suite_version": "1.0.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "settings": settings,
        "hardware": hardware,
        "preflight": preflight,
        "git": _git_metadata(project_root),
        "ledger_sha256": sha256(ledger_path.read_bytes()).hexdigest(),
    }
    _atomic_write_json(root / "suite_manifest.json", manifest)

    failed = bool(suite_errors) or any(not analysis.passed for analysis in analyses)
    if mixed is not None and mixed["status"] != "passed":
        failed = True
    if scaling is not None and scaling["status"] != "passed":
        failed = True
    status = "failed" if failed else ("incomplete" if args.allow_incomplete else "passed")
    summary = {
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_count": len(records),
        "passed_runs": sum(record["passed"] for record in records),
        "failed_runs": sum(not record["passed"] for record in records),
        "suite_errors": suite_errors,
        "suite_warnings": suite_warnings,
        "runs": records,
        "mixed_scheme_comparison": mixed,
        "strong_scaling": scaling,
    }
    _atomic_write_json(root / "suite_summary.json", summary)
    _write_summary_csv(root / "suite_summary.csv", records)
    _atomic_write_text(
        root / "suite_report.md",
        _markdown_report(
            root,
            status,
            records,
            scaling,
            mixed,
            suite_errors,
            suite_warnings,
        ),
    )
    print(f"Xeon16 suite report: {root / 'suite_report.md'}")
    print(f"Status: {status}; passed runs: {summary['passed_runs']}/{len(records)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
