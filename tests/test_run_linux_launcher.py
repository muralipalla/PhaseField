from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "linux_cluster" / "run_linux.sh"

pytestmark = pytest.mark.skipif(
    os.name == "nt", reason="launcher behavior tests require a native Linux Bash"
)


def _write_executable(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _fake_mpi_launcher(path: Path) -> None:
    _write_executable(
        path,
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \" $* \" == *mpi_compatibility_probe.py* ]]; then\n"
        "  if [[ -n \"${PROBE_CAPTURE_FILE:-}\" ]]; then printf '%s\\n' \"$@\" > \"${PROBE_CAPTURE_FILE}\"; fi\n"
        "  echo 'MPI_COMPATIBILITY_OK {\"compatible\": true}'\n"
        "else\n"
        "  printf '%s\\n' \"$@\" >> \"${CAPTURE_FILE:?}\"\n"
        "fi\n",
    )


def _fake_incompatible_mpi_launcher(path: Path) -> None:
    _write_executable(
        path,
        "#!/usr/bin/env bash\n"
        "if [[ \" $* \" == *mpi_compatibility_probe.py* ]]; then exit 1; fi\n"
        "printf '%s\\n' \"$@\" >> \"${CAPTURE_FILE:?}\"\n",
    )


def _fake_probe_only_mpi_launcher(path: Path) -> None:
    _write_executable(
        path,
        "#!/usr/bin/env bash\n"
        "if [[ \" $* \" == *mpi_compatibility_probe.py* ]]; then\n"
        "  echo 'MPI_COMPATIBILITY_OK {\"compatible\": true}'\n"
        "  exit 0\n"
        "fi\n"
        "exit 19\n",
    )


def _fake_python(
    path: Path, *, physical_cores: int = 64, scheduler_slots: int = 0
) -> None:
    _write_executable(
        path,
        "#!/usr/bin/env bash\n"
        f"printf '{physical_cores}\\t{scheduler_slots}\\tcomplete\\n'\n",
    )


def _base_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "ALLOW_EXTERNAL_MPI",
        "FORCE_INCOMPATIBLE_MPI",
        "MPI_HOME",
        "MPI_LAUNCHER",
        "NPROCS",
        "SLURM_JOB_ID",
        "SLURM_NTASKS",
        "SLURM_NNODES",
        "SLURM_PROCID",
        "PBS_NP",
        "PBS_NODEFILE",
        "PMI_RANK",
        "PMIX_RANK",
        "OMPI_COMM_WORLD_RANK",
        "MV2_COMM_WORLD_RANK",
    ):
        environment.pop(name, None)
    return environment


def _active_environment(tmp_path: Path) -> tuple[dict[str, str], Path]:
    capture = tmp_path / "launcher-argv.txt"
    probe_capture = tmp_path / "probe-argv.txt"
    mpi = tmp_path / "external" / "mpiexec"
    python = tmp_path / "runtime" / "python"
    _fake_mpi_launcher(mpi)
    _fake_python(python)
    environment = _base_environment()
    environment.update(
        {
            "RUNTIME_ENV": "active",
            "PYTHON_BIN": str(python),
            "MPI_PREFERENCE": str(mpi),
            "PREFLIGHT_ONLY": "1",
            "CAPTURE_FILE": str(capture),
            "PROBE_CAPTURE_FILE": str(probe_capture),
            "OUTPUT_DIR": str(tmp_path / "output"),
        }
    )
    return environment, capture


def _run(*arguments: str, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *arguments],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize("ranks", (2, 4, 16, 7))
def test_rank_option_reaches_launcher_and_preflight(
    tmp_path: Path, ranks: int
) -> None:
    environment, capture = _active_environment(tmp_path)
    result = _run("--ranks", str(ranks), environment=environment)

    assert result.returncode == 0, result.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[:2] == ["-n", str(ranks)]
    assert arguments[arguments.index("--expected-ranks") + 1] == str(ranks)
    probe_arguments = (tmp_path / "probe-argv.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    expected_probe_ranks = 2 if ranks >= 2 else 1
    assert probe_arguments[:2] == ["-n", str(expected_probe_ranks)]
    assert probe_arguments[probe_arguments.index("--expected-ranks") + 1] == str(
        expected_probe_ranks
    )
    assert "Preflight-only request completed" in result.stdout


def test_cli_rank_overrides_environment_rank(tmp_path: Path) -> None:
    environment, capture = _active_environment(tmp_path)
    environment["NPROCS"] = "16"

    result = _run("--ranks=4", environment=environment)

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").splitlines()[:2] == ["-n", "4"]


@pytest.mark.parametrize("ranks", ("", "0", "-1", "1.5", "four", "2 4"))
def test_invalid_rank_is_rejected_before_runtime_setup(
    tmp_path: Path, ranks: str
) -> None:
    environment, capture = _active_environment(tmp_path)

    result = _run(f"--ranks={ranks}", environment=environment)

    assert result.returncode == 2
    assert "positive integer" in result.stderr
    assert not capture.exists()


def test_one_rank_explicitly_enables_preflight_developer_mode(tmp_path: Path) -> None:
    environment, capture = _active_environment(tmp_path)

    result = _run("--ranks", "1", environment=environment)

    assert result.returncode == 0, result.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert "--allow-one-rank" in arguments


def test_scheduler_rank_is_used_when_direct_rank_is_omitted(tmp_path: Path) -> None:
    environment, capture = _active_environment(tmp_path)
    environment["SLURM_NTASKS"] = "4"

    result = _run(environment=environment)

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").splitlines()[:2] == ["-n", "4"]


def test_auto_uses_detected_physical_capacity(tmp_path: Path) -> None:
    environment, capture = _active_environment(tmp_path)

    result = _run("--ranks", "auto", environment=environment)

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").splitlines()[:2] == ["-n", "64"]
    assert "MPI ranks   : 64" in result.stdout


def test_oversized_request_is_reduced_unless_strict(tmp_path: Path) -> None:
    environment, capture = _active_environment(tmp_path)
    python = Path(environment["PYTHON_BIN"])
    _fake_python(python, physical_cores=8, scheduler_slots=0)

    reduced = _run("--ranks", "16", environment=environment)
    assert reduced.returncode == 0, reduced.stderr
    assert "using detected capacity 8" in reduced.stderr
    assert capture.read_text(encoding="utf-8").splitlines()[:2] == ["-n", "8"]

    capture.unlink()
    environment["STRICT_RESOURCES"] = "1"
    strict = _run("--ranks", "16", environment=environment)
    assert strict.returncode == 2
    assert "only 8" in strict.stderr
    assert not capture.exists()


def test_multinode_auto_respects_physical_and_scheduler_capacity(
    tmp_path: Path,
) -> None:
    environment, capture = _active_environment(tmp_path)
    python = Path(environment["PYTHON_BIN"])
    _fake_python(python, physical_cores=16, scheduler_slots=32)
    environment["SLURM_NTASKS"] = "32"
    environment["SLURM_NNODES"] = "2"

    result = _run("--ranks", "auto", environment=environment)

    assert result.returncode == 0, result.stderr
    assert capture.read_text(encoding="utf-8").splitlines()[:2] == ["-n", "32"]
    assert "physical cores and scheduler slots across 2 allocated nodes" in result.stdout

    capture.unlink()
    _fake_python(python, physical_cores=16, scheduler_slots=64)
    environment["SLURM_NTASKS"] = "64"
    capped = _run("--ranks", "auto", environment=environment)
    assert capped.returncode == 0, capped.stderr
    assert capture.read_text(encoding="utf-8").splitlines()[:2] == ["-n", "32"]

    capture.unlink()
    environment["STRICT_RESOURCES"] = "1"
    strict = _run("--ranks", "64", environment=environment)
    assert strict.returncode == 2
    assert "only 32" in strict.stderr
    assert not capture.exists()


def test_slurm_auto_prefers_scheduler_native_srun(tmp_path: Path) -> None:
    environment, capture = _active_environment(tmp_path)
    srun = tmp_path / "scheduler" / "srun"
    _fake_mpi_launcher(srun)
    current_mpi_dir = str((tmp_path / "external").resolve())
    environment.update(
        {
            "MPI_PREFERENCE": "auto",
            "SLURM_JOB_ID": "12345",
            "SLURM_NTASKS": "4",
            "SLURM_NNODES": "1",
            "PATH": os.pathsep.join(
                [str(srun.parent), current_mpi_dir, environment.get("PATH", "")]
            ),
        }
    )

    result = _run(environment=environment)

    assert result.returncode == 0, result.stderr
    arguments = capture.read_text(encoding="utf-8").splitlines()
    assert arguments[:2] == ["--ntasks=4", "--cpus-per-task=1"]
    assert f"Launcher    : {srun} --ntasks=4" in result.stdout


def _fake_conda_base(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "conda-base"
    prefix = tmp_path / "conda-environment"
    prefix.mkdir()
    hook = base / "etc" / "profile.d" / "conda.sh"
    _write_executable(
        hook,
        "conda() {\n"
        "  if [[ \"${1:-}\" != \"activate\" ]]; then return 2; fi\n"
        "  export CONDA_PREFIX=\"${FAKE_CONDA_PREFIX:?}\"\n"
        "}\n",
    )
    return base, prefix


def test_compatible_external_launcher_is_selected_automatically(
    tmp_path: Path,
) -> None:
    capture = tmp_path / "launcher-argv.txt"
    mpi = tmp_path / "system" / "mpiexec"
    python = tmp_path / "runtime" / "python"
    _fake_mpi_launcher(mpi)
    _fake_python(python)
    base, prefix = _fake_conda_base(tmp_path)
    environment = _base_environment()
    environment.update(
        {
            "RUNTIME_ENV": "conda",
            "CONDA_BASE": str(base),
            "FAKE_CONDA_PREFIX": str(prefix),
            "PYTHON_BIN": str(python),
            "MPI_PREFERENCE": str(mpi),
            "PREFLIGHT_ONLY": "1",
            "CAPTURE_FILE": str(capture),
            "OUTPUT_DIR": str(tmp_path / "output"),
        }
    )
    environment.pop("ALLOW_EXTERNAL_MPI", None)

    allowed = _run("--ranks", "4", environment=environment)
    assert allowed.returncode == 0, allowed.stderr
    assert "compatibility probe passed" in allowed.stdout
    assert capture.read_text(encoding="utf-8").splitlines()[:2] == ["-n", "4"]


def test_incompatible_system_mpi_falls_back_to_conda(tmp_path: Path) -> None:
    capture = tmp_path / "launcher-argv.txt"
    system_root = tmp_path / "system-mpi"
    system_mpi = system_root / "bin" / "mpiexec"
    _fake_incompatible_mpi_launcher(system_mpi)
    python = tmp_path / "runtime" / "python"
    _fake_python(python)
    base, prefix = _fake_conda_base(tmp_path)
    conda_mpi = prefix / "bin" / "mpiexec"
    _fake_mpi_launcher(conda_mpi)
    environment = _base_environment()
    environment.update(
        {
            "RUNTIME_ENV": "conda",
            "CONDA_BASE": str(base),
            "FAKE_CONDA_PREFIX": str(prefix),
            "PYTHON_BIN": str(python),
            "MPI_HOME": str(system_root),
            "MPI_PREFERENCE": "auto",
            "PREFLIGHT_ONLY": "1",
            "CAPTURE_FILE": str(capture),
            "OUTPUT_DIR": str(tmp_path / "output"),
        }
    )
    environment["ALLOW_EXTERNAL_MPI"] = "1"

    result = _run("--ranks", "4", environment=environment)

    assert result.returncode == 0, result.stderr
    assert "MPI candidate did not pass; trying the next" in result.stdout
    assert str(conda_mpi) in result.stdout
    assert capture.read_text(encoding="utf-8").splitlines()[:2] == ["-n", "4"]


def test_failed_external_full_preflight_retries_with_conda(tmp_path: Path) -> None:
    capture = tmp_path / "launcher-argv.txt"
    system_mpi = tmp_path / "system" / "mpiexec"
    _fake_probe_only_mpi_launcher(system_mpi)
    python = tmp_path / "runtime" / "python"
    _fake_python(python)
    base, prefix = _fake_conda_base(tmp_path)
    conda_mpi = prefix / "bin" / "mpiexec"
    _fake_mpi_launcher(conda_mpi)
    environment = _base_environment()
    environment.update(
        {
            "RUNTIME_ENV": "conda",
            "CONDA_BASE": str(base),
            "FAKE_CONDA_PREFIX": str(prefix),
            "PYTHON_BIN": str(python),
            "MPI_PREFERENCE": str(system_mpi),
            "PREFLIGHT_ONLY": "1",
            "CAPTURE_FILE": str(capture),
            "OUTPUT_DIR": str(tmp_path / "output"),
        }
    )

    result = _run("--ranks", "4", environment=environment)

    assert result.returncode == 0, result.stderr
    assert "failed the full preflight; testing fallback" in result.stdout
    assert f"Fallback launcher: {conda_mpi} -n 4" in result.stdout
    assert capture.read_text(encoding="utf-8").splitlines()[:2] == ["-n", "4"]


def test_force_incompatible_mpi_is_an_explicit_separate_override(
    tmp_path: Path,
) -> None:
    environment, capture = _active_environment(tmp_path)
    mpi = Path(environment["MPI_PREFERENCE"])
    _fake_incompatible_mpi_launcher(mpi)
    environment["ALLOW_EXTERNAL_MPI"] = "0"
    environment["FORCE_INCOMPATIBLE_MPI"] = "1"

    result = _run("--ranks", "4", environment=environment)

    assert result.returncode == 0, result.stderr
    assert "failed probe deliberately forced" in result.stdout
    assert capture.read_text(encoding="utf-8").splitlines()[:2] == ["-n", "4"]


def test_real_two_rank_compatibility_probe() -> None:
    pytest.importorskip("mpi4py")
    pytest.importorskip("petsc4py")
    pytest.importorskip("dolfinx")
    prefix = os.environ.get("CONDA_PREFIX")
    prefix_mpi = Path(prefix, "bin", "mpiexec") if prefix else None
    mpi = (
        str(prefix_mpi)
        if prefix_mpi is not None and prefix_mpi.is_file()
        else shutil.which("mpiexec")
    )
    if not mpi:
        pytest.skip("no MPI launcher is available")
    environment = os.environ.copy()
    environment.update(
        {
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
        }
    )

    result = subprocess.run(
        [
            mpi,
            "-n",
            "2",
            sys.executable,
            str(ROOT / "linux_cluster" / "mpi_compatibility_probe.py"),
            "--expected-ranks",
            "2",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    marker = next(
        line for line in result.stdout.splitlines() if line.startswith("MPI_COMPATIBILITY_OK ")
    )
    payload = json.loads(marker.removeprefix("MPI_COMPATIBILITY_OK "))
    assert payload["compatible"] is True
    assert payload["ranks"] == 2
