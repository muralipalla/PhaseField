"""Detect affinity-visible physical cores before launching the Xeon16 suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shlex
import socket
import sys
from typing import Mapping, Sequence


def parse_rank_sweep(text: str) -> list[int]:
    """Parse a comma- or whitespace-delimited, strictly increasing rank list."""

    tokens = text.replace(",", " ").split()
    if not tokens:
        raise ValueError("rank sweep is empty")
    try:
        ranks = [int(token) for token in tokens]
    except ValueError as error:
        raise ValueError("rank sweep must contain integers") from error
    if any(rank < 1 for rank in ranks):
        raise ValueError("rank sweep values must be positive")
    if ranks != sorted(set(ranks)):
        raise ValueError("rank sweep must be unique and strictly increasing")
    return ranks


def _environment_positive_integer(
    environment: Mapping[str, str], name: str
) -> int | None:
    value = environment.get(name)
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def scheduler_capacities(
    environment: Mapping[str, str] | None = None,
) -> dict[str, int]:
    """Return scheduler task limits visible to this process."""

    environment = os.environ if environment is None else environment
    result: dict[str, int] = {}
    for variable in ("SLURM_NTASKS", "PBS_NP"):
        value = _environment_positive_integer(environment, variable)
        if value is not None:
            result[variable] = value

    nodefile_text = environment.get("PBS_NODEFILE")
    if nodefile_text:
        nodefile = Path(nodefile_text)
        if nodefile.is_file():
            slots = sum(
                bool(line.strip())
                for line in nodefile.read_text(encoding="utf-8").splitlines()
            )
            if slots > 0:
                result["PBS_NODEFILE"] = slots
    return result


def physical_core_topology(
    cpu_ids: Sequence[int],
    sysfs_root: Path = Path("/sys/devices/system/cpu"),
) -> tuple[set[tuple[int, int]], list[int]]:
    """Map logical CPUs to unique (socket, core) pairs using Linux sysfs."""

    cores: set[tuple[int, int]] = set()
    missing: list[int] = []
    for cpu_id in cpu_ids:
        topology = sysfs_root / f"cpu{cpu_id}" / "topology"
        try:
            package_id = int(
                (topology / "physical_package_id").read_text(
                    encoding="utf-8"
                ).strip()
            )
            core_id = int(
                (topology / "core_id").read_text(encoding="utf-8").strip()
            )
        except (OSError, ValueError):
            missing.append(cpu_id)
            continue
        cores.add((package_id, core_id))
    return cores, missing


def _cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return None
    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.casefold().startswith("model name") and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def _memory_kib() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        return None
    for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("MemTotal:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1])
                except ValueError:
                    return None
    return None


def detect_hardware(
    *,
    environment: Mapping[str, str] | None = None,
    affinity_cpu_ids: Sequence[int] | None = None,
    sysfs_root: Path = Path("/sys/devices/system/cpu"),
    allow_unknown_topology: bool = False,
) -> dict[str, object]:
    """Build a non-secret workstation/allocation-capacity report."""

    environment = os.environ if environment is None else environment
    if affinity_cpu_ids is None:
        if not hasattr(os, "sched_getaffinity"):
            raise RuntimeError("os.sched_getaffinity is unavailable; Linux is required")
        affinity_cpu_ids = sorted(os.sched_getaffinity(0))
    else:
        affinity_cpu_ids = sorted(set(int(value) for value in affinity_cpu_ids))
    if not affinity_cpu_ids:
        raise RuntimeError("the process has an empty CPU affinity mask")

    physical_cores, missing = physical_core_topology(
        affinity_cpu_ids, sysfs_root=sysfs_root
    )
    topology_complete = not missing and bool(physical_cores)
    if not topology_complete and not allow_unknown_topology:
        raise RuntimeError(
            "physical-core topology is incomplete for affinity CPUs "
            f"{missing}; pass --allow-unknown-topology only if this is expected"
        )
    physical_capacity = (
        len(physical_cores) if topology_complete else len(affinity_cpu_ids)
    )
    scheduler = scheduler_capacities(environment)
    effective_capacity = min(
        [physical_capacity, *scheduler.values()]
        if scheduler
        else [physical_capacity]
    )
    return {
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "memory_kib": _memory_kib(),
        "affinity_cpu_ids": list(affinity_cpu_ids),
        "affinity_logical_cpus": len(affinity_cpu_ids),
        "physical_core_ids": [
            {"socket": package, "core": core}
            for package, core in sorted(physical_cores)
        ],
        "physical_core_capacity": physical_capacity,
        "physical_topology_complete": topology_complete,
        "topology_missing_cpu_ids": missing,
        "scheduler_capacities": scheduler,
        "effective_rank_capacity": effective_capacity,
    }


def _atomic_write_json(path: Path, payload: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate physical-core capacity for the Xeon16 MPI suite."
    )
    parser.add_argument("--max-ranks", type=int, default=16)
    parser.add_argument("--rank-sweep", default="1 2 4 8 16")
    parser.add_argument("--require-sweep-endpoints", action="store_true")
    parser.add_argument("--mpi-extra-args", default="")
    parser.add_argument("--allow-unknown-topology", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if platform.system() != "Linux":
        raise SystemExit("Xeon16 hardware detection must run on Linux.")
    if args.max_ranks < 1:
        raise SystemExit("--max-ranks must be positive.")
    try:
        ranks = parse_rank_sweep(args.rank_sweep)
        report = detect_hardware(
            allow_unknown_topology=args.allow_unknown_topology
        )
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    if args.require_sweep_endpoints and (
        ranks[0] != 1 or ranks[-1] != args.max_ranks
    ):
        raise SystemExit(
            "the scaling rank sweep must start at 1 and end at --max-ranks"
        )
    if max(ranks) > args.max_ranks:
        raise SystemExit("the rank sweep exceeds --max-ranks")
    capacity = int(report["effective_rank_capacity"])
    if args.max_ranks > capacity:
        raise SystemExit(
            f"requested {args.max_ranks} ranks but only {capacity} "
            "affinity-visible physical core(s)/scheduler slot(s) are available"
        )
    try:
        mpi_tokens = [token.casefold() for token in shlex.split(args.mpi_extra_args)]
    except ValueError as error:
        raise SystemExit(f"cannot parse MPI_EXTRA_ARGS: {error}") from error
    oversubscribe_requested = any(
        token in {"--oversubscribe", "-oversubscribe"}
        or token.endswith(":oversubscribe")
        for token in mpi_tokens
    )
    if oversubscribe_requested:
        raise SystemExit(
            "MPI_EXTRA_ARGS requests oversubscription, which is not permitted by "
            "the Xeon16 validation/benchmark suite"
        )

    report.update(
        {
            "requested_max_ranks": args.max_ranks,
            "requested_rank_sweep": ranks,
            "oversubscribe_requested": oversubscribe_requested,
            "capacity_check": "passed",
        }
    )
    _atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
