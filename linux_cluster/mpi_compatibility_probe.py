"""Small collective probe used to select an MPI launcher safely.

This deliberately runs before the full finite-element/XDMF preflight.  It
checks that mpi4py and PETSc see congruent world communicators when started by
the candidate launcher.  A successful probe is necessary but not sufficient;
``check_environment.py`` remains the final runtime gate.
"""

from __future__ import annotations

import argparse
import json
import sys

from mpi4py import MPI
import mpi4py
import numpy as np
from petsc4py import PETSc
import petsc4py

import dolfinx


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check launcher compatibility with the active MPI/PETSc stack."
    )
    parser.add_argument("--expected-ranks", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    comm = MPI.COMM_WORLD
    local_errors: list[str] = []

    if args.expected_ranks < 1:
        local_errors.append("--expected-ranks must be positive")
    elif comm.size != args.expected_ranks:
        local_errors.append(
            f"MPI world has {comm.size} rank(s), expected {args.expected_ranks}"
        )

    try:
        petsc_comm = PETSc.COMM_WORLD.tompi4py()
        relation = MPI.Comm.Compare(comm, petsc_comm)
        if relation not in {MPI.IDENT, MPI.CONGRUENT}:
            local_errors.append(
                "PETSc.COMM_WORLD is not congruent with mpi4py MPI.COMM_WORLD "
                f"(comparison={relation})"
            )
    except Exception as error:  # pragma: no cover - depends on external MPI
        local_errors.append(f"PETSc/mpi4py communicator comparison failed: {error}")

    if np.issubdtype(np.dtype(PETSc.ScalarType), np.complexfloating):
        local_errors.append("PETSc uses complex scalars; the solver requires real scalars")

    try:
        reduced = comm.allreduce(comm.rank + 1, op=MPI.SUM)
        expected = comm.size * (comm.size + 1) // 2
        if reduced != expected:
            local_errors.append(
                f"MPI allreduce returned {reduced}, expected {expected}"
            )
    except Exception as error:  # pragma: no cover - depends on external MPI
        local_errors.append(f"MPI allreduce failed: {error}")

    gathered = comm.gather(local_errors, root=0)
    failed = comm.allreduce(bool(local_errors), op=MPI.LOR)
    if failed:
        if comm.rank == 0:
            for rank, errors in enumerate(gathered):
                for error in errors:
                    print(f"rank {rank}: {error}", file=sys.stderr)
        return 1

    if comm.rank == 0:
        payload = {
            "compatible": True,
            "ranks": comm.size,
            "mpi_library": MPI.Get_library_version().strip(),
            "mpi4py": mpi4py.__version__,
            "petsc": ".".join(str(value) for value in PETSc.Sys.getVersion()),
            "petsc4py": petsc4py.__version__,
            "dolfinx": dolfinx.__version__,
        }
        print("MPI_COMPATIBILITY_OK " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
