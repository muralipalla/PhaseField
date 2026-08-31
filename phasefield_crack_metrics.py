"""Backend-independent connected-crack diagnostics.

The serial and PETSc phase-field solvers both represent the phase topology as
one coordinate row and one neighbor array per phase degree of freedom.  This
module deliberately depends only on NumPy so either backend can use the same
diagnostic implementation after it has formed those arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike


@dataclass(frozen=True)
class CrackMetrics:
    """Geometric diagnostics for the damaged component attached to a notch."""

    front_x: float
    extension_x: float
    tip_x: float
    tip_y: float
    path_length: float
    kink_angle_degrees: float
    connected_node_count: int


def _seed_indices(seed_nodes: ArrayLike, node_count: int) -> np.ndarray:
    """Normalize a Boolean seed mask or integer seed indices."""

    raw = np.asarray(seed_nodes)
    if raw.ndim == 0:
        if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(
            raw.dtype, np.integer
        ):
            raise ValueError(
                "seed_nodes must be a one-dimensional Boolean mask or "
                "integer index array."
            )
        raw = raw.reshape(1)
    if raw.ndim != 1:
        raise ValueError(
            "seed_nodes must be a one-dimensional Boolean mask or integer "
            "index array."
        )

    if np.issubdtype(raw.dtype, np.bool_):
        if raw.size != node_count:
            raise ValueError(
                "A Boolean seed_nodes mask must have one entry per phase node."
            )
        return np.flatnonzero(raw).astype(np.int64, copy=False)

    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError("seed_nodes indices must have an integer dtype.")

    indices = np.unique(raw.astype(np.int64, copy=False))
    if np.any(indices < 0) or np.any(indices >= node_count):
        raise ValueError("seed_nodes contains an out-of-range node index.")
    return indices


def _neighbor_arrays(
    neighbors: Iterable[ArrayLike], node_count: int
) -> tuple[np.ndarray, ...]:
    """Validate and normalize the per-node neighbor arrays."""

    try:
        rows = tuple(neighbors)
    except TypeError as error:
        raise ValueError("neighbors must be an iterable with one row per node.") from error
    if len(rows) != node_count:
        raise ValueError("neighbors must contain exactly one row per phase node.")

    normalized: list[np.ndarray] = []
    for node, row in enumerate(rows):
        array = np.asarray(row)
        if array.ndim != 1:
            raise ValueError(f"neighbors[{node}] must be one-dimensional.")
        if array.size == 0:
            normalized.append(np.empty(0, dtype=np.int64))
            continue
        if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
            array.dtype, np.integer
        ):
            raise ValueError(f"neighbors[{node}] must contain integer indices.")
        indices = np.unique(array.astype(np.int64, copy=False))
        if np.any(indices < 0) or np.any(indices >= node_count):
            raise ValueError(f"neighbors[{node}] contains an out-of-range index.")
        normalized.append(indices)
    return tuple(normalized)


def _choose_tip(
    candidates: np.ndarray,
    distances: np.ndarray,
    coordinates: np.ndarray,
    tolerance: float,
) -> int:
    """Choose a farthest geodesic node with deterministic geometric ties."""

    candidate_distances = distances[candidates]
    maximum_distance = float(np.max(candidate_distances))
    tied = candidates[candidate_distances >= maximum_distance - tolerance]

    maximum_x = float(np.max(coordinates[tied, 0]))
    tied = tied[coordinates[tied, 0] >= maximum_x - tolerance]

    minimum_y = float(np.min(coordinates[tied, 1]))
    tied = tied[coordinates[tied, 1] <= minimum_y + tolerance]

    # Duplicate coordinate rows can remain after the geometric tie-break.  The
    # compact global index is then the final, explicitly deterministic key.
    return int(np.min(tied))


def compute_crack_metrics(
    phase_values: ArrayLike,
    coordinates: ArrayLike,
    neighbors: Iterable[ArrayLike],
    seed_nodes: ArrayLike,
    threshold: float,
    represented_notch_tip: tuple[float, float] | ArrayLike,
    tolerance: float = 1.0e-12,
) -> CrackMetrics:
    """Measure the damaged component connected to the starter notch.

    Parameters
    ----------
    phase_values:
        One scalar intactness value per phase node.  Nodes satisfying
        ``phase_values <= threshold`` are damaged.
    coordinates:
        An ``(n, 2)`` array, or an ``(n, m)`` array with ``m >= 2`` for
        DOLFINx coordinates embedded in a higher-dimensional storage array.
        Only x and y are used.
    neighbors:
        One one-dimensional integer neighbor array/list per node.  The current
        solver topologies are undirected; callers supplying another topology
        must include every traversal direction that should be reachable.
    seed_nodes:
        Integer notch-node indices, a single integer index, or a Boolean mask
        with one entry per node.  Only damaged seed nodes initiate a component.
    threshold:
        Inclusive damage threshold.
    represented_notch_tip:
        The represented starter-notch tip ``(x, y)`` used as the origin for
        extension and kink diagnostics.
    tolerance:
        Nonnegative geometric tolerance used for extension and deterministic
        tie decisions.

    Notes
    -----
    Dijkstra distances use Euclidean mesh-edge lengths and are initialized to
    zero at every damaged seed node.  Once the component has advanced beyond
    the represented notch tip in x, the reported tip is the connected node with
    greatest shortest-path distance.  Ties select greatest x, then smallest y,
    then smallest compact node index.  ``front_x`` remains the maximum x over
    the entire connected damaged component, preserving the legacy diagnostic.
    """

    values = np.asarray(phase_values, dtype=float)
    if values.ndim != 1:
        raise ValueError("phase_values must be one-dimensional.")
    if not np.all(np.isfinite(values)):
        raise ValueError("phase_values must contain only finite values.")
    node_count = int(values.size)

    coordinate_array = np.asarray(coordinates, dtype=float)
    if (
        coordinate_array.ndim != 2
        or coordinate_array.shape[0] != node_count
        or coordinate_array.shape[1] < 2
    ):
        raise ValueError(
            "coordinates must have shape (number_of_phase_nodes, m) with m >= 2."
        )
    xy = np.ascontiguousarray(coordinate_array[:, :2])
    if not np.all(np.isfinite(xy)):
        raise ValueError("coordinates must contain only finite x and y values.")

    threshold_value = float(threshold)
    if not math.isfinite(threshold_value):
        raise ValueError("threshold must be finite.")
    tolerance_value = float(tolerance)
    if not math.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError("tolerance must be finite and nonnegative.")

    notch_tip = np.asarray(represented_notch_tip, dtype=float)
    if notch_tip.shape != (2,) or not np.all(np.isfinite(notch_tip)):
        raise ValueError("represented_notch_tip must contain two finite values.")
    notch_tip_x = float(notch_tip[0])
    notch_tip_y = float(notch_tip[1])

    adjacency = _neighbor_arrays(neighbors, node_count)
    seeds = _seed_indices(seed_nodes, node_count)
    damaged = values <= threshold_value
    damaged_seeds = seeds[damaged[seeds]]

    distances = np.full(node_count, math.inf, dtype=float)
    queue: list[tuple[float, int]] = []
    for seed in damaged_seeds:
        seed_index = int(seed)
        distances[seed_index] = 0.0
        heapq.heappush(queue, (0.0, seed_index))

    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances[node]:
            continue
        for neighbor_value in adjacency[node]:
            neighbor = int(neighbor_value)
            if not damaged[neighbor]:
                continue
            edge_length = math.hypot(
                float(xy[neighbor, 0] - xy[node, 0]),
                float(xy[neighbor, 1] - xy[node, 1]),
            )
            candidate_distance = distance + edge_length
            if candidate_distance < distances[neighbor]:
                distances[neighbor] = candidate_distance
                heapq.heappush(queue, (candidate_distance, neighbor))

    connected = np.isfinite(distances)
    connected_indices = np.flatnonzero(connected)
    connected_node_count = int(connected_indices.size)
    if connected_node_count == 0:
        return CrackMetrics(
            front_x=0.0,
            extension_x=0.0,
            tip_x=notch_tip_x,
            tip_y=notch_tip_y,
            path_length=0.0,
            kink_angle_degrees=0.0,
            connected_node_count=0,
        )

    front_x = float(np.max(xy[connected_indices, 0]))
    raw_extension = max(0.0, front_x - notch_tip_x)
    if raw_extension <= tolerance_value:
        return CrackMetrics(
            front_x=front_x,
            extension_x=0.0,
            tip_x=notch_tip_x,
            tip_y=notch_tip_y,
            path_length=0.0,
            kink_angle_degrees=0.0,
            connected_node_count=connected_node_count,
        )

    tip_index = _choose_tip(
        connected_indices, distances, xy, tolerance_value
    )
    tip_x = float(xy[tip_index, 0])
    tip_y = float(xy[tip_index, 1])
    path_length = float(distances[tip_index])
    kink_angle = (
        math.degrees(math.atan2(tip_y - notch_tip_y, tip_x - notch_tip_x))
        if path_length > tolerance_value
        else 0.0
    )

    return CrackMetrics(
        front_x=front_x,
        extension_x=raw_extension,
        tip_x=tip_x,
        tip_y=tip_y,
        path_length=path_length,
        kink_angle_degrees=kink_angle,
        connected_node_count=connected_node_count,
    )


__all__ = ["CrackMetrics", "compute_crack_metrics"]
