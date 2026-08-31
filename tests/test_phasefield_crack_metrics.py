from __future__ import annotations

import math

import numpy as np
import pytest

from phasefield_crack_metrics import CrackMetrics, compute_crack_metrics


def _line_neighbors(node_count: int) -> tuple[np.ndarray, ...]:
    return tuple(
        np.asarray(
            [candidate for candidate in (node - 1, node + 1) if 0 <= candidate < node_count],
            dtype=np.int32,
        )
        for node in range(node_count)
    )


def test_straight_connected_crack_preserves_front_and_extension() -> None:
    metrics = compute_crack_metrics(
        phase_values=np.asarray([0.0, 0.0, 0.1, 0.2, 0.9]),
        coordinates=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
        ),
        neighbors=_line_neighbors(5),
        seed_nodes=np.asarray([0, 1], dtype=np.int32),
        threshold=0.2,
        represented_notch_tip=(1.0, 0.0),
        tolerance=1.0e-12,
    )

    assert isinstance(metrics, CrackMetrics)
    assert metrics.front_x == pytest.approx(3.0)
    assert metrics.extension_x == pytest.approx(2.0)
    assert (metrics.tip_x, metrics.tip_y) == pytest.approx((3.0, 0.0))
    assert metrics.path_length == pytest.approx(2.0)
    assert metrics.kink_angle_degrees == pytest.approx(0.0)
    assert metrics.connected_node_count == 4


def test_disconnected_damaged_island_is_excluded() -> None:
    metrics = compute_crack_metrics(
        phase_values=np.zeros(5),
        coordinates=np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [10.0, 4.0], [11.0, 4.0]]
        ),
        neighbors=([1], [0, 2], [1], [4], [3]),
        seed_nodes=[0, 1],
        threshold=0.2,
        represented_notch_tip=(1.0, 0.0),
    )

    assert metrics.front_x == pytest.approx(2.0)
    assert metrics.extension_x == pytest.approx(1.0)
    assert (metrics.tip_x, metrics.tip_y) == pytest.approx((2.0, 0.0))
    assert metrics.connected_node_count == 3


def test_dijkstra_tip_uses_longest_shortest_path_not_maximum_x() -> None:
    coordinates = np.asarray(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.5, 1.0], [1.0, 2.0]]
    )
    metrics = compute_crack_metrics(
        phase_values=np.zeros(5),
        coordinates=coordinates,
        neighbors=([1, 3], [0, 2], [1], [0, 4], [3]),
        seed_nodes=[0],
        threshold=0.2,
        represented_notch_tip=(0.0, 0.0),
    )

    expected_path = 2.0 * math.hypot(0.5, 1.0)
    assert metrics.front_x == pytest.approx(2.0)
    assert metrics.extension_x == pytest.approx(2.0)
    assert (metrics.tip_x, metrics.tip_y) == pytest.approx((1.0, 2.0))
    assert metrics.path_length == pytest.approx(expected_path)
    assert metrics.kink_angle_degrees == pytest.approx(math.degrees(math.atan2(2.0, 1.0)))


def test_geometric_tie_break_is_independent_of_neighbor_order() -> None:
    coordinates = np.asarray([[0.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
    arguments = {
        "phase_values": np.zeros(3),
        "coordinates": coordinates,
        "seed_nodes": np.asarray([True, False, False]),
        "threshold": 0.2,
        "represented_notch_tip": (0.0, 0.0),
    }

    first = compute_crack_metrics(neighbors=([1, 2], [0], [0]), **arguments)
    second = compute_crack_metrics(neighbors=([2, 1], [0], [0]), **arguments)

    assert first == second
    assert (first.tip_x, first.tip_y) == pytest.approx((1.0, -1.0))
    assert first.kink_angle_degrees == pytest.approx(-45.0)


def test_no_forward_extension_resets_tip_path_and_kink() -> None:
    metrics = compute_crack_metrics(
        phase_values=np.zeros(3),
        coordinates=np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]),
        neighbors=([1], [0, 2], [1]),
        seed_nodes=[0, 1],
        threshold=0.2,
        represented_notch_tip=(1.0, 0.0),
    )

    assert metrics.front_x == pytest.approx(1.0)
    assert metrics.extension_x == 0.0
    assert (metrics.tip_x, metrics.tip_y) == (1.0, 0.0)
    assert metrics.path_length == 0.0
    assert metrics.kink_angle_degrees == 0.0
    assert metrics.connected_node_count == 3


def test_embedded_coordinates_threshold_and_damaged_seed_mask() -> None:
    metrics = compute_crack_metrics(
        phase_values=[0.2, 0.2, 0.20001],
        coordinates=[[0.0, 0.0, 7.0], [1.0, 0.5, 8.0], [2.0, 1.0, 9.0]],
        neighbors=([1], [0, 2], [1]),
        seed_nodes=[True, False, False],
        threshold=0.2,
        represented_notch_tip=(0.0, 0.0),
    )

    assert metrics.front_x == pytest.approx(1.0)
    assert metrics.connected_node_count == 2
    assert metrics.path_length == pytest.approx(math.hypot(1.0, 0.5))
    assert metrics.kink_angle_degrees == pytest.approx(math.degrees(math.atan2(0.5, 1.0)))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"phase_values": [[0.0, 0.0]]}, "phase_values"),
        ({"coordinates": [[0.0], [1.0]]}, "coordinates"),
        ({"neighbors": ([1],)}, "one row"),
        ({"neighbors": ([2], [0])}, "out-of-range"),
        ({"seed_nodes": [2]}, "out-of-range"),
        ({"seed_nodes": [True]}, "one entry"),
        ({"threshold": math.inf}, "threshold"),
        ({"tolerance": -1.0}, "tolerance"),
        ({"represented_notch_tip": (0.0, math.nan)}, "represented_notch_tip"),
    ],
)
def test_invalid_inputs_are_rejected(overrides: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "phase_values": [0.0, 0.0],
        "coordinates": [[0.0, 0.0], [1.0, 0.0]],
        "neighbors": ([1], [0]),
        "seed_nodes": [0],
        "threshold": 0.2,
        "represented_notch_tip": (0.0, 0.0),
        "tolerance": 1.0e-12,
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        compute_crack_metrics(**arguments)


def test_no_damaged_seed_returns_legacy_empty_front() -> None:
    metrics = compute_crack_metrics(
        phase_values=[1.0, 0.0],
        coordinates=[[0.0, 0.0], [5.0, 0.0]],
        neighbors=([1], [0]),
        seed_nodes=[0],
        threshold=0.2,
        represented_notch_tip=(1.0, 2.0),
    )

    assert metrics == CrackMetrics(
        front_x=0.0,
        extension_x=0.0,
        tip_x=1.0,
        tip_y=2.0,
        path_length=0.0,
        kink_angle_degrees=0.0,
        connected_node_count=0,
    )
