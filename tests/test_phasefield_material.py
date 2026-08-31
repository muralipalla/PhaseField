import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from phasefield_material import (
    MATERIAL_PROPERTIES,
    MaterialFileError,
    MaterialSpec,
    Profile,
    Region,
    parse_material_file,
    uniform_material_spec,
)


DEFAULTS = {
    "young_modulus": 210.0,
    "poisson_ratio": 0.30,
    "fracture_toughness": 2.7e-3,
    "length_scale": 0.030,
}


def test_uniform_spec_evaluates_both_coordinate_layouts():
    spec = uniform_material_spec({**DEFAULTS, "unrelated_config_value": 7})
    coordinates = np.array([[0.0, 0.4, 1.0], [0.0, 0.7, 1.0]])

    column_layout = spec.evaluate(coordinates, width=1.0, height=1.0)
    row_layout = spec.evaluate(coordinates.T, width=1.0, height=1.0)

    assert tuple(column_layout) == MATERIAL_PROPERTIES
    for property_name, expected in DEFAULTS.items():
        np.testing.assert_allclose(column_layout[property_name], expected)
        np.testing.assert_allclose(row_layout[property_name], expected)


def test_constant_linear_and_affine_profiles_are_vectorized():
    spec = MaterialSpec(
        profiles={
            "young_modulus": Profile("linear_x", (100.0, 300.0)),
            "poisson_ratio": Profile("linear_y", (0.20, 0.40)),
            "fracture_toughness": Profile(
                "affine", (1.0e-3, 5.0e-4, 2.5e-4)
            ),
            "length_scale": Profile("constant", (0.03,)),
        }
    )
    coordinates = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])

    values = spec.evaluate(coordinates, width=2.0, height=4.0)

    np.testing.assert_allclose(values["young_modulus"], [100.0, 200.0, 300.0])
    np.testing.assert_allclose(values["poisson_ratio"], [0.20, 0.30, 0.40])
    np.testing.assert_allclose(
        values["fracture_toughness"], [1.0e-3, 2.0e-3, 3.0e-3]
    )
    np.testing.assert_allclose(values["length_scale"], 0.03)


def test_region_priorities_masks_and_overrides():
    base = uniform_material_spec(DEFAULTS)
    spec = MaterialSpec(
        profiles=base.profiles,
        # Deliberately pass high priority first; MaterialSpec normalizes order.
        regions=(
            Region(
                "core",
                "circle",
                (0.5, 0.5, 0.1),
                20,
                {"young_modulus": 25.0},
            ),
            Region(
                "band",
                "box",
                (0.2, 0.8, 0.2, 0.9),
                10,
                {"young_modulus": 50.0, "fracture_toughness": 1.0e-3},
            ),
        ),
    )
    coordinates = np.array(
        [[0.1, 0.5, 0.5, 0.9], [0.1, 0.5, 0.8, 0.9]], dtype=float
    )

    assert [region.name for region in spec.regions] == ["band", "core"]
    masks = spec.region_masks(coordinates)
    np.testing.assert_array_equal(masks["band"], [False, True, True, False])
    np.testing.assert_array_equal(masks["core"], [False, True, False, False])

    values = spec.evaluate(coordinates, width=1.0, height=1.0)
    np.testing.assert_allclose(values["young_modulus"], [210.0, 25.0, 50.0, 210.0])
    np.testing.assert_allclose(
        values["fracture_toughness"], [2.7e-3, 1.0e-3, 1.0e-3, 2.7e-3]
    )


def test_material_spec_json_round_trip_is_lossless_and_serializable():
    spec = MaterialSpec(
        profiles=uniform_material_spec(DEFAULTS).profiles,
        regions=(
            Region("strip", "box", (0.2, 0.4, 0.0, 1.0), 3, {"young_modulus": 80.0}),
        ),
        source_path=str(Path("material.in").resolve()),
        source_sha256="a" * 64,
    )

    payload = spec.to_dict()
    encoded = json.dumps(payload)
    restored = MaterialSpec.from_dict(json.loads(encoded))

    assert restored == spec
    assert restored.to_dict() == payload


def test_parser_inherits_defaults_records_provenance_and_allows_forward_reference(
    tmp_path,
):
    source = tmp_path / "graded material.in"
    source.write_text(
        """# Commands and canonical names are case-insensitive.
PROFILE young_modulus linear_x 100.0 300.0
profile fracture_toughness affine 0.002 0.001 0.0
override Core young_modulus 40.0
region core circle 0.5 0.5 0.2 20
""",
        encoding="utf-8",
    )

    spec = parse_material_file(source, DEFAULTS)

    assert spec.source_path == str(source.resolve())
    assert spec.source_sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert spec.profiles["poisson_ratio"] == Profile("constant", (0.30,))
    assert spec.profiles["length_scale"] == Profile("constant", (0.030,))
    values = spec.evaluate(
        np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]),
        width=1.0,
        height=1.0,
    )
    np.testing.assert_allclose(values["young_modulus"], [100.0, 40.0, 300.0])
    np.testing.assert_allclose(
        values["fracture_toughness"], [0.002, 0.0025, 0.003]
    )


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (
            "profile young_modulus constant 100\n"
            "profile YOUNG_MODULUS constant 200\n",
            "duplicate profile",
        ),
        (
            "region A box 0 1 0 1 1\nregion a circle 0.5 0.5 0.1 2\n",
            "duplicate region name",
        ),
        (
            "region A box 0 1 0 1 7\nregion B circle 0.5 0.5 0.1 7\n",
            "duplicate region priority",
        ),
        (
            "region A box 0 1 0 1 1\n"
            "override A length_scale 0.03\n"
            "override a LENGTH_SCALE 0.04\n",
            "duplicate override",
        ),
        ("override missing young_modulus 10\n", "unknown region"),
        ("profile density constant 1\n", "Unknown material property"),
        ("unknown_command value\n", "unknown material command"),
    ],
)
def test_parser_rejects_duplicates_unknown_references_and_unknown_commands(
    tmp_path, contents, message
):
    source = tmp_path / "bad.in"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(MaterialFileError, match=message):
        parse_material_file(source, DEFAULTS)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("profile young_modulus linear_x 10\n", "expects 4 argument"),
        ("profile young_modulus constant nan\n", "must be finite"),
        ("region bad box 1 0 0 1 2\n", "xmin < xmax"),
        ("region bad circle 0 0 0 2\n", "positive radius"),
        ("region bad circle 0 0 1 2.5\n", "priority must be an integer"),
        ("override name young_modulus\n", "expects 3 argument"),
    ],
)
def test_parser_rejects_bad_arity_nonfinite_values_and_invalid_geometry(
    tmp_path, contents, message
):
    source = tmp_path / "bad_geometry.in"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(MaterialFileError, match=message):
        parse_material_file(source, DEFAULTS)


def test_evaluation_rejects_invalid_spatial_values_and_bad_coordinates():
    profiles = dict(uniform_material_spec(DEFAULTS).profiles)
    profiles["young_modulus"] = Profile("affine", (1.0, -2.0, 0.0))
    spec = MaterialSpec(profiles=profiles)

    with pytest.raises(ValueError, match="young_modulus must be strictly positive"):
        spec.evaluate(np.array([[0.0, 1.0], [0.0, 0.0]]), 1.0, 1.0)
    with pytest.raises(ValueError, match=r"shape \(2, n\) or \(n, m\)"):
        spec.evaluate(np.zeros((3, 1)), 1.0, 1.0)
    with pytest.raises(ValueError, match="finite values"):
        spec.region_masks(np.array([[0.0, np.nan, 1.0], [0.0, 0.5, 1.0]]))
    with pytest.raises(ValueError, match="strictly positive"):
        spec.evaluate(np.zeros((2, 0)), width=0.0, height=1.0)


@pytest.mark.parametrize(
    "update",
    [
        {"young_modulus": 0.0},
        {"poisson_ratio": 0.5},
        {"fracture_toughness": -1.0},
        {"length_scale": float("inf")},
    ],
)
def test_uniform_material_spec_rejects_invalid_defaults(update):
    with pytest.raises(ValueError):
        uniform_material_spec({**DEFAULTS, **update})


def test_empty_coordinate_partitions_are_supported():
    spec = uniform_material_spec(DEFAULTS)

    values = spec.evaluate(np.empty((2, 0)), width=1.0, height=1.0)

    assert all(array.shape == (0,) for array in values.values())
    assert spec.region_masks(np.empty((0, 2))) == {}
