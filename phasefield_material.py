"""Spatial material definitions shared by the serial and PETSc backends.

The module deliberately has no FEniCSx dependency.  Material files are parsed
once into a JSON-serializable :class:`MaterialSpec`, which can be broadcast to
MPI ranks and evaluated vectorially at local interpolation coordinates.
"""

from __future__ import annotations

import hashlib
import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


MATERIAL_PROPERTIES = (
    "young_modulus",
    "poisson_ratio",
    "fracture_toughness",
    "length_scale",
)

__all__ = [
    "MATERIAL_PROPERTIES",
    "MaterialFileError",
    "Profile",
    "Region",
    "MaterialSpec",
    "uniform_material_spec",
    "parse_material_file",
]

_PROFILE_ARITIES = {
    "constant": 1,
    "linear_x": 2,
    "linear_y": 2,
    "affine": 3,
}
_REGION_ARITIES = {"box": 4, "circle": 3}


class MaterialFileError(ValueError):
    """A user-facing material-file syntax or validation error."""


def _finite_float(value: Any, description: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{description} must be a finite number, not a boolean.")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{description} must be a finite number.") from error
    if not math.isfinite(converted):
        raise ValueError(f"{description} must be finite.")
    return converted


def _property_name(value: Any, description: str = "material property") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a nonempty string.")
    normalized = value.casefold()
    if normalized not in MATERIAL_PROPERTIES:
        accepted = ", ".join(MATERIAL_PROPERTIES)
        raise ValueError(
            f"Unknown {description} '{value}'; expected one of: {accepted}."
        )
    return normalized


def _validate_property_values(property_name: str, values: np.ndarray) -> None:
    """Validate physical admissibility for a scalar or evaluated array."""

    if not np.all(np.isfinite(values)):
        raise ValueError(f"Evaluated {property_name} contains a nonfinite value.")
    if property_name == "poisson_ratio":
        if np.any((values <= -0.99) | (values >= 0.5)):
            raise ValueError(
                "Evaluated poisson_ratio must lie strictly between -0.99 and 0.5."
            )
    elif np.any(values <= 0.0):
        raise ValueError(f"Evaluated {property_name} must be strictly positive.")


def _coordinate_components(coordinates: Any) -> tuple[np.ndarray, np.ndarray]:
    try:
        array = np.asarray(coordinates, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("coordinates must be a numeric two-dimensional array.") from error
    if array.ndim != 2:
        raise ValueError(
            "coordinates must have shape (2, n) or (n, 2); "
            f"received an array with {array.ndim} dimensions."
        )
    # DOLFINx interpolation callbacks use (geometric_dimension, point_count),
    # so that convention wins for the inherently ambiguous (2, 2) case.
    if array.shape[0] == 2:
        x, y = array[0], array[1]
    elif array.shape[1] >= 2:
        x, y = array[:, 0], array[:, 1]
    else:
        raise ValueError(
            "coordinates must have shape (2, n) or (n, m) with m >= 2; "
            f"received {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("coordinates must contain only finite values.")
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


@dataclass(frozen=True)
class Profile:
    """One domain-wide scalar profile.

    ``affine`` values mean ``(value_at_origin, derivative_x, derivative_y)``.
    Linear profiles store values at the two opposite specimen boundaries.
    """

    kind: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("Profile kind must be a nonempty string.")
        kind = self.kind.casefold()
        if kind not in _PROFILE_ARITIES:
            accepted = ", ".join(_PROFILE_ARITIES)
            raise ValueError(
                f"Unknown profile kind '{self.kind}'; expected one of: {accepted}."
            )
        try:
            raw_values = tuple(self.values)
        except TypeError as error:
            raise ValueError("Profile values must be a finite numeric sequence.") from error
        expected = _PROFILE_ARITIES[kind]
        if len(raw_values) != expected:
            raise ValueError(
                f"Profile kind '{kind}' expects {expected} value(s), "
                f"got {len(raw_values)}."
            )
        values = tuple(
            _finite_float(value, f"Profile '{kind}' value") for value in raw_values
        )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "values", values)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "values": list(self.values)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Profile":
        if not isinstance(data, Mapping):
            raise ValueError("A serialized profile must be an object.")
        unknown = set(data) - {"kind", "values"}
        if unknown:
            raise ValueError(
                "Unknown serialized profile field(s): " + ", ".join(sorted(unknown))
            )
        if "kind" not in data or "values" not in data:
            raise ValueError("A serialized profile requires 'kind' and 'values'.")
        return cls(kind=data["kind"], values=data["values"])


@dataclass(frozen=True)
class Region:
    """A box or circle whose constant overrides have an explicit priority."""

    name: str
    shape: str
    parameters: tuple[float, ...]
    priority: int
    overrides: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Region name must be a nonempty string.")
        if not isinstance(self.shape, str) or not self.shape.strip():
            raise ValueError("Region shape must be a nonempty string.")
        shape = self.shape.casefold()
        if shape not in _REGION_ARITIES:
            accepted = ", ".join(_REGION_ARITIES)
            raise ValueError(
                f"Unknown region shape '{self.shape}'; expected one of: {accepted}."
            )
        try:
            raw_parameters = tuple(self.parameters)
        except TypeError as error:
            raise ValueError("Region parameters must be a finite numeric sequence.") from error
        expected = _REGION_ARITIES[shape]
        if len(raw_parameters) != expected:
            raise ValueError(
                f"Region shape '{shape}' expects {expected} parameter(s), "
                f"got {len(raw_parameters)}."
            )
        parameters = tuple(
            _finite_float(value, f"Region '{self.name}' parameter")
            for value in raw_parameters
        )
        if shape == "box":
            xmin, xmax, ymin, ymax = parameters
            if not xmin < xmax or not ymin < ymax:
                raise ValueError(
                    f"Box region '{self.name}' requires xmin < xmax and ymin < ymax."
                )
        elif parameters[2] <= 0.0:
            raise ValueError(f"Circle region '{self.name}' requires a positive radius.")

        if isinstance(self.priority, (bool, np.bool_)) or not isinstance(
            self.priority, (int, np.integer)
        ):
            raise ValueError(f"Region '{self.name}' priority must be an integer.")

        if not isinstance(self.overrides, Mapping):
            raise ValueError(f"Region '{self.name}' overrides must be a mapping.")
        normalized_overrides: dict[str, float] = {}
        for raw_property, raw_value in self.overrides.items():
            property_name = _property_name(raw_property)
            if property_name in normalized_overrides:
                raise ValueError(
                    f"Region '{self.name}' has a duplicate override for "
                    f"'{property_name}'."
                )
            value = _finite_float(
                raw_value, f"Region '{self.name}' override '{property_name}'"
            )
            _validate_property_values(property_name, np.asarray([value]))
            normalized_overrides[property_name] = value

        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "priority", int(self.priority))
        object.__setattr__(
            self, "overrides", MappingProxyType(normalized_overrides)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "parameters": list(self.parameters),
            "priority": self.priority,
            "overrides": dict(self.overrides),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Region":
        if not isinstance(data, Mapping):
            raise ValueError("A serialized region must be an object.")
        required = {"name", "shape", "parameters", "priority", "overrides"}
        unknown = set(data) - required
        if unknown:
            raise ValueError(
                "Unknown serialized region field(s): " + ", ".join(sorted(unknown))
            )
        missing = required - set(data)
        if missing:
            raise ValueError(
                "Missing serialized region field(s): " + ", ".join(sorted(missing))
            )
        return cls(
            name=data["name"],
            shape=data["shape"],
            parameters=data["parameters"],
            priority=data["priority"],
            overrides=data["overrides"],
        )


@dataclass(frozen=True)
class MaterialSpec:
    """Complete, immutable, and broadcast-safe spatial material definition."""

    profiles: Mapping[str, Profile]
    regions: tuple[Region, ...] = ()
    source_path: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.profiles, Mapping):
            raise ValueError("Material profiles must be a mapping.")
        normalized_profiles: dict[str, Profile] = {}
        for raw_property, raw_profile in self.profiles.items():
            property_name = _property_name(raw_property)
            if property_name in normalized_profiles:
                raise ValueError(f"Duplicate material profile '{property_name}'.")
            profile = (
                raw_profile
                if isinstance(raw_profile, Profile)
                else Profile.from_dict(raw_profile)
            )
            normalized_profiles[property_name] = profile

        missing = set(MATERIAL_PROPERTIES) - set(normalized_profiles)
        if missing:
            raise ValueError(
                "Missing material profile(s): " + ", ".join(sorted(missing))
            )
        normalized_profiles = {
            name: normalized_profiles[name] for name in MATERIAL_PROPERTIES
        }

        try:
            regions = tuple(
                region if isinstance(region, Region) else Region.from_dict(region)
                for region in self.regions
            )
        except TypeError as error:
            raise ValueError("Material regions must be a sequence.") from error
        names: set[str] = set()
        priorities: set[int] = set()
        for region in regions:
            normalized_name = region.name.casefold()
            if normalized_name in names:
                raise ValueError(f"Duplicate region name '{region.name}'.")
            if region.priority in priorities:
                raise ValueError(f"Duplicate region priority {region.priority}.")
            names.add(normalized_name)
            priorities.add(region.priority)
        regions = tuple(sorted(regions, key=lambda region: region.priority))

        # Endpoints of linear profiles and the origin of affine profiles are
        # guaranteed evaluation points and can be checked before a mesh exists.
        for property_name, profile in normalized_profiles.items():
            if profile.kind in {"constant", "linear_x", "linear_y"}:
                checked_values = profile.values
            else:
                checked_values = profile.values[:1]
            _validate_property_values(
                property_name, np.asarray(checked_values, dtype=float)
            )

        if self.source_path is not None and not isinstance(self.source_path, str):
            raise ValueError("source_path must be a string or None.")
        if self.source_sha256 is not None:
            if not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64:
                raise ValueError("source_sha256 must be a 64-character hexadecimal string.")
            try:
                int(self.source_sha256, 16)
            except ValueError as error:
                raise ValueError(
                    "source_sha256 must be a 64-character hexadecimal string."
                ) from error

        object.__setattr__(
            self, "profiles", MappingProxyType(normalized_profiles)
        )
        object.__setattr__(self, "regions", regions)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable representation."""

        return {
            "profiles": {
                property_name: self.profiles[property_name].to_dict()
                for property_name in MATERIAL_PROPERTIES
            },
            "regions": [region.to_dict() for region in self.regions],
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MaterialSpec":
        """Reconstruct a specification received through JSON or MPI."""

        if not isinstance(data, Mapping):
            raise ValueError("A serialized material specification must be an object.")
        allowed = {"profiles", "regions", "source_path", "source_sha256"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(
                "Unknown serialized material field(s): " + ", ".join(sorted(unknown))
            )
        if "profiles" not in data:
            raise ValueError("A serialized material specification requires 'profiles'.")
        return cls(
            profiles=data["profiles"],
            regions=data.get("regions", ()),
            source_path=data.get("source_path"),
            source_sha256=data.get("source_sha256"),
        )

    def region_masks(self, coordinates: Any) -> dict[str, np.ndarray]:
        """Return each named region's geometric membership mask.

        The public masks let MPI callers count owned DG0 cells per region via
        global reductions without duplicating the geometry rules.
        """

        x, y = _coordinate_components(coordinates)
        masks: dict[str, np.ndarray] = {}
        for region in self.regions:
            if region.shape == "box":
                xmin, xmax, ymin, ymax = region.parameters
                mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
            else:
                center_x, center_y, radius = region.parameters
                mask = (
                    (x - center_x) * (x - center_x)
                    + (y - center_y) * (y - center_y)
                    <= radius * radius
                )
            masks[region.name] = np.asarray(mask, dtype=bool)
        return masks

    def evaluate(
        self, coordinates: Any, width: float, height: float
    ) -> dict[str, np.ndarray]:
        """Evaluate every material property at points in either supported layout."""

        x, y = _coordinate_components(coordinates)
        width = _finite_float(width, "Specimen width")
        height = _finite_float(height, "Specimen height")
        if width <= 0.0 or height <= 0.0:
            raise ValueError("Specimen width and height must be strictly positive.")

        evaluated: dict[str, np.ndarray] = {}
        with np.errstate(over="ignore", invalid="ignore"):
            for property_name in MATERIAL_PROPERTIES:
                profile = self.profiles[property_name]
                if profile.kind == "constant":
                    values = np.full(x.shape, profile.values[0], dtype=float)
                elif profile.kind == "linear_x":
                    left, right = profile.values
                    values = left + (right - left) * (x / width)
                elif profile.kind == "linear_y":
                    bottom, top = profile.values
                    values = bottom + (top - bottom) * (y / height)
                else:
                    origin, derivative_x, derivative_y = profile.values
                    values = origin + derivative_x * x + derivative_y * y
                evaluated[property_name] = np.asarray(values, dtype=float)

            region_masks = self.region_masks(np.vstack((x, y)))
            for region in self.regions:
                mask = region_masks[region.name]
                for property_name, value in region.overrides.items():
                    evaluated[property_name][mask] = value

        for property_name, values in evaluated.items():
            _validate_property_values(property_name, values)
        return evaluated


def uniform_material_spec(defaults: Mapping[str, Any]) -> MaterialSpec:
    """Build a complete constant specification from a defaults mapping."""

    if not isinstance(defaults, Mapping):
        raise ValueError("Material defaults must be a mapping.")
    normalized: dict[str, Any] = {}
    for raw_property, value in defaults.items():
        if not isinstance(raw_property, str):
            continue
        property_name = raw_property.casefold()
        if property_name not in MATERIAL_PROPERTIES:
            # This permits passing ``vars(config)`` or ``asdict(config)``.
            continue
        if property_name in normalized:
            raise ValueError(f"Duplicate material default '{property_name}'.")
        normalized[property_name] = value
    missing = set(MATERIAL_PROPERTIES) - set(normalized)
    if missing:
        raise ValueError(
            "Missing material default(s): " + ", ".join(sorted(missing))
        )
    profiles = {
        property_name: Profile("constant", (normalized[property_name],))
        for property_name in MATERIAL_PROPERTIES
    }
    return MaterialSpec(profiles=profiles)


def _tokens(line: str, source: Path, line_number: int) -> list[str]:
    lexer = shlex.shlex(line, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        return list(lexer)
    except ValueError as error:
        raise MaterialFileError(f"{source}:{line_number}: {error}.") from error


def _parse_float(token: str, description: str, source: Path, line_number: int) -> float:
    try:
        return _finite_float(token, description)
    except ValueError as error:
        raise MaterialFileError(f"{source}:{line_number}: {error}") from error


def _parse_property(token: str, source: Path, line_number: int) -> str:
    try:
        return _property_name(token)
    except ValueError as error:
        raise MaterialFileError(f"{source}:{line_number}: {error}") from error


def _arity_error(
    source: Path, line_number: int, command: str, expected: int, actual: int
) -> MaterialFileError:
    return MaterialFileError(
        f"{source}:{line_number}: {command} expects {expected} argument(s), "
        f"got {actual}."
    )


def parse_material_file(
    path: str | Path, defaults: Mapping[str, Any]
) -> MaterialSpec:
    """Parse a strict LAMMPS-like spatial material definition.

    Missing ``profile`` commands inherit the corresponding constant from
    ``defaults``. Commands and property/profile/shape names are
    case-insensitive; region references are matched case-insensitively.
    """

    source = Path(path).expanduser().resolve()
    try:
        raw = source.read_bytes()
    except OSError as error:
        detail = error.strerror or str(error)
        raise MaterialFileError(
            f"Cannot read material file '{source}': {detail}."
        ) from error
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeError as error:
        raise MaterialFileError(
            f"Cannot decode material file '{source}' as UTF-8: {error}."
        ) from error

    base = uniform_material_spec(defaults)
    profiles = dict(base.profiles)
    declared_profiles: set[str] = set()
    region_records: list[dict[str, Any]] = []
    region_names: dict[str, dict[str, Any]] = {}
    priorities: set[int] = set()
    pending_overrides: list[tuple[str, str, float, int]] = []
    declared_overrides: set[tuple[str, str]] = set()

    for line_number, line in enumerate(text.splitlines(), start=1):
        tokens = _tokens(line, source, line_number)
        if not tokens:
            continue
        command = tokens[0].casefold()

        if command == "profile":
            if len(tokens) < 3:
                raise _arity_error(
                    source, line_number, "profile", 3, len(tokens) - 1
                )
            property_name = _parse_property(tokens[1], source, line_number)
            kind = tokens[2].casefold()
            if kind not in _PROFILE_ARITIES:
                accepted = ", ".join(_PROFILE_ARITIES)
                raise MaterialFileError(
                    f"{source}:{line_number}: unknown profile kind '{tokens[2]}'; "
                    f"expected one of: {accepted}."
                )
            expected_values = _PROFILE_ARITIES[kind]
            expected_arguments = 2 + expected_values
            if len(tokens) - 1 != expected_arguments:
                raise _arity_error(
                    source,
                    line_number,
                    f"profile {property_name} {kind}",
                    expected_arguments,
                    len(tokens) - 1,
                )
            if property_name in declared_profiles:
                raise MaterialFileError(
                    f"{source}:{line_number}: duplicate profile for "
                    f"'{property_name}'."
                )
            values = tuple(
                _parse_float(
                    token,
                    f"Profile '{property_name}' value",
                    source,
                    line_number,
                )
                for token in tokens[3:]
            )
            try:
                profiles[property_name] = Profile(kind, values)
            except ValueError as error:
                raise MaterialFileError(f"{source}:{line_number}: {error}") from error
            declared_profiles.add(property_name)
            continue

        if command == "region":
            if len(tokens) < 3:
                raise _arity_error(
                    source, line_number, "region", 3, len(tokens) - 1
                )
            name = tokens[1].strip()
            if not name:
                raise MaterialFileError(
                    f"{source}:{line_number}: region name must be nonempty."
                )
            normalized_name = name.casefold()
            if normalized_name in region_names:
                raise MaterialFileError(
                    f"{source}:{line_number}: duplicate region name '{name}'."
                )
            shape = tokens[2].casefold()
            if shape not in _REGION_ARITIES:
                accepted = ", ".join(_REGION_ARITIES)
                raise MaterialFileError(
                    f"{source}:{line_number}: unknown region shape '{tokens[2]}'; "
                    f"expected one of: {accepted}."
                )
            expected_parameters = _REGION_ARITIES[shape]
            expected_arguments = 2 + expected_parameters + 1
            if len(tokens) - 1 != expected_arguments:
                raise _arity_error(
                    source,
                    line_number,
                    f"region {name} {shape}",
                    expected_arguments,
                    len(tokens) - 1,
                )
            parameters = tuple(
                _parse_float(
                    token,
                    f"Region '{name}' parameter",
                    source,
                    line_number,
                )
                for token in tokens[3 : 3 + expected_parameters]
            )
            priority_token = tokens[-1]
            try:
                priority = int(priority_token)
            except (TypeError, ValueError, OverflowError) as error:
                raise MaterialFileError(
                    f"{source}:{line_number}: region priority must be an integer, "
                    f"got '{priority_token}'."
                ) from error
            if priority in priorities:
                raise MaterialFileError(
                    f"{source}:{line_number}: duplicate region priority {priority}."
                )
            record: dict[str, Any] = {
                "name": name,
                "shape": shape,
                "parameters": parameters,
                "priority": priority,
                "overrides": {},
                "line": line_number,
            }
            try:
                # Validate geometry immediately, while preserving the source line.
                Region(name, shape, parameters, priority, {})
            except ValueError as error:
                raise MaterialFileError(f"{source}:{line_number}: {error}") from error
            region_records.append(record)
            region_names[normalized_name] = record
            priorities.add(priority)
            continue

        if command == "override":
            if len(tokens) - 1 != 3:
                raise _arity_error(
                    source, line_number, "override", 3, len(tokens) - 1
                )
            region_reference = tokens[1].strip()
            if not region_reference:
                raise MaterialFileError(
                    f"{source}:{line_number}: override region name must be nonempty."
                )
            property_name = _parse_property(tokens[2], source, line_number)
            key = (region_reference.casefold(), property_name)
            if key in declared_overrides:
                raise MaterialFileError(
                    f"{source}:{line_number}: duplicate override for region "
                    f"'{region_reference}' property '{property_name}'."
                )
            value = _parse_float(
                tokens[3],
                f"Region '{region_reference}' override '{property_name}'",
                source,
                line_number,
            )
            try:
                _validate_property_values(property_name, np.asarray([value]))
            except ValueError as error:
                raise MaterialFileError(
                    f"{source}:{line_number}: {error}"
                ) from error
            pending_overrides.append(
                (region_reference, property_name, value, line_number)
            )
            declared_overrides.add(key)
            continue

        raise MaterialFileError(
            f"{source}:{line_number}: unknown material command '{tokens[0]}'."
        )

    for region_reference, property_name, value, line_number in pending_overrides:
        record = region_names.get(region_reference.casefold())
        if record is None:
            raise MaterialFileError(
                f"{source}:{line_number}: override references unknown region "
                f"'{region_reference}'."
            )
        record["overrides"][property_name] = value

    regions: list[Region] = []
    for record in region_records:
        try:
            regions.append(
                Region(
                    record["name"],
                    record["shape"],
                    record["parameters"],
                    record["priority"],
                    record["overrides"],
                )
            )
        except ValueError as error:
            raise MaterialFileError(
                f"{source}:{record['line']}: {error}"
            ) from error

    try:
        return MaterialSpec(
            profiles=profiles,
            regions=tuple(regions),
            source_path=str(source),
            source_sha256=hashlib.sha256(raw).hexdigest(),
        )
    except ValueError as error:
        raise MaterialFileError(f"{source}: {error}") from error
