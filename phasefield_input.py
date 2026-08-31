"""LAMMPS-style text input parsing for phase-field simulation dataclasses.

Each nonempty line contains one dataclass field name followed by one value.
Parsing is intentionally strict so input mistakes fail before an expensive solve.
"""

from __future__ import annotations

import difflib
import math
import shlex
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar, get_type_hints


ConfigT = TypeVar("ConfigT")


class InputFileError(ValueError):
    """A user-facing syntax, type, or input-file access error."""


_TRUE_VALUES = frozenset({"true", "yes", "on", "1"})
_FALSE_VALUES = frozenset({"false", "no", "off", "0"})
_SUPPORTED_TYPES = frozenset({bool, int, float, str})


def supported_keywords(base_config: Any) -> dict[str, type[Any]]:
    """Return accepted field keywords and their scalar Python types."""

    if isinstance(base_config, type) or not is_dataclass(base_config):
        raise TypeError("base_config must be a dataclass instance.")

    type_hints = get_type_hints(type(base_config))
    keywords: dict[str, type[Any]] = {}
    for field in fields(base_config):
        expected_type = type_hints.get(field.name)
        if expected_type not in _SUPPORTED_TYPES:
            raise TypeError(
                f"Input keyword '{field.name}' has unsupported type "
                f"{expected_type!r}; only bool, int, float, and str are supported."
            )
        keywords[field.name] = expected_type
    return keywords


def _unquote(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        return token[1:-1]
    return token


def _tokenize(line: str, source: Path, line_number: int) -> list[str]:
    lexer = shlex.shlex(line, posix=False)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    try:
        return [_unquote(token) for token in lexer]
    except ValueError as error:
        raise InputFileError(f"{source}:{line_number}: {error}.") from error


def _convert_value(
    keyword: str,
    raw_value: str,
    expected_type: type[Any],
    source: Path,
    line_number: int,
) -> bool | int | float | str:
    location = f"{source}:{line_number}"
    if expected_type is str:
        if raw_value == "":
            raise InputFileError(
                f"{location}: keyword '{keyword}' requires a nonempty string."
            )
        return raw_value

    if expected_type is bool:
        normalized = raw_value.casefold()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
        raise InputFileError(
            f"{location}: keyword '{keyword}' expects a boolean "
            "(true/false, yes/no, on/off, or 1/0), "
            f"got '{raw_value}'."
        )

    try:
        converted = expected_type(raw_value)
    except (TypeError, ValueError, OverflowError) as error:
        expectation = (
            "an integer" if expected_type is int else "a floating-point number"
        )
        raise InputFileError(
            f"{location}: keyword '{keyword}' expects {expectation}, "
            f"got '{raw_value}'."
        ) from error

    if expected_type is float and not math.isfinite(converted):
        raise InputFileError(
            f"{location}: keyword '{keyword}' requires a finite value, "
            f"got '{raw_value}'."
        )
    return converted


def parse_input_file(path: str | Path, base_config: ConfigT) -> ConfigT:
    """Apply a strict text input file to a dataclass configuration instance.

    The parser performs file, line, keyword, arity, duplicate, and scalar-type
    checks. Cross-field and physical validation remains the configuration
    class's responsibility after any command-line overrides have been applied.
    """

    source = Path(path).expanduser().resolve()
    try:
        text = source.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        detail = error.strerror if isinstance(error, OSError) else str(error)
        raise InputFileError(f"Cannot read input file '{source}': {detail}.") from error

    keywords = supported_keywords(base_config)
    updates: dict[str, bool | int | float | str] = {}
    first_occurrence: dict[str, int] = {}

    for line_number, line in enumerate(text.splitlines(), start=1):
        tokens = _tokenize(line, source, line_number)
        if not tokens:
            continue

        keyword = tokens[0].casefold()
        if keyword not in keywords:
            match = difflib.get_close_matches(keyword, keywords, n=1, cutoff=0.6)
            suggestion = f" Did you mean '{match[0]}'?" if match else ""
            raise InputFileError(
                f"{source}:{line_number}: unknown keyword '{tokens[0]}'."
                f"{suggestion}"
            )

        argument_count = len(tokens) - 1
        if argument_count != 1:
            suffix = " Quote string values that contain spaces." if argument_count > 1 else ""
            raise InputFileError(
                f"{source}:{line_number}: keyword '{keyword}' expects exactly "
                f"1 argument, got {argument_count}.{suffix}"
            )

        if keyword in first_occurrence:
            raise InputFileError(
                f"{source}:{line_number}: duplicate keyword '{keyword}'; "
                f"first set on line {first_occurrence[keyword]}."
            )

        updates[keyword] = _convert_value(
            keyword,
            tokens[1],
            keywords[keyword],
            source,
            line_number,
        )
        first_occurrence[keyword] = line_number

    if not updates:
        raise InputFileError(
            f"Input file '{source}' contains no configuration commands."
        )

    return replace(base_config, **updates)
