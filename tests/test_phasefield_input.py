from dataclasses import fields
from pathlib import Path

import pytest

from phasefield_crack import (
    SimulationConfig,
    _config_from_args,
    _parser,
    main,
    quick_config,
)
from phasefield_input import (
    InputFileError,
    parse_input_file,
    supported_keywords,
)


def test_parser_supports_comments_quotes_booleans_and_scalar_types(tmp_path):
    source = tmp_path / "syntax.in"
    source.write_text(
        r'''# full-line comment

WIDTH 2.0                # keywords are case-insensitive
nx 40
plane_stress YES
write_xdmf off
max_displacement 1.25e-3
output_directory "D:\Simulation Results\case #1"
''',
        encoding="utf-8",
    )

    base = SimulationConfig()
    config = parse_input_file(source, base)

    assert config.width == pytest.approx(2.0)
    assert config.nx == 40
    assert config.plane_stress is True
    assert config.write_xdmf is False
    assert config.max_displacement == pytest.approx(1.25e-3)
    assert config.output_directory == r"D:\Simulation Results\case #1"
    assert config.height == base.height


def test_supported_keywords_exactly_match_simulation_config_fields():
    config = SimulationConfig()
    assert set(supported_keywords(config)) == {
        field.name for field in fields(config)
    }


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("lenght_scale 0.1\n", r"bad\.in:1: unknown keyword.*length_scale"),
        ("nx 10\nNX 20\n", r"bad\.in:2: duplicate keyword 'nx'.*line 1"),
        ("plane_stress maybe\n", r"bad\.in:1: keyword 'plane_stress' expects a boolean"),
        ("nx 10.5\n", r"bad\.in:1: keyword 'nx' expects an integer"),
        ("length_scale nan\n", r"bad\.in:1: keyword 'length_scale' requires a finite value"),
        ("width 1.0 2.0\n", r"bad\.in:1: keyword 'width' expects exactly 1 argument, got 2"),
        ("height\n", r"bad\.in:1: keyword 'height' expects exactly 1 argument, got 0"),
    ],
)
def test_parser_reports_line_numbered_errors(tmp_path, contents, message):
    source = tmp_path / "bad.in"
    source.write_text(contents, encoding="utf-8")

    with pytest.raises(InputFileError, match=message):
        parse_input_file(source, SimulationConfig())


def test_parser_rejects_empty_and_missing_files(tmp_path):
    empty = tmp_path / "empty.in"
    empty.write_text("# comments only\n\n", encoding="utf-8")

    with pytest.raises(InputFileError, match="contains no configuration commands"):
        parse_input_file(empty, SimulationConfig())
    with pytest.raises(InputFileError, match="Cannot read input file"):
        parse_input_file(tmp_path / "missing.in", SimulationConfig())


def test_input_file_precedence_is_base_then_file_then_cli(tmp_path):
    source = tmp_path / "precedence.in"
    source.write_text(
        "\n".join(
            [
                "nx 10",
                "ny 10",
                "length_scale 0.30",
                "load_steps 7",
                'output_directory "results/from file"',
                "max_staggered_iterations 77",
                "write_xdmf true",
                "verbose true",
            ]
        ),
        encoding="utf-8",
    )

    args = _parser().parse_args(
        [
            "--quick",
            "-in",
            str(source),
            "--nx",
            "20",
            "--steps",
            "2",
            "--max-staggered",
            "31",
            "--output-dir",
            "results/from_cli",
            "--no-xdmf",
            "--quiet",
        ]
    )
    config = _config_from_args(args)

    assert config.nx == 20
    assert config.ny == 10
    assert config.length_scale == pytest.approx(0.30)
    assert config.load_steps == 2
    assert config.max_staggered_iterations == 31
    assert config.output_directory == "results/from_cli"
    assert config.write_xdmf is False
    assert config.verbose is False
    assert config.max_displacement == quick_config().max_displacement


def test_final_validation_runs_after_cli_overrides(tmp_path):
    source = tmp_path / "validation.in"
    source.write_text("ny 9\nlength_scale 0.30\n", encoding="utf-8")

    invalid_args = _parser().parse_args(["--input-file", str(source)])
    with pytest.raises(
        InputFileError,
        match=r"after applying input file.*ny must be even",
    ):
        _config_from_args(invalid_args)

    repaired_args = _parser().parse_args(
        ["--input-file", str(source), "--ny", "10"]
    )
    repaired = _config_from_args(repaired_args)
    assert repaired.ny == 10
    repaired.validate()


def test_material_file_path_is_resolved_relative_to_main_input(tmp_path):
    material_directory = tmp_path / "materials"
    material_directory.mkdir()
    material_path = material_directory / "graded.material"
    material_path.write_text(
        "profile young_modulus constant 210.0\n", encoding="utf-8"
    )
    source = tmp_path / "case.in"
    source.write_text(
        "material_mode file\nmaterial_file materials/graded.material\n",
        encoding="utf-8",
    )

    config = _config_from_args(_parser().parse_args(["-in", str(source)]))

    assert config.material_file == str(material_path.resolve())


def test_file_material_mode_requires_an_explicit_material_file():
    args = _parser().parse_args(["--material-mode", "file"])

    with pytest.raises(InputFileError, match="material_file must name a file"):
        _config_from_args(args)


def test_main_formats_input_errors_as_clean_argparse_errors(tmp_path, capsys):
    missing = tmp_path / "missing.in"

    with pytest.raises(SystemExit) as exit_info:
        main(["-in", str(missing)])

    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert "error: Cannot read input file" in captured.err
    assert str(missing.resolve()) in captured.err


def test_checked_example_lists_every_keyword_and_validates():
    source = Path(__file__).resolve().parents[1] / "inputs" / "notched_tension.in"
    configured_keywords = {
        line.split(maxsplit=1)[0].casefold()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert configured_keywords == set(supported_keywords(SimulationConfig()))

    config = parse_input_file(source, SimulationConfig())
    config.validate()
    assert config.output_directory == "results/input example"


@pytest.mark.parametrize(
    "name",
    ("mixed_mode.in", "graded_linear_x.in", "graded_file.in"),
)
def test_specialized_examples_parse_resolve_and_validate(name):
    source = Path(__file__).resolve().parents[1] / "inputs" / name

    config = _config_from_args(_parser().parse_args(["-in", str(source)]))

    config.validate()
    if config.material_mode == "file":
        assert Path(config.material_file).is_file()
