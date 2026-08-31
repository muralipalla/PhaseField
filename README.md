# Corrected AT2 phase-field crack simulation

This project implements the phase-field fracture outline in `KarmaCrack.pdf` as a
reproducible single-edge-notched tension benchmark. It uses FEniCSx/DOLFINx for
finite-element forms and assembly, with serial SciPy sparse solves on native
Windows. It supports proportional Mode-I/II displacement loading and uniform,
linearly graded, or region/file-defined material coefficients. The formal
algorithm acceptance and corrections review is available as
[LaTeX source](docs/PHASEFIELD_ALGORITHM_ACCEPTANCE_AND_CORRECTIONS.tex) and a
[compiled PDF](output/pdf/PHASEFIELD_ALGORITHM_ACCEPTANCE_AND_CORRECTIONS.pdf). The
[installation home page](index.html) provides the checked Windows and Linux
setup, verification, and first-run commands. The full
[HTML model guide](docs/index.html) covers the governing equations, input system,
execution, and results, while the dedicated
[installation reference](docs/installation.html) adds updates and troubleshooting.

For institute clusters, the separate
[Linux PETSc/MPI package](linux_cluster/README.md) provides a distributed
DOLFINx backend, conda environment, preflight, inputs, and Slurm/PBS launchers
for 2, 4, 16, `auto`, or any positive user-selected rank count, subject to the
detected physical-core and scheduler capacity. The Windows/SciPy backend remains
serial; both backends use the same governing forms and material parser.

## Get the code

Clone the repository and enter the project directory:

```bash
git clone https://github.com/muralipalla/PhaseField.git
cd PhaseField
```

If Git is unavailable, [download the main-branch ZIP archive](https://github.com/muralipalla/PhaseField/archive/refs/heads/main.zip),
extract it, and open a terminal in the extracted `PhaseField-main` directory.

## Windows environment

The native-Windows backend uses Python 3.12, DOLFINx 0.11.0, and serial SciPy
solves. Install Miniconda and Visual Studio 2022 Build Tools with the C++
workload and a Windows SDK. In PowerShell from the project root:

Environment file: [`environment.yml`](environment.yml)

```powershell
conda env create --file environment.yml
conda activate phasefield-fenicsx
python scripts\verify_fenicsx_install.py
```

If the named environment already exists, synchronize it instead:

```powershell
conda env update --name phasefield-fenicsx --file environment.yml --prune
```

Always activate the environment before invoking Python on Windows; activation
sets the native runtime and DLL search paths. If activation is not initialized,
run `conda init powershell`, reopen PowerShell, or source your Miniconda
`shell\condabin\conda-hook.ps1` explicitly.

For Linux PETSc/MPI installation, use
[`linux_cluster/environment-linux.yml`](linux_cluster/environment-linux.yml)
and follow the [Windows and Linux installation guide](docs/installation.html).

The Linux launcher discovers physical cores and MPI executables without assuming
`/usr/bin`. To inspect its automatic choices with a local MPICH prefix:

```bash
MPI_HOME=/home/palla/Software/mpich-local-install \
  bash linux_cluster/run_linux.sh --ranks auto --mpi auto --show-options
```

To test the exact local MPICH 4.2.3 executable with four ranks:

```bash
PREFLIGHT_ONLY=1 EXPECTED_NODES=1 \
  bash linux_cluster/run_linux.sh --ranks 4 \
  --mpi /home/palla/Software/mpich-local-install/bin/mpiexec
```

A compatible current MPI is retained. If its launcher cannot start the active
mpi4py/PETSc/DOLFINx stack correctly, the default Conda workflow bypasses it and
uses Conda MPI automatically. See the cluster guide for `RUNTIME_ENV=active`
when the institute provides the complete Python/MPI/PETSc stack.

## Run

The quick run is a resolved installation/smoke case:

```powershell
python phasefield_crack.py --quick --output-dir results/quick
```

The default benchmark uses a 100 by 100 rectangle grid (20,000 triangles), 60
nominal load increments, `ell=0.03 mm`, and plane strain:

```powershell
python phasefield_crack.py --output-dir results/full
```

The same configuration can be supplied as a LAMMPS-like keyword/value file. The
checked example lists every supported setting:

```powershell
python phasefield_crack.py -in inputs/notched_tension.in
# Equivalent long option:
python phasefield_crack.py --input-file inputs/notched_tension.in
```

Files use one `SimulationConfig` keyword and one value per line, `#` comments,
quoted paths, and explicit booleans. Configuration precedence is built-in defaults
(or `--quick`), then input-file values, then explicit CLI overrides. See the
complete [input-file reference](docs/INPUT_FILE.md) for every keyword, type, unit,
constraint, and error rule.

Ready-to-run variants are included:

```powershell
# Relative-clamped mixed Mode I/II loading
python phasefield_crack.py -in inputs/mixed_mode.in

# Built-in left-to-right material gradient
python phasefield_crack.py -in inputs/graded_linear_x.in

# Profiles plus a prioritized circular inclusion in a separate material file
python phasefield_crack.py -in inputs/graded_file.in
```

`mechanical_bc_scheme` selects `legacy_roller_pin`, `relative_clamped`, or
`symmetric_clamped`; `max_displacement` and `max_sliding_displacement` are the
relative opening and signed sliding at full load. `material_mode` selects
`uniform`, `linear_x`, `linear_y`, or `file`. Material files use strict
`profile`, `region`, and `override` commands and are resolved relative to the
main input file.

Useful CLI overrides include:

```powershell
python phasefield_crack.py --nx 100 --ny 100 --length-scale 0.03 `
  --steps 60 --max-displacement 0.015 --plane-stress `
  --output-dir results/plane_stress
```

Use `--no-xdmf`, `--no-plots`, or `--quiet` to suppress those outputs. The mesh
validator normally requires the true triangular cell diameter
`sqrt(dx^2+dy^2) <= ell/2`, an even `ny`, and `notch_length/dx` to be an integer.
`--allow-underresolved-mesh` is intended only for exploratory runs.

`load_steps` is the nominal schedule. A failed load trial is rolled back and
retried with a smaller increment, so a nonlinear run can contain more accepted
steps than requested. The final pseudo-time is still 1. Solver tolerances, cutback
controls, material values, and geometry are available through `SimulationConfig`
in `phasefield_crack.py`.

## Outputs

Each output directory contains:

- `run_manifest.json`: input configuration, derived mesh/DOF data, and software versions.
- `summary.json`: `running`, `completed`, or `failed` status, accepted-step and cutback counts, peak response, last accepted state, and any terminal error.
- `load_response.csv`: accepted states only, including reaction, energies, damage/crack measures, KKT and coupled convergence data, and load cutbacks.
- `displacement.xdmf/.h5` and `phase_field.xdmf/.h5`: accepted field states, unless disabled.
- `material_fields.xdmf/.h5`: static DG0 fields for `E`, `nu`, `Gc`, and `ell`, when `write_material_fields` is enabled.
- `load_displacement.png` and `final_phase_field.png`: plots, unless disabled.

The CSV preserves `displacement` and `reaction_force` as opening/top-normal
aliases and also reports opening, sliding, both top/bottom reaction components,
force-balance residuals, connected crack-tip coordinates, graph path length,
and kink angle. The manifest records the normalized material definition, file
SHA-256, coefficient ranges, and region cell counts.

Rejected trials are not written as field frames or CSV records. On terminal load
failure, the code restores the last accepted fields and leaves an auditable partial
CSV and failure summary.

## Tests

```powershell
python -m pytest -q
```

The tests cover input/material parsing, mixed-mode load constants, spatial DG0
coefficients, connected-crack geometry, constitutive identities, mesh/notch
validation, an end-to-end XDMF run, rollback, and failure reporting.

## Units and conventions

- Length and displacement: mm.
- Young's modulus: kN/mm².
- Fracture toughness `Gc`: kN/mm.
- Reaction force: kN for unit out-of-plane thickness.
- Internal energy: kN·mm.
- `phi=1` is intact and `phi=0` is fully broken.
- Plane strain is the default; `--plane-stress` changes the effective 2D Lamé parameter.

## Limitations

- The history-field formulation is an engineering approximation to the exact
  crack-irreversibility variational inequality, although each phase subproblem
  enforces nodal `0 <= phi <= phi_accepted` bounds with a KKT check.
- The starter notch is an internal zero-phase constraint, not a geometric slit.
- The native-Windows SciPy backend is serial and must be run with one MPI rank.
- The separate Linux backend supports distributed PETSc/MPI execution; validate
  its MPI/HDF5 stack with `linux_cluster/check_environment.py` on the target
  allocation before a production run. A ready-made
  [16-core Xeon validation and scaling suite](linux_cluster/XEON16_TEST_PLAN.md)
  exercises the MPI backend at 1, 2, 4, 8, and 16 ranks.
- Reported crack-front position follows the notch-connected nodal set with
  `phi <= 0.2`; the tip/path diagnostics are mesh-graph quantities rather than
  exact continuum crack measures.
- A prescribed displacement ratio is not a calibrated `KII/KI` ratio. Mixed-mode
  predictions remain sensitive to the spectral split, crack-face contact model,
  mesh orientation, and refinement.
- Material coefficients are cellwise DG0 reference-coordinate fields. Sharp
  interfaces should align with cells, and a single bulk phase field represents
  perfect bonding rather than interfacial delamination.
- `instantaneous_internal_energy` is an elastic-plus-fracture snapshot, not the
  history-driven phase KKT potential or a complete energy-balance residual.
- Absolute KKT and mechanical residual tolerances are calibrated for this kN-mm
  demonstration and should be reconsidered after large unit, mesh, or material
  changes.
- Prefer a fresh output directory for each parameter study. Reusing a directory
  with `--no-xdmf` or `--no-plots` does not remove older optional artifacts; use
  `summary.json` and `run_manifest.json` as the authoritative record for the run.
- The included benchmark is a numerical demonstration, not experimental material
  calibration or validation.
