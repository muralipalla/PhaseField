# Phase-field text input reference

The solver accepts a LAMMPS-like UTF-8 text file through either `-in` or
`--input-file`:

```powershell
python phasefield_crack.py -in inputs/notched_tension.in
python phasefield_crack.py --input-file inputs/notched_tension.in
```

Invoking the command runs the simulation; there is no separate `run` keyword.

## Syntax

Each nonempty line contains exactly one keyword and one value:

```text
keyword value
```

For example:

```text
# Geometry in mm
width 1.0
height 1.0

plane_stress false                 # false means plane strain
fracture_toughness 2.7e-3
output_directory "results/case 01" # quote strings containing spaces
```

The syntax rules are:

- Keywords are the `SimulationConfig` field names listed below. They are
  case-insensitive.
- A keyword may occur only once. Duplicate settings are rejected rather than
  silently overwritten.
- `#` begins a comment outside quoted text. Blank and comment-only lines are
  ignored.
- Single or double quotes preserve whitespace and `#` inside a string value.
- Integers use integer syntax such as `100`; `100.0` is not accepted as an
  integer. Floating-point values may use scientific notation.
- Floating-point values must be finite; `nan`, `inf`, and `-inf` are rejected.
- Boolean values are explicit: `true/false`, `yes/no`, `on/off`, or `1/0`, in any
  letter case.
- Environment-variable expansion, expressions, continuation lines, includes,
  unit conversion, and unknown commands are not supported.

The main input path and a relative `output_directory` are resolved from the
process working directory. A relative `material_file` read from the main input
is different by design: it is resolved from the directory containing that input
file. This makes a case and its material description portable when launched from
another directory. An explicit relative `--material-file` CLI override is
resolved from the process working directory.
Quote Windows paths that contain spaces, for example:

```text
output_directory "D:\Simulation Results\case 01"
```

The separate material file uses its own command grammar, documented under
[File-defined materials](#file-defined-materials). It is not parsed as a main
keyword/value file.

## Precedence

Configuration is assembled in this order:

1. `SimulationConfig()` supplies the normal built-in defaults. If `--quick` is
   present, `quick_config()` supplies the base instead.
2. Keywords in the input file replace base values.
3. Explicit command-line override flags replace input-file values.
4. `SimulationConfig.validate()` checks the final combined configuration.

This permits a CLI override to repair or temporarily vary a file setting:

```powershell
python phasefield_crack.py -in inputs/notched_tension.in `
  --nx 40 --ny 40 --length-scale 0.075 `
  --output-dir results/refined
```

The explicit CLI mappings are:

| CLI option | Configuration field/effect |
|---|---|
| `--output-dir PATH` | `output_directory` |
| `--nx N` | `nx` |
| `--ny N` | `ny` |
| `--length-scale VALUE` | `length_scale` |
| `--steps N` | `load_steps` |
| `--max-displacement VALUE` | `max_displacement` |
| `--max-sliding-displacement VALUE` | `max_sliding_displacement` |
| `--mechanical-bc-scheme NAME` | `mechanical_bc_scheme` |
| `--material-mode NAME` | `material_mode` |
| `--material-file PATH` | `material_file`; a relative CLI path is resolved from the process working directory |
| `--max-staggered N` | `max_staggered_iterations` |
| `--plane-stress` | Sets `plane_stress=true` |
| `--no-xdmf` | Sets `write_xdmf=false` |
| `--no-plots` | Sets `make_plots=false` |
| `--quiet` | Sets `verbose=false` |
| `--allow-underresolved-mesh` | Sets `strict_mesh_resolution=false` |

Unspecified CLI options do not replace file values. In particular, omitting
`--plane-stress` does not change `plane_stress` from the file.

## Keyword reference

The fixed unit system is kN-mm with unit out-of-plane thickness. Defaults below
are the normal `SimulationConfig()` defaults, before `--quick` or file overrides.

### Geometry and mesh

| Keyword | Type | Default | Unit | Constraint/meaning |
|---|---:|---:|---|---|
| `width` | float | `1.0` | mm | Specimen width; greater than 0. |
| `height` | float | `1.0` | mm | Specimen height; greater than 0. |
| `notch_length` | float | `0.20` | mm | Strictly between 0 and `width`; must align with an x-direction mesh vertex. |
| `nx` | integer | `100` | divisions | At least 2; `notch_length/(width/nx)` must be an integer. |
| `ny` | integer | `100` | divisions | At least 2 and even, so the notch lies on the mid-height vertex row. |

The triangular cell diameter used by validation is
`h=sqrt((width/nx)^2+(height/ny)^2)`. When `strict_mesh_resolution=true`, the
solver requires `h/min(ell(x)) <= 0.5`; in uniform mode this is simply
`h/length_scale <= 0.5`.

### Material and phase-field model

| Keyword | Type | Default | Unit | Constraint/meaning |
|---|---:|---:|---|---|
| `young_modulus` | float | `210.0` | kN/mm² | Greater than 0. |
| `poisson_ratio` | float | `0.30` | dimensionless | Strictly between `-0.99` and `0.5`. |
| `fracture_toughness` | float | `2.7e-3` | kN/mm | AT2 `Gc`; greater than 0. |
| `length_scale` | float | `0.030` | mm | AT2 regularization length `ell`; greater than 0. |
| `young_modulus_end` | float | `210.0` | kN/mm² | Value of `E` at `x=width` or `y=height` for a linear material mode; greater than 0. |
| `poisson_ratio_end` | float | `0.30` | dimensionless | End value of `nu`; strictly between `-0.99` and `0.5`. |
| `fracture_toughness_end` | float | `2.7e-3` | kN/mm | End value of `Gc`; greater than 0. |
| `length_scale_end` | float | `0.030` | mm | End value of `ell`; greater than 0 and included in mesh-resolution validation. |
| `material_mode` | string | `uniform` | — | `uniform`, `linear_x`, `linear_y`, or `file`. |
| `material_file` | string | `none` | path | Required in `file` mode; a relative path is anchored at the main input file. |
| `residual_stiffness` | float | `1.0e-8` | dimensionless | Normalized degradation floor; strictly between 0 and 1. |
| `plane_stress` | boolean | `false` | — | `false` selects plane strain; `true` selects plane stress. |

For `material_mode=uniform`, the ordinary values are constant and the four
`*_end` values are ignored. For `linear_x`, every supported property `p` is

```text
p(x,y) = p_start + (p_end - p_start) x / width.
```

For `linear_y`, replace `x/width` by `y/height`. Thus the existing property
keywords are the left/bottom values, and the `*_end` keywords are the right/top
values. The Lamé fields are evaluated locally from `E(x)` and `nu(x)` using the
selected plane-stress or plane-strain relation. `Gc(x)` and `ell(x)` also enter
the phase bilinear and linear forms locally.

### Loading

| Keyword | Type | Default | Unit | Constraint/meaning |
|---|---:|---:|---|---|
| `mechanical_bc_scheme` | string | `legacy_roller_pin` | — | `legacy_roller_pin`, `relative_clamped`, or `symmetric_clamped`. |
| `max_displacement` | float | `0.015` | mm | Final relative opening displacement; at least 0. |
| `max_sliding_displacement` | float | `0.0` | mm | Final relative tangential displacement; signed. Must be zero for `legacy_roller_pin`. |
| `load_steps` | integer | `60` | nominal increments | At least 1. Adaptive cutback can create more accepted increments. |

At pseudo-time `t`, define the requested relative displacement
`Delta(t)=t(max_sliding_displacement,max_displacement)`. The schemes impose:

| Scheme | Bottom edge | Top edge | Intended use |
|---|---|---|---|
| `legacy_roller_pin` | `uy=0`, plus `ux=0` at the lower-left vertex | `uy=t*max_displacement`; `ux` free | Backward-compatible mode-I roller-and-pin benchmark; sliding is rejected. |
| `relative_clamped` | `ux=uy=0` | `ux=t*max_sliding_displacement`, `uy=t*max_displacement` | Mixed-mode loading with the full relative displacement on the top. |
| `symmetric_clamped` | `u=-Delta(t)/2` | `u=+Delta(t)/2` | Mixed-mode loading centered about zero, with the same top-to-bottom relative displacement. |

The opening/sliding ratio is a kinematic load ratio; it is **not** generally the
fracture-mechanics ratio `KII/KI`. Geometry, crack length, elastic mismatch, and
finite boundaries all affect the actual near-tip mode mixity.

### Newton and staggered solution

| Keyword | Type | Default | Unit | Constraint/meaning |
|---|---:|---:|---|---|
| `max_newton_iterations` | integer | `25` | iterations | At least 1 per displacement solve. |
| `newton_relative_tolerance` | float | `1.0e-8` | dimensionless | Greater than 0. |
| `newton_absolute_tolerance` | float | `1.0e-10` | kN | Greater than 0; absolute assembled mechanical residual tolerance. |
| `newton_increment_tolerance` | float | `1.0e-10` | mm | Greater than 0; displacement increment and boundary-violation tolerance. |
| `max_staggered_iterations` | integer | `250` | iterations | At least 1 per load attempt. |
| `staggered_tolerance` | float | `1.0e-5` | dimensionless | Greater than 0; phase increment infinity-norm tolerance. |
| `staggered_mechanical_tolerance` | float | `1.0e-6` | kN | Greater than 0; coupled free-DOF mechanical residual tolerance after the phase update. |
| `phase_kkt_tolerance` | float | `1.0e-8` | kN·mm | Greater than 0; projected-gradient residual tolerance for the constrained phase solve. |
| `phase_optimizer_max_iterations` | integer | `200` | iterations | At least 1 for L-BFGS-B phase optimization. |
| `spectral_smoothing` | float | `1.0e-10` | strain | Greater than 0; regularizes spectral eigenvalue/Macaulay derivatives near zero. |

The absolute residual tolerances are mesh- and unit-dependent. Reassess them after
large changes to units, mesh density, geometry, or material stiffness.

### File-defined materials

Set `material_mode file` and name a material description in `material_file` to
combine analytic baseline profiles with box or circular overrides:

```text
material_mode file
material_file materials/inclusion.material
write_material_fields true
```

Unlike the main input, a material file contains repeatable commands with several
arguments. Supported commands are:

```text
profile PROPERTY constant VALUE
profile PROPERTY linear_x LEFT RIGHT
profile PROPERTY linear_y BOTTOM TOP
profile PROPERTY affine ORIGIN DX DY

region NAME box XMIN XMAX YMIN YMAX PRIORITY
region NAME circle CX CY RADIUS PRIORITY
override NAME PROPERTY VALUE
```

Commands and property names are case-insensitive; blank lines and `#` comments
are allowed. At most one explicit profile may be supplied for each property;
an omitted profile inherits the corresponding ordinary main-input value as a
constant. Region names are unique case-insensitively, override pairs cannot be
duplicated, and region priorities are unique integers.

`PROPERTY` is one of `young_modulus`, `poisson_ratio`,
`fracture_toughness`, or `length_scale`. The profile equations are

```text
constant:  p(x,y) = VALUE
linear_x:  p(x,y) = LEFT + (RIGHT - LEFT) x / width
linear_y:  p(x,y) = BOTTOM + (TOP - BOTTOM) y / height
affine:    p(x,y) = ORIGIN + DX*x + DY*y
```

Profiles provide the whole-domain values. A region override replaces a named
property on cells whose midpoint lies in that region. Higher-priority regions
take precedence where regions overlap, and priorities must be unique. Names are
identifiers local to the material file, coordinates use mm,
and `DX`/`DY` carry the property unit per mm.

Complete inclusion example:

```text
profile young_modulus constant 210.0
profile poisson_ratio constant 0.30
profile fracture_toughness constant 2.7e-3
profile length_scale constant 0.150

region stiff_inclusion circle 0.65 0.50 0.12 20
override stiff_inclusion young_modulus 420.0
override stiff_inclusion poisson_ratio 0.28
```

Material profiles and regions are evaluated into cellwise DG0 coefficient
fields. Consequently, smooth analytic grading is represented piecewise
constantly, while a curved or non-mesh-fitted interface is a cell-midpoint
stair-step approximation. Refine the mesh and repeat the study whenever a
gradient or inclusion controls the crack path. With `write_material_fields=true`,
the four static fields are written once to `material_fields.xdmf/.h5` for visual
inspection. The manifest records the normalized material specification, source
path and SHA, coefficient ranges, and region cell counts.

On the PETSc backend the material description is normalized once and shared
collectively. Each rank evaluates its owned cells, global extrema and region
counts use MPI reductions, and static XDMF/HDF5 output is collective. This avoids
rank-dependent assignment at partition boundaries.

### Adaptive load control

| Keyword | Type | Default | Unit | Constraint/meaning |
|---|---:|---:|---|---|
| `max_load_cutbacks` | integer | `8` | retries | At least 0; maximum consecutive rejected attempts before failure. |
| `load_cutback_factor` | float | `0.5` | dimensionless | Strictly between 0 and 1; multiplier applied after rejection. |
| `load_growth_patience` | integer | `3` | accepted steps | At least 1; easy accepted steps required before increment recovery. |
| `load_growth_iteration_threshold` | integer | `10` | staggered iterations | At least 1; an accepted step is considered easy at or below this count. |

### Output and safeguards

| Keyword | Type | Default | Unit | Constraint/meaning |
|---|---:|---:|---|---|
| `output_directory` | string | `results/default` | path | Nonempty; quote it when it contains spaces. Existing optional artifacts are not cleaned automatically. |
| `write_every` | integer | `1` | accepted steps | At least 1; XDMF write cadence. Initial and final accepted states are still written. |
| `write_xdmf` | boolean | `true` | — | Write displacement and phase XDMF/HDF5 field files. |
| `write_material_fields` | boolean | `true` | — | Write static `E`, `nu`, `Gc`, and `ell` DG0 fields once to `material_fields.xdmf/.h5`; independent of `write_xdmf`. |
| `make_plots` | boolean | `true` | — | Write load-response and final phase PNG plots. |
| `strict_mesh_resolution` | boolean | `true` | — | Enforce `h/ell <= 0.5`; disable only for exploratory work. |
| `verbose` | boolean | `true` | — | Print accepted-step and cutback progress. |

## Errors and validation

The parser fails before mesh construction and reports the resolved file path and
line number for line-local mistakes. Examples include:

```text
.../case.in:12: unknown keyword 'lenght_scale'. Did you mean 'length_scale'?
.../case.in:18: duplicate keyword 'nx'; first set on line 4.
.../case.in:21: keyword 'plane_stress' expects a boolean (...), got 'maybe'.
.../case.in:25: keyword 'width' expects exactly 1 argument, got 2. Quote string values that contain spaces.
```

Missing, unreadable, non-UTF-8, and command-free main input files also fail
clearly. After
file and CLI precedence is resolved, cross-field errors are prefixed with
`Configuration after applying input file ... and command-line overrides is
invalid`, followed by the `SimulationConfig.validate()` explanation. These main
configuration failures occur before finite-element construction.

During material-field construction, spatial validation additionally rejects a missing material file,
unsupported material command/property, malformed or duplicate profile, invalid
region geometry, empty region, nonfinite coefficient, nonpositive `E`, `Gc`, or
`ell`, inadmissible `nu`, and an under-resolved local length scale. Scheme
validation rejects unknown boundary schemes and any nonzero sliding displacement
with `legacy_roller_pin`.

## Reactions and mixed-mode diagnostics

Every accepted row reports the actual relative kinematics as
`opening_displacement` and `sliding_displacement`. It also reports
`top_reaction_x`, `top_reaction_y`, `bottom_reaction_x`,
`bottom_reaction_y`, and the resultant checks `force_balance_x` and
`force_balance_y`. The historical columns remain compatible:
`displacement=opening_displacement` and `reaction_force=top_reaction_y`.
The PNG load curve shows top-y reaction versus relative opening and adds a
top-x-reaction versus relative-sliding panel when sliding is nonzero. The CSV
remains the authoritative source for custom sign conventions and comparisons.

The four grip reactions are computed from the assembled **unconstrained**
mechanical residual: owned residual entries for each grip/component are summed,
then MPI-reduced on the PETSc backend. This is the discrete reaction associated
with the imposed Dirichlet values and avoids introducing a separate boundary
traction quadrature. `force_balance_x/y` are the corresponding top-plus-bottom
sums and should be near zero at mechanical equilibrium.

For a kinked crack, `crack_front_x` alone is not a path measure. The extended
diagnostics therefore include `crack_tip_x`, `crack_tip_y`,
`crack_path_length`, `crack_kink_angle_degrees`, and
`connected_crack_node_count`, while retaining `crack_front_x` and
`crack_extension` for older post-processing.

These are mesh/threshold diagnostics, not stress-intensity factors. Mixed-mode
phase-field results can be sensitive to the tension/compression split, crack-face
contact treatment, diffuse-notch representation, mesh orientation, and the
regularization length. In particular, this model does not add a separate
frictional contact law for closed crack faces.

## Complete example

[`inputs/notched_tension.in`](../inputs/notched_tension.in) lists every supported
keyword. It is a short, mesh-resolved demonstration with an enlarged length scale:

```powershell
python phasefield_crack.py -in inputs/notched_tension.in
```

Additional runnable cases are:

```powershell
python phasefield_crack.py -in inputs/mixed_mode.in
python phasefield_crack.py -in inputs/graded_linear_x.in
python phasefield_crack.py -in inputs/graded_file.in
```

- [`inputs/mixed_mode.in`](../inputs/mixed_mode.in) combines opening and sliding
  under `relative_clamped` loading.
- [`inputs/graded_linear_x.in`](../inputs/graded_linear_x.in) uses the built-in
  end-value interpolation.
- [`inputs/graded_file.in`](../inputs/graded_file.in) references
  [`inputs/materials/inclusion.material`](../inputs/materials/inclusion.material).

To avoid reusing its output directory while experimenting, override it on the
command line:

```powershell
python phasefield_crack.py -in inputs/notched_tension.in `
  --output-dir results/input_trial_02 --no-xdmf
```
