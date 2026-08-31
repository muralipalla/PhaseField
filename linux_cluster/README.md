# Linux PETSc/MPI cluster version

This directory is a separate Linux backend for genuinely distributed runs. It
uses DOLFINx partitioned meshes, PETSc SNES for nonlinear mechanics, and PETSc
SNESVI for the phase-field variational inequality. It is intended for commands
at 2, 4, 16, `auto`, or any positive user-selected MPI rank count; it does not
call the serial SciPy solver in `phasefield_crack.py`.

The physical model, spectral split, history update, staggered load loop,
transactional cutbacks, and accepted-state irreversibility are the same as in
the Windows implementation. The parallel backend additionally makes scalar
integrals and norms global, fixes the phase bounds on distributed owned DOFs,
uses collective XDMF/HDF5 output, and restricts JSON/CSV/PNG writes to rank 0.

## Files

- `phasefield_crack_petsc.py`: distributed simulation and command-line entry point.
- `petsc_config.py`: the shared physical fields plus PETSc-specific input fields.
- `environment-linux.yml`: reproducible conda-forge environment using MPICH.
- `check_environment.py`: collective MPI/PETSc communicator, KSP, SNESVI,
  placement, ghost exchange, and XDMF/HDF5 write/read preflight.
- `mpi_compatibility_probe.py`: small world-size and communicator check used to
  test each launcher candidate before the full preflight.
- `run_linux.sh`: allocation-aware direct launcher with automatic core and MPI
  discovery.
- [`run_xeon16_suite.sh`](run_xeon16_suite.sh): one-host functional and
  1/2/4/8/16-rank validation suite; see the
  [16-core Xeon test plan](XEON16_TEST_PLAN.md).
- `summarize_xeon16_suite.py` and `xeon16_hardware.py`: automated result
  checking, scaling aggregation, provenance, and physical-core safeguards.
- `slurm_phasefield.sbatch`: scheduler template for Slurm.
- `pbs_phasefield.pbs`: scheduler template for PBS Professional/OpenPBS.
- `inputs/cluster_smoke.in`: two-step installation test.
- `inputs/notched_tension_cluster.in`: 200 x 200 production example.
- [`inputs/mixed_mode.in`](inputs/mixed_mode.in): relative-clamped opening plus sliding example.
- [`inputs/graded_linear_x.in`](inputs/graded_linear_x.in): built-in material-gradient example.
- [`inputs/graded_file.in`](inputs/graded_file.in) and
  [`inputs/materials/inclusion.material`](inputs/materials/inclusion.material):
  file-defined circular-inclusion example.

## 1. Transfer and install

Copy the **whole project**, not only this directory, because the PETSc backend
shares the audited constitutive definitions and strict input parser at the
project root. Obtain it on your workstation with Git:

```bash
git clone https://github.com/muralipalla/PhaseField.git
```

If Git is unavailable, [download the main-branch ZIP archive](https://github.com/muralipalla/PhaseField/archive/refs/heads/main.zip)
and extract it. Then transfer the complete project to the cluster, for example:

```bash
rsync -av PhaseField/ username@login.cluster.edu:/path/to/PhaseField/
```

On the cluster login node:

```bash
cd /path/to/PhaseField
```

Environment file: [`linux_cluster/environment-linux.yml`](environment-linux.yml)

```bash
conda env create --file linux_cluster/environment-linux.yml
conda activate phasefield-fenicsx-linux
```

If the environment already exists:

```bash
conda env update --name phasefield-fenicsx-linux \
  --file linux_cluster/environment-linux.yml --prune
```

The environment is solved as one conda-forge stack: DOLFINx 0.11, real-scalar
PETSc/petsc4py 3.25, mpi4py, HDF5, and MPICH must remain ABI-compatible. Do not
install a separate pip `petsc4py` over it.

## 2. Obtain compute resources and verify MPI

Do not run the preflight or simulation on an institute login node. Use a batch
submission, or first request the site's documented interactive compute
allocation (for example, an `salloc`/interactive PBS job) and run the direct
commands inside that allocation.

First ask the launcher to report the affinity-visible physical cores, scheduler
capacity, MPI found before Conda activation, Conda MPI, and final tested choice:

```bash
bash linux_cluster/run_linux.sh --ranks auto --mpi auto --show-options
```

Then run the collective finite-element/KSP test at the rank count planned for
the job. This four-rank example is deliberately small; replace `4` and the node
count to match the allocation:

```bash
PREFLIGHT_ONLY=1 EXPECTED_NODES=1 \
  bash linux_cluster/run_linux.sh --ranks 4
```

The check fails if the MPI communicators or sizes disagree, PETSc uses complex
scalars, DOLFINx lacks PETSc support, the distributed Poisson or SNESVI solve
diverges, or collective XDMF/HDF5 write/read fails. It prints the rank count on
each host. Set `EXPECTED_NODES` to the actual allocation; this catches a
launcher that incorrectly places every rank on one host. A one-rank check is
rejected unless `--allow-one-rank` is supplied explicitly.

Then run the small phase-field case:

```bash
RANKS=4
INPUT_FILE=linux_cluster/inputs/cluster_smoke.in \
OUTPUT_DIR="results/smoke_${RANKS}r" \
bash linux_cluster/run_linux.sh --ranks "${RANKS}"
```

For a 16-physical-core Linux Xeon workstation, the prepared one-command suite
adds mixed-mode, graded-material, parallel-I/O, rank-regression, and strong-
scaling checks:

```bash
bash linux_cluster/run_xeon16_suite.sh smoke
bash linux_cluster/run_xeon16_suite.sh validation
SCALING_REPEATS=3 bash linux_cluster/run_xeon16_suite.sh scaling
```

See [`XEON16_TEST_PLAN.md`](XEON16_TEST_PLAN.md) for the exact meshes, loads,
pass criteria, output reports, and troubleshooting controls.

## 3. Variable-rank direct runs

Inside an interactive compute allocation, `run_linux.sh` activates the conda
environment, defaults threaded BLAS/OpenMP to one thread per rank unless the
caller explicitly overrides those variables, performs the preflight, then
starts the simulation. The Xeon16 benchmark suite force-sets them to one.

```bash
bash linux_cluster/run_linux.sh --ranks 2
bash linux_cluster/run_linux.sh --ranks 4
bash linux_cluster/run_linux.sh --ranks 16
bash linux_cluster/run_linux.sh --ranks auto
bash linux_cluster/run_linux.sh --ranks 7  # any positive custom count
```

`--ranks` takes precedence over legacy `NPROCS`, then scheduler variables. With
`auto`, the workstation choice is the affinity-visible physical-core count. A
one-node scheduler job also respects its task slots; a multi-node job uses the
smaller of the scheduler slot count and the per-node physical-core estimate
multiplied by allocated nodes. An explicit request above detected capacity is
reduced with a clear notice, or rejected when `STRICT_RESOURCES=1`.

The launcher does not assume MPI is in `/usr/bin`. In `--mpi auto` mode it checks
`MPI_HOME`, captures `mpiexec`/`mpirun` visible before Conda activation, and also
finds Conda MPI. Each selected launcher must create the expected MPI world and
show congruent mpi4py/PETSc communicators. A current MPI that passes is used; if
it fails the micro-probe or full PETSc/XDMF preflight, the Conda workflow retries
once with Conda MPI.

For the local MPICH 4.2.3 installation described in this project setup:

```bash
# Prefix discovery; accepts either PREFIX or PREFIX/bin through MPI_HOME.
MPI_HOME=/home/palla/Software/mpich-local-install \
  bash linux_cluster/run_linux.sh --ranks auto --mpi auto --show-options

# Prefer and test the exact launcher path; automatic fallback remains enabled.
PREFLIGHT_ONLY=1 EXPECTED_NODES=1 \
  bash linux_cluster/run_linux.sh --ranks 4 \
  --mpi /home/palla/Software/mpich-local-install/bin/mpiexec

# A full smoke solve using the same launcher.
RANKS=4
INPUT_FILE=linux_cluster/inputs/cluster_smoke.in \
OUTPUT_DIR="results/local_mpich_${RANKS}r" \
bash linux_cluster/run_linux.sh --ranks "${RANKS}" \
  --mpi /home/palla/Software/mpich-local-install/bin/mpiexec
```

No `LD_LIBRARY_PATH` edit is needed merely to try an external launcher: the
probe determines whether it can start the Conda Python stack. If the institute
provides a complete FEniCSx/PETSc/mpi4py stack built against its MPI, activate
that stack first and use `RUNTIME_ENV=active`; do not activate the Conda stack.

Set `EXPECTED_NODES` to the allocated node count when the scheduler does not
provide `SLURM_NNODES`; the supplied Slurm/PBS batch templates do this
automatically.

Useful launcher variables are:

| Variable | Default | Purpose |
|---|---:|---|
| `--ranks` / `NPROCS` | `auto` | Positive rank count or allocation-aware automatic capacity. |
| `--mpi` / `MPI_PREFERENCE` | `auto` | `auto`, `conda`, `system`, `srun`, or an executable path. |
| `MPI_HOME` | unset | MPI prefix or bin directory searched before the pre-activation PATH candidate. |
| `MPI_EXTRA_ARGS` | empty | Site-required launcher arguments. |
| `CONDA_BASE` | unset | Explicit Miniconda root; otherwise `conda` must be on `PATH`. |
| `ENV_NAME` | `phasefield-fenicsx-linux` | Conda environment name. |
| `RUNTIME_ENV` | `conda` | Set `active` for a complete, already activated site stack. |
| `PYTHON_BIN` | `python` | Python command or path within the selected runtime. |
| `INPUT_FILE` | production input | LAMMPS-style parameter file. |
| `OUTPUT_DIR` | unique timestamp | Shared output directory. |
| `SKIP_PREFLIGHT` | `0` | Set to `1` only after the same allocation was checked. |
| `PREFLIGHT_ONLY` | `0` | Set to `1` to stop after the collective preflight. |
| `PREFLIGHT_REPORT_FILE` | empty | Atomically persist the structured preflight JSON report. |
| `ALLOW_EXISTING_OUTPUT` | `0` | Set to `1` to permit a nonempty result directory. |
| `STRICT_RESOURCES` | `0` | Fail instead of reducing an oversized rank request. |
| `MPI_PROBE_TIMEOUT_SECONDS` | `30` | Timeout for the small launcher/runtime compatibility probe. |
| `MPI_PROBE_VERBOSE` | `0` | Print failed-probe diagnostics as well as successful probe metadata. |
| `FORCE_INCOMPATIBLE_MPI` | `0` | Expert-only override that forces a non-Conda launcher after a failed probe. |
| `ALLOW_EXTERNAL_MPI` | `0` | Deprecated no-op retained for old job scripts; external paths no longer need permission. |
| `EXPECTED_NODES` | unset | Required distinct host count in preflight; scheduler templates set it. |

`MPI_LAUNCHER` remains a legacy alias for `MPI_PREFERENCE`. Prefer the new name
because the latter clearly accepts policies as well as executable names. Run the
script once from the batch shell; do not wrap it in `mpiexec` or `srun`, because
it creates the MPI step itself.

## 4. Slurm

The template intentionally has no site-specific account, partition, or module
commands. Add only those required by the institute. Submit from the project
root:

```bash
# Uses the template's safe two-task default
sbatch linux_cluster/slurm_phasefield.sbatch

# Common workstation/allocation choices
sbatch --ntasks=4 linux_cluster/slurm_phasefield.sbatch
sbatch --ntasks=16 linux_cluster/slurm_phasefield.sbatch

# Any scheduler-approved custom count, here 7
INPUT_FILE="$PWD/linux_cluster/inputs/my_case.in" \
OUTPUT_DIR="$PWD/results/my_case_7r" \
sbatch --ntasks=7 --time=08:00:00 linux_cluster/slurm_phasefield.sbatch
```

If submitting from another directory, export the absolute project path:

```bash
export PHASEFIELD_PROJECT_ROOT=/path/to/PhaseField
sbatch /path/to/PhaseField/linux_cluster/slurm_phasefield.sbatch
```

On a Slurm site that does not export submission variables by default, pass
them explicitly with
`--export=ALL,CONDA_BASE=...,PHASEFIELD_PROJECT_ROOT=...,INPUT_FILE=...,OUTPUT_DIR=...`.

The template defaults `MPI_PREFERENCE` to `auto`, while preserving an inherited
legacy `MPI_LAUNCHER`. In automatic Slurm jobs, an explicit `MPI_HOME` candidate
is tried first, then scheduler-native `srun`, the pre-activation PATH MPI, and
Conda MPI. Every candidate must pass the world-size/communicator probe. To
require scheduler-native launch as the first and only non-Conda choice, submit
with `MPI_PREFERENCE=srun`.

## 5. PBS

The PBS template has a safe two-rank default. Typical submissions are:

```bash
qsub linux_cluster/pbs_phasefield.pbs

# Four or 16 ranks on one suitable node
qsub -l select=1:ncpus=4:mpiprocs=4 linux_cluster/pbs_phasefield.pbs
qsub -l select=1:ncpus=16:mpiprocs=16 linux_cluster/pbs_phasefield.pbs

# Any custom scheduler-approved shape, here 32 ranks across two nodes
qsub -l select=2:ncpus=16:mpiprocs=16 -l mem=64gb \
  linux_cluster/pbs_phasefield.pbs
```

PBS syntax and resource names vary. The script derives `NPROCS` from `PBS_NP`
or `PBS_NODEFILE`; add a queue/project directive only as documented locally.
PBS installations also vary in environment forwarding: use `qsub -V`, or
explicit
`-v CONDA_BASE=...,PHASEFIELD_PROJECT_ROOT=...,INPUT_FILE=...,OUTPUT_DIR=...`,
when required.

## 6. Input-file structure

The grammar is the same strict LAMMPS-like structure used by the serial solver:

```text
keyword value              # optional comment
output_directory "path containing spaces"
write_xdmf true
```

Each nonempty line has exactly one case-insensitive keyword and one argument.
Duplicate or unknown keywords, invalid types, non-finite numbers, and extra
arguments fail before mesh construction. Booleans accept `true/false`,
`yes/no`, `on/off`, or `1/0`. Precedence is defaults (or `--quick`), input file,
then explicit command-line overrides. The main input and relative output paths
resolve from the job's working directory. A relative `material_file` is anchored
at the directory containing the main input, so submitted jobs can find a material
file without depending on the scheduler's launch directory. The inherited
physical and algorithmic fields are documented in `../docs/INPUT_FILE.md`; all
configuration fields are demonstrated in
`inputs/notched_tension_cluster.in`.

MPI rank count and launcher choice intentionally do not belong in the
LAMMPS-like physics input. MPI must create the process world before Python can
parse that file. Use `--ranks`, `--mpi`, `MPI_HOME`, or `RUNTIME_ENV` at launch.

### Mixed-mode boundary schemes

At pseudo-time `t`, the requested relative edge displacement is

```text
Delta(t) = t * (max_sliding_displacement, max_displacement).
```

`max_sliding_displacement` is signed; `max_displacement` is the nonnegative
opening component. Select one of:

| `mechanical_bc_scheme` | Bottom | Top | Meaning |
|---|---|---|---|
| `legacy_roller_pin` | `uy=0`, lower-left `ux=0` | `uy=t*max_displacement`, `ux` free | Original mode-I case; nonzero sliding is rejected. |
| `relative_clamped` | `u=(0,0)` | `u=Delta(t)` | Full relative load applied at the top. |
| `symmetric_clamped` | `u=-Delta(t)/2` | `u=+Delta(t)/2` | Equal and opposite clamped-edge motion. |

For example:

```text
mechanical_bc_scheme relative_clamped
max_displacement 1.0e-3
max_sliding_displacement 5.0e-4
```

The sliding/opening displacement ratio is not automatically
`KII/KI`. It specifies boundary kinematics; actual near-tip mode mixity also
depends on geometry, finite boundaries, crack length, and material contrast.

### Spatial material coefficients

The ordinary `young_modulus`, `poisson_ratio`, `fracture_toughness`, and
`length_scale` values are the left (`x=0`) or bottom (`y=0`) endpoints. Their
`*_end` fields are the right (`x=width`) or top (`y=height`) endpoints. For any
supported property `p`, the built-in modes are

```text
material_mode uniform   # p(x,y) = p_start
material_mode linear_x  # p = p_start + (p_end-p_start)*x/width
material_mode linear_y  # p = p_start + (p_end-p_start)*y/height
```

File mode uses a separate command file:

```text
material_mode file
material_file materials/inclusion.material
write_material_fields true
```

Its grammar is:

```text
profile PROPERTY constant VALUE
profile PROPERTY linear_x LEFT RIGHT
profile PROPERTY linear_y BOTTOM TOP
profile PROPERTY affine ORIGIN DX DY
region NAME box XMIN XMAX YMIN YMAX PRIORITY
region NAME circle CX CY RADIUS PRIORITY
override NAME PROPERTY VALUE
```

`PROPERTY` is `young_modulus`, `poisson_ratio`, `fracture_toughness`, or
`length_scale`; an affine profile means
`p(x,y)=ORIGIN+DX*x+DY*y`. Profiles cover the domain and higher-priority region
overrides replace named properties on cells whose midpoint lies in the region;
region priorities must be unique. A property with no explicit profile inherits
its ordinary main-input value as a constant.
The solver validates global coefficient ranges and local `h/ell`; when requested,
it writes the four static DG0 fields once to `material_fields.xdmf/.h5`.

Runnable examples at a user-selected rank count are:

```bash
RANKS=4
INPUT_FILE=linux_cluster/inputs/mixed_mode.in \
  OUTPUT_DIR="results/mixed_mode_${RANKS}r" \
  bash linux_cluster/run_linux.sh --ranks "${RANKS}"

INPUT_FILE=linux_cluster/inputs/graded_linear_x.in \
  OUTPUT_DIR="results/gradient_${RANKS}r" \
  bash linux_cluster/run_linux.sh --ranks "${RANKS}"

INPUT_FILE=linux_cluster/inputs/graded_file.in \
  OUTPUT_DIR="results/inclusion_${RANKS}r" \
  bash linux_cluster/run_linux.sh --ranks "${RANKS}"
```

The Linux backend adds these PETSc fields:

| Keyword | Default | Meaning |
|---|---|---|
| `displacement_snes_type` | `newtonls` | PETSc nonlinear method for mechanics. |
| `displacement_ksp_type` | `gmres` | Krylov method for the spectral-split tangent. |
| `displacement_pc_type` | `gamg` | Mechanical preconditioner. |
| `phase_snes_type` | `vinewtonrsls` | Reduced-space VI solver; only `vinewtonrsls` or `vinewtonssls` is accepted. |
| `phase_ksp_type` | `cg` | Krylov method on the SPD phase operator/inactive set. |
| `phase_pc_type` | `gamg` | Phase preconditioner. |
| `ksp_relative_tolerance` | `1e-10` | Relative KSP tolerance for both subproblems. |
| `ksp_absolute_tolerance` | `1e-14` | Absolute KSP tolerance. |
| `ksp_max_iterations` | `1000` | KSP iteration cap. |
| `petsc_monitor` | `false` | Print SNES iteration residuals from rank 0. |

CLI values such as `--output-dir`, `--nx`, `--steps`,
`--max-sliding-displacement`, `--mechanical-bc-scheme`, `--material-mode`,
`--material-file`, `--u-ksp-type`, and `--phase-pc-type` override the input.
For advanced PETSc options, use the
prefixed environment database without editing code:

```bash
export PETSC_OPTIONS="-pf_u_ksp_monitor -pf_phi_ksp_monitor"
bash linux_cluster/run_linux.sh --ranks auto
```

The prefixes are `pf_u_` for mechanics and `pf_phi_` for phase evolution.

## 7. What is parallelized

1. DOLFINx partitions the triangular mesh and all function spaces.
2. A file-defined material specification is normalized before the solve and
   shared consistently. Each rank evaluates DG0 coefficients on its owned
   cells; coefficient extrema and region cell counts are collective reductions.
3. Mechanical residual/Jacobian assembly is distributed; PETSc SNES performs
   the global Newton solve with KSP/preconditioning.
4. For fixed history, `F(phi) = A*phi - b` is the gradient of the convex AT2
   phase quadratic. PETSc SNESVI solves its box variational inequality with
   `0 <= phi <= phi_accepted`; equal zero bounds retain the starter notch.
5. An independent distributed projected-gradient infinity norm verifies the
   KKT conditions after every phase solve.
6. Grip reactions sum owned entries of the assembled unconstrained mechanical
   residual for each boundary/component, followed by MPI reductions. The
   top-plus-bottom `force_balance_x/y` values therefore directly check discrete
   equilibrium. Energy, residual, min/max, active-set, and convergence metrics use
   MPI reductions, so every rank takes the same accept/cutback branch.
7. XDMF/HDF5 field writes are collective, including optional static material
   fields. Manifest, CSV, summary, console, and
   plot files are rank-0-only. The crack-front diagnostic gathers global P1
   topology/values to rank 0 and broadcasts the connected-front result.

## 8. Output and performance notes

The result schema matches the serial version and adds `backend` and `mpi_ranks`
to the manifest/summary. Each accepted CSV row contains
`opening_displacement`, `sliding_displacement`, `top_reaction_x/y`,
`bottom_reaction_x/y`, and `force_balance_x/y`. Compatibility aliases remain
`displacement=opening_displacement` and `reaction_force=top_reaction_y`.
Connected-crack output includes `crack_tip_x/y`, `crack_path_length`,
`crack_kink_angle_degrees`, and `connected_crack_node_count` in addition to the
older x-front and extension measures.
The supplied PNG shows top-y reaction versus relative opening and, when sliding
is nonzero, a second top-x-reaction versus relative-sliding panel. Use the CSV
for custom sign conventions and comparisons.

For file-defined materials, the manifest records the normalized specification,
resolved source path and SHA, property ranges, and global region cell counts.
The output path must be visible at the same location on every node. Prefer a
parallel/shared scratch filesystem for collective XDMF/HDF5 and copy completed
results to long-term storage afterward.

More ranks do not automatically make a small case faster. The 20 x 20 smoke
mesh exists only to verify correctness. The production example uses 80,000
triangles; benchmark a sweep up to the detected physical-core/allocation
capacity before choosing a production count. The hardware-specific Xeon16 suite
uses 1/2/4/8/16; larger cluster allocations should use an analogous fixed-work
sweep suited to their mesh and site.

Conda MPICH is convenient and reproducible, but it may not exploit a cluster's
high-speed fabric as well as the site MPI. The automatic probe establishes
whether a current launcher can start this runtime; it does not make an
independently built PETSc or HDF5 library compatible. For a tuned multi-node
deployment, activate an administrator-provided coherent FEniCSx/PETSc/MPI
module, container, or Spack environment and use `RUNTIME_ENV=active`.

Current limitations include no checkpoint/restart, a rank-0 gather for crack
connectivity/final plotting, and scheduler templates that necessarily require
site-specific account/queue choices. The PETSc source can be syntax- and
configuration-tested on Windows, but every selected multi-rank configuration
must be validated by the supplied preflight and smoke case on Linux.

For mixed-mode or heterogeneous studies, also account for these modeling
limitations:

- The imposed sliding/opening ratio is not a stress-intensity-factor ratio.
- Crack direction can depend on the spectral tension/compression split, the
  diffuse starter notch, regularization length, and mesh orientation.
- No separate frictional contact law is added for closed crack faces; shear
  transfer under compression therefore requires careful interpretation.
- Smooth profiles and curved/non-fitted interfaces are represented by DG0
  cell-midpoint values. Refine the mesh to assess gradient and interface bias.
- Changing the MPI rank count changes parallel decomposition, not the need for mesh,
  length-scale, load-step, and material-interface convergence studies.

Official references: [DOLFINx 0.11 PETSc API](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/python/generated/dolfinx.fem.petsc.html),
[PETSc SNES variational inequalities](https://petsc.org/release/manual/snes/#variational-inequalities),
and the [DOLFINx installation/HPC guidance](https://github.com/FEniCS/dolfinx).
