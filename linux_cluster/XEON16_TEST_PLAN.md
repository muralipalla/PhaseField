# 16-core Xeon workstation test plan

This suite validates the Linux PETSc/MPI backend on one workstation with 16
affinity-visible **physical** CPU cores. It combines environment checks,
functional fracture cases, cross-boundary-condition regression, collective
field output, and a fixed-work 1/2/4/8/16-rank strong-scaling sweep.

## Before running

Create or update the Linux environment from the project root:

Environment file: [`linux_cluster/environment-linux.yml`](environment-linux.yml)

```bash
conda env create --file linux_cluster/environment-linux.yml
# If it already exists:
conda env update --name phasefield-fenicsx-linux \
  --file linux_cluster/environment-linux.yml --prune
```

The suite and each isolated run activate `phasefield-fenicsx-linux`. If
`conda` is not initialized in non-interactive shells, provide its base path:

```bash
export CONDA_BASE="$HOME/miniconda3"
```

The reference suite rejects a nonempty `PETSC_OPTIONS`. Site- or experiment-
specific PETSc overrides would make the supplied numerical comparisons
ambiguous; an explicit opt-in for modified experiments is documented below.

## One-command runs

The recommended first sequence is:

```bash
# Hardware guard, 16-rank preflight, and tiny solve
bash linux_cluster/run_xeon16_suite.sh smoke

# Functional physics and parallel-I/O validation at 16 ranks
bash linux_cluster/run_xeon16_suite.sh validation

# Warm-up followed by fixed-work runs at 1, 2, 4, 8, and 16 ranks
bash linux_cluster/run_xeon16_suite.sh scaling
```

After those pass, run everything together with three timing repetitions:

```bash
SCALING_REPEATS=3 bash linux_cluster/run_xeon16_suite.sh all
```

Every invocation gets a unique directory such as
`results/xeon16_20260831T120000Z_12345`. A custom path must not already exist:

```bash
OUT_ROOT="$PWD/results/my_xeon16_validation" \
  bash linux_cluster/run_xeon16_suite.sh all
```

The suite expects one host. It passes `--expected-nodes 1` to the collective
preflight. It never enables MPI oversubscription and keeps OpenMP, BLAS, MKL,
and NumExpr at one thread per MPI rank through the standard launcher.

## Supplied cases

| Case | Ranks | Mesh/cells | Purpose |
|---|---:|---:|---|
| `xeon16_smoke.in` | 16 | 40 x 40 / 3,200 | Communicator, SNES, SNESVI, assembly, and summary smoke test. |
| `xeon16_mode_i.in` | 16 | 100 x 100 / 20,000 | Legacy roller/pin Mode-I fracture and adaptive loading. |
| `xeon16_mixed_mode.in` | 16 | 80 x 80 / 12,800 | Relative-clamped opening plus sliding and vector reactions. |
| `xeon16_mixed_symmetric.in` | 16 | 80 x 80 / 12,800 | Same relative load split symmetrically between both grips. |
| `xeon16_graded_linear.in` | 16 | 80 x 80 / 12,800 | Analytic `linear_x` elasticity and toughness gradient. |
| `xeon16_graded_inclusion.in` | 16 | 80 x 80 / 12,800 | File-defined gradient/inclusion and collective material/result XDMF/HDF5. |
| `xeon16_scaling.in` | 1,2,4,8,16 | 320 x 320 / 204,800 | Fixed-work strong scaling with field and plot output disabled. |

The graded cases have deterministic DG0 midpoint expectations which the
checker verifies directly:

| Case | Quantity | Expected represented value |
|---|---|---:|
| Linear gradient | Young modulus range | 105.4375 to 209.5625 |
| Linear gradient | Fracture toughness range | 0.00180375 to 0.00269625 |
| Inclusion | `stiff_tough_inclusion` cells | 576 |
| Inclusion | Young modulus range | 168.175 to 315.0 |
| Inclusion | Poisson-ratio range | 0.28 to 0.30 |
| Inclusion | Fracture-toughness range | 0.0027 to 0.0032 |

The material source is re-read after each run and its SHA-256 must match the
manifest, so a changed or ignored inclusion file cannot silently pass.

The scaling mesh contains 103,041 phase DOFs, 206,082 displacement DOFs, and
204,800 DG0 history DOFs. At 16 ranks it averages 12,800 cells per rank. This is
large enough to expose useful trends on a workstation while keeping the
one-rank baseline practical. If it is still too small to scale on your Xeon,
copy the input, increase `nx` and `ny` together to 480 (which remains
notch-aligned), choose a matching resolved `length_scale`, and use that single
byte-identical input for every rank count.

## What is checked automatically

Before any solve, `xeon16_hardware.py` reads the process affinity mask and
Linux CPU topology. A 16-core/32-thread Xeon is counted as 16 physical cores,
not 32. Slurm or PBS task limits are also included when present. The suite
stops before MPI launch if the requested ranks exceed the effective capacity.

The fatal 16-rank preflight then checks:

- mpi4py and PETSc communicator congruence and exact rank/node placement;
- each MPI rank's Linux CPU-affinity mask, retained in `preflight.json`;
- compatible real-scalar DOLFINx, PETSc, petsc4py, and MPI versions;
- distributed KSP and bound-constrained SNESVI solves;
- collective XDMF/HDF5 write and read-back.

The final report also rejects incomplete or duplicated case ledgers, missing or
empty preflight/run logs, too few reported affinity masks, and multiple ranks
pinned to the same single CPU. A shared broad affinity mask is retained as a
timing warning because it is valid on launchers that leave binding to the OS.

For every phase-field case, the result checker requires:

- launcher status zero and `summary.json` status `completed`;
- requested rank count in both summary and manifest;
- final pseudo-time 1 and exact configured opening/sliding endpoints;
- finite CSV diagnostics and monotonically nondecreasing integrated damage;
- phase bounds, KKT residual, free mechanical residual, phase increment, and
  boundary violation within the configured acceptance tolerances;
- at every accepted load state and for both components,
  `|F_top + F_bottom| <= max(1e-8, 1e-5 max(|F_top|, |F_bottom|))`;
- material provenance/ranges and a nonempty file-defined inclusion region;
- every enabled CSV, JSON, PNG, XDMF, and HDF5 artifact present and nonempty.

The Mode-I fracture case additionally must advance the notch-connected
`phi <= 0.2` crack front by at least one mesh interval (`0.01 mm`). An increase
in diffuse damage alone is not treated as evidence of crack propagation.

The relative- and symmetric-clamped mixed-mode response curves are compared at
every accepted load state. They prescribe the same relative edge displacement,
so reactions, elastic/fracture energies, damage, and minimum phase should agree
within relative `1e-5` and absolute `1e-8` tolerances.

The scaling sweep is compared with the one-rank solution using relative
`5e-5` and absolute `1e-8` tolerances for reactions, energies, damage, phase
minimum, and crack metrics. This is deliberately looser than the same-rank
mixed-boundary comparison because reduction order and accepted nonlinear
iterates can vary slightly with partitioning. Bitwise equality is neither
expected nor required. Timing is marked non-comparable if accepted steps,
cutbacks, or aggregate nonlinear iteration work differ between ranks.

## Reports

Each suite directory contains:

- `suite_report.md`: short human-readable pass/fail and scaling report;
- `suite_summary.json`: complete machine-readable checks and comparisons;
- `suite_summary.csv`: one row per simulation run;
- `suite_manifest.json`: hardware, preflight, input, and repository provenance;
- `hardware.json` and `preflight.json`: structured capacity/runtime reports;
- `cases.tsv`: input SHA-256, paths, rank count, return status, and wall time;
- `logs/`: full preflight and solver console logs;
- `runs/`: isolated solver result directories.

The scaling table reports

```text
speedup    = median(T1) / median(Tp)
efficiency = speedup / p
```

The exact scaling input is run once at 16 ranks as a warm-up before timing. It
is listed in the run ledger and report but excluded from scaling statistics.
Its SHA-256 must equal the timed input's SHA-256.
This prevents the first measured case from paying the full FFCx JIT compilation
cost while later rank counts reuse cached forms. Use at least three repetitions
for a serious benchmark; the median is reported and individual runs remain
available.

Low parallel efficiency is a performance observation, not a physics failure.
The checker warns below 25% at the largest rank count but does not fail solely
on speed. PETSc GAMG coarse-grid work, MPI reductions, root-gathered crack
diagnostics, memory bandwidth, CPU frequency changes, and binding can all
produce a plateau.

## Controls and recovery

```bash
# Stop after the first failed simulation instead of collecting all failures
FAIL_FAST=1 bash linux_cluster/run_xeon16_suite.sh all

# Use a different compatible launcher
MPI_LAUNCHER=mpirun bash linux_cluster/run_xeon16_suite.sh smoke

# Conda MPICH/Hydra core binding; verify the accepted option with mpiexec -help
MPI_EXTRA_ARGS="-bind-to core" bash linux_cluster/run_xeon16_suite.sh scaling

# Custom ascending sweep; scaling mode requires first=1 and last=MAX_RANKS
RANK_SWEEP="1 4 8 16" bash linux_cluster/run_xeon16_suite.sh scaling

# Deliberately test PETSc overrides; they are rejected by default
ALLOW_PETSC_OPTIONS=1 PETSC_OPTIONS="..." \
  bash linux_cluster/run_xeon16_suite.sh validation
```

Do not use an Open MPI binding option with the bundled MPICH executable unless
the option is supported by that executable. The suite deliberately does not
hardcode vendor-specific binding flags. The preflight records every rank's
affinity and the report warns when all ranks share one broad, unbound mask. The
suite always rejects launcher oversubscription because its timings and
physical-core comparisons would no longer be meaningful.

The functional meshes intentionally sit near the accepted `h/ell <= 0.5`
resolution limit so the suite remains practical. They are software-validation
cases, not mesh-converged crack-path studies. For quantitative follow-up, repeat
with at least 160 x 160 cells for `ell=0.04`, or 200 x 200 for `ell=0.03`, and
perform explicit mesh and load-step convergence studies.

If a run fails, start with `suite_report.md`, then open its linked log and the
case's `summary.json`. The launcher continues independent cases by default, so
one physics failure does not hide unrelated material or I/O failures. A failed
preflight stops immediately because later results would not be trustworthy.
