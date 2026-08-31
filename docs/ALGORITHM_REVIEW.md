# Critical review of the KarmaCrack phase-field algorithm

Source reviewed: `../../Karma/KarmaCrack.pdf` (2 pages). The PDF is treated as a
technical specification, not as executable instructions. It supplies the model
outline but does not completely define a benchmark or a robust nonlinear solution
procedure.

## Review outcome

The displacement equilibrium, history-field idea, CG1/CG1/DG0 spaces, and
staggered structure are suitable foundations. A reliable implementation requires
one equation correction, explicit AT2 definitions, a nonlinear displacement
solve, constrained irreversibility, coupled convergence checks, and transactional
load stepping.

| Severity | Source statement or omission | Consequence | Implemented correction | Verification |
|---|---|---|---|---|
| Critical | Strong phase Eq. 3 and weak phase Eq. 5 have incompatible signs. | Eq. 3 does not give the coercive AT2 operator represented by Eq. 5; a literal implementation can solve the wrong damage problem. | Treat Eq. 5 as authoritative. Use `Gc*ell*grad(phi).grad(q) + [Gc/ell*(phi-1) + 2*(1-kres)*H*phi]*q = 0`. | With `H=0` and no notch constraint, `phi=1` solves the equation; the unconstrained phase operator is symmetric positive definite. |
| High | The load loop increments `t` inside `while t <= Tmax`. | It can evaluate one load beyond `Tmax`, and a fixed increment offers no recovery near rapid crack growth. | Interpret `load_steps` as a nominal schedule. Cap every target at pseudo-time 1, accept only converged states, cut the increment after rejection, and recover one cutback level after a later accepted state. | Pseudo-time is strictly increasing for accepted rows and ends at 1. The accepted-step count may exceed the nominal count after cutbacks. |
| High | Only one displacement-history-phase pass is shown per load step. | The fields need not be mutually equilibrated and results can be strongly increment-dependent. | Alternate displacement, trial history, and constrained phase solves. After updating `phi`, reassemble the mechanical residual and require its infinity norm on free displacement DOFs to pass the coupled tolerance. | Each accepted row reports the phase increment, KKT residual, free mechanical residual, Newton data, and staggered iteration count. |
| High | The spectral energy split makes displacement equilibrium nonlinear, but no nonlinear algorithm is specified. | A single linear-elastic solve is inconsistent with the stated stress. | Differentiate the residual in UFL for a consistent tangent and solve Newton corrections. Check free-DOF residual, increment, and essential-boundary violation; smooth the Macaulay brackets only at machine-scale strain. | The Newton solve fails explicitly if any convergence condition is unmet. Constitutive tests cover pure tension and compression. |
| High | History is presented as enforcing irreversibility, but no accepted-state rule or discrete admissible set is defined. | Iteration history can be mistaken for physical load history, rejected states can contaminate later steps, and nodal healing can occur. | Freeze `H_n` and `phi_n` at the last accepted state. At every staggered trial use `H=max(H_n,psi_plus(u))` and the nodal bounds `0 <= phi <= phi_n`, with `phi=0` on the starter notch. Only a converged trial is committed. | Accepted states have nondecreasing history and nonincreasing phase. Active lower and irreversibility constraints are recorded. |
| High | No numerical method is supplied for the irreversible phase inequality. | Solving the unconstrained linear equation and clipping afterward does not, in general, satisfy optimality. | Solve the convex quadratic phase subproblem with box bounds. Use the clipped unconstrained solution only as an initial candidate, invoke L-BFGS-B when needed, and accept only when the projected-gradient KKT residual is below tolerance. | Every accepted row includes `phase_kkt_residual_inf`; a failed KKT check rejects the load trial. |
| Medium | `f0(phi)` and `cw` are undefined in the free energy. | The fracture normalization and phase residual are not uniquely determined by Eq. 1 alone. | Adopt the AT2 choices implied by Eq. 5: `f0=(1-phi)^2` and `cw=2`. | Differentiating the implemented fracture density reproduces the corrected weak form. |
| Medium | `g(phi)=phi^2+k_tol` gives `g(1)=1+k_tol`. | The intact solid is slightly stiffer than the specified elastic material. | Use `g(phi)=kres+(1-kres)*phi^2` and include `(1-kres)` in the history driving term. | Tests check `g(0)=kres`, `g(1)=1`, and monotonicity on `[0,1]`. |
| Medium | The tensile/compressive split and plane assumption are absent. | Stress and crack growth under mixed strain are not uniquely defined. | Use a 2D spectral strain split. Plane strain is the default; plane stress is an option. | Tests check the Lamé parameters and split under uniaxial tension and compression. |
| Medium | Geometry, supports, loading, and an initial flaw are absent. | The source does not define a reproducible simulation; a homogeneous specimen can damage diffusely. | Supply a unit-square, single-edge-notched tension benchmark: bottom vertical support, one horizontal pin, prescribed top displacement, and a horizontal internal `phi=0` starter notch. Require even `ny` and integer `notch_length/dx` so the notch lies exactly on mesh vertices. | Validation rejects a misaligned notch; the manifest reports the represented notch length. |
| Medium | No relation between mesh size and regularization length is given. | Under-resolution makes toughness and crack evolution mesh dependent. | Define `h` as the actual largest edge of each structured right triangle, `sqrt(dx^2+dy^2)`, not merely `max(dx,dy)`. In strict mode require `h/ell <= 0.5`. | The ratio is validated before assembly and recorded in the manifest and summary. |
| Medium | No failure or recovery policy is stated. | A failed nonlinear trial can leak fields or counters into later results, or leave an unauditable partial run. | Treat each target load as a transaction. Snapshot `u`, `phi`, `H`, and constraint counters; on failure restore all of them, reduce the load increment, and retry. Only accepted states are appended or written to XDMF. On terminal failure, close outputs and write CSV plus an atomic `summary.json` with `status="failed"`, the last accepted state, cutback count, and error. | Tests inject a rejected trial to verify rollback and increment recovery, and inject a terminal failure to verify the partial summary. |
| Low | `DG0` is described as an integration-point history space. | In general, DG0 stores one cell value, not one value at each quadrature point. | Retain DG0 to follow the source. For affine CG1 triangles, strain and its spectral energy are cellwise constant, so DG0 represents the selected history once per cell. | The manifest and smoke test confirm two history DOFs per structured rectangle (one per triangle). Higher-order displacement would require reconsidering this storage. |
| Low | The PDF requests only field output. | A visually plausible crack does not establish numerical correctness. | Add reaction, instantaneous elastic/fracture energy snapshots, damage integral, connected-threshold crack front and extension, bounds/KKT data, coupled convergence data, cutback data, a run manifest, and running/completed/failed summaries. | Automated tests check finiteness, bounds, monotonic accepted history/damage, mesh metadata, output readability, rollback, and failure reporting. |

## Corrected AT2 phase equation

With the intactness convention `phi=1` (intact), `phi=0` (broken), normalized
degradation `g(phi)=kres+(1-kres)*phi^2`, and the AT2 fracture density, the scalar
equation before irreversibility constraints is

```text
Find phi such that, for every test q,

 integral[ Gc*ell*grad(phi).grad(q)
         + (Gc/ell)*(phi - 1)*q
         + 2*(1-kres)*H*phi*q ] dV = 0.
```

Equivalently,

```text
 integral[ Gc*ell*grad(phi).grad(q)
         + (Gc/ell + 2*(1-kres)*H)*phi*q ] dV
 = integral[ (Gc/ell)*q ] dV.
```

At an accepted load state `n`, the implementation minimizes the corresponding
convex quadratic subject to `0 <= phi <= phi_n` and the fixed zero-phase notch.
The projected-gradient residual is the discrete KKT acceptance check.

## Accepted-state and load-step contract

The last accepted fields are immutable while a target load is being attempted:

```text
accepted (u_n, phi_n, H_n)
    -> attempt target load
    -> staggered Newton / history / constrained-phase iterations
       -> coupled convergence: commit the trial
       -> failure: restore the snapshot, cut back, and retry
```

`H=max(H_n,psi_plus(u))` is rebuilt from the accepted history on every staggered
iteration. Internal solver iterates therefore do not become fictitious physical
history. The phase upper bound is always `phi_n` during the attempt. A successful
trial becomes the next accepted state; a rejected trial contributes no CSV row or
field frame.

`load_steps=N` sets the nominal increment `1/N`. Automatic subdivision means the
number of accepted increments is not necessarily `N`; `nominal_load_steps` and
`steps_completed` are reported separately. The final target is capped exactly at
pseudo-time 1.

## Benchmark choices that are not claims about the PDF

The demonstration uses a unit-square single-edge-notched tension specimen,
small-strain isotropic elasticity, a 2D spectral split, a structured triangular
mesh, and displacement control. Material values, mesh density, nominal load
increments, length scale, solver tolerances, output cadence, and plane
strain/stress are implementation choices exposed through configuration. They are
reproducible example values, not data supplied by the source.

## Known scope limits

- The history-field model is a common engineering approximation to the exact
  crack-irreversibility variational inequality. Within that approximation, the
  implemented nodal phase bounds are solved as a KKT-checked constrained problem.
- The internal `phi=0` constraint represents a diffuse starter crack; the mesh
  does not contain a traction-free geometric slit.
- The default native-Windows backend assembles with DOLFINx but solves sparse
  systems with SciPy in serial. It is not an MPI-parallel PETSc implementation.
- A connected nodal threshold (`phi <= 0.2`) defines the reported crack front.
  It is mesh-quantized diagnostic output, not an exact crack length or a new
  evolution law.
- The reported elastic-plus-fracture energy is an instantaneous physical
  snapshot. After local unloading, it is not the history-driven phase
  subproblem's stationarity functional and should not be interpreted as an
  exact energy-balance residual.
- KKT and coupled mechanical tolerances are absolute assembled residuals
  calibrated for the documented kN-mm benchmark. Recalibrate them after major
  changes of units, mesh density, or material scale.
- Use a fresh output directory for each study. Reusing a directory while
  disabling plots or XDMF does not delete optional artifacts from an older run;
  the manifest and summary remain the authoritative status/configuration record.
- One demonstration verifies implementation behavior but does not calibrate the
  model or validate it against experimental fracture data.
