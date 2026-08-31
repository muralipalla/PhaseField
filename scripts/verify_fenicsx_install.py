"""Exercise FEniCSx JIT, assembly, boundary lifting, and a serial solve."""

from mpi4py import MPI
import numpy as np
from scipy.sparse.linalg import spsolve
import ufl

import dolfinx
from dolfinx import fem, mesh


if MPI.COMM_WORLD.size != 1:
    raise RuntimeError(
        "This verifier targets the serial Windows/SciPy backend; run it with "
        "exactly one MPI rank."
    )

domain = mesh.create_unit_square(MPI.COMM_WORLD, 4, 4)
space = fem.functionspace(domain, ("Lagrange", 1))

trial = ufl.TrialFunction(space)
test = ufl.TestFunction(space)
source = fem.Constant(domain, np.float64(1.0))
bilinear = fem.form(ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx)
linear = fem.form(source * test * ufl.dx)

facet_dimension = domain.topology.dim - 1
facets = mesh.locate_entities_boundary(
    domain,
    facet_dimension,
    lambda x: np.isclose(x[0], 0.0)
    | np.isclose(x[0], 1.0)
    | np.isclose(x[1], 0.0)
    | np.isclose(x[1], 1.0),
)
dofs = fem.locate_dofs_topological(space, facet_dimension, facets)
boundary_condition = fem.dirichletbc(np.float64(0.0), dofs, space)

matrix = fem.assemble_matrix(bilinear, bcs=[boundary_condition]).to_scipy()
right_hand_side = fem.assemble_vector(linear)
fem.apply_lifting(
    right_hand_side.array,
    [bilinear],
    bcs=[[boundary_condition]],
)
boundary_condition.set(right_hand_side.array)

solution = spsolve(matrix, right_hand_side.array)
solution_norm = float(np.linalg.norm(solution))
if not np.all(np.isfinite(solution)) or solution_norm <= 0.0:
    raise RuntimeError("The assembled Poisson solve did not produce a finite solution.")

print(f"dolfinx={dolfinx.__version__}")
print(f"mpi_size={MPI.COMM_WORLD.size}")
print(f"matrix_shape={matrix.shape}, nnz={matrix.nnz}")
print(f"solution_l2={solution_norm:.12e}")
print("FENICSX_INSTALL_OK")
