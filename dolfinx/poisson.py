#
import numpy as np

import ufl
from ufl import ds, dx, grad, inner, dot, variable, diff, derivative
from ufl import sin, cos, tan, log, exp, pi

import dolfinx
from dolfinx import fem, io, mesh, plot
from dolfinx.fem import Function, FunctionSpace, Constant

from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

#import pyvista

# local
from pfbase import *

"""
Solve Poisson equation
        -\Delta u = f on \Omega   = [0, 2] \cross [0, 1]
                u = 0 on \Gamma_D = {(0, y) U (2, y)}
  n \cdot \grad u = g on \Gamma_N = {(x, 0) U (x, 1)}

f = 10 * exp(-((x-0.5)**2 + (y - 0.5)**2) / 0.02 )
g = sin(5 * x)
"""

# mesh
Nx = 32
Ny = 16

Lx = 2.0
Ly = 1.0

msh = mesh.create_rectangle(comm = MPI.COMM_WORLD,
                            points = ((0.0, 0.0), (Lx, Ly)), n = (Nx, Ny),
                            cell_type = mesh.CellType.triangle,
                            )

V = fem.FunctionSpace(msh, ("Lagrange", 1))

# boundary condition
BC_dirichlet = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], Lx))
BC_neumann   = lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], Ly))

facets_dirichlet = mesh.locate_entities_boundary(msh, dim = 1, marker = BC_dirichlet)

dofs_dirichlet = fem.locate_dofs_topological(V = V, entity_dim = 1,
                                             entities = facets_dirichlet)

bc_dirichlet = fem.dirichletbc(value = ScalarType(0), dofs = dofs_dirichlet, V = V)

bcs = [
        bc_dirichlet,
       ]

# equation
x = ufl.SpatialCoordinate(msh)

f = 10 * exp(-((x[0]-0.5)**2 + (x[1] - 0.5)**2) / 0.02)
g = sin(5 * x[0])

#f = Constant(msh, 1.0)
#g = Constant(msh, 0.0)

#############
# LinearProblem
#############
#u = ufl.TrialFunction(V)
#v = ufl.TestFunction(V)
#
#a = inner(grad(u), grad(v)) * dx
#L = inner(f, v) * dx + inner(g, v) * ds
#
## solve
#petsc_options = {
#    "ksp_type" : "gmres",
#    "pc_type" : "sor",
#    }
#
#problem = fem.petsc.LinearProblem(a, L, bcs = bcs, petsc_options = petsc_options)
#uh = problem.solve()
#############

#############
# NonlinearProblem
#############

w = Function(V)
w_ = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)

F = poisson_weak_form(w, w_, f, -1.0) - inner(g, w_) * ds

problem = NonlinearProblem(F, w, bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)

solver.convergence_criterion = "incremental" # 'residual'
solver.rtol = 1e-6

solver.report = True
solver.error_on_nonconvergence = False

ksp  = solver.krylov_solver
opts = PETSc.Options()
pfx  = ksp.getOptionsPrefix()

opts[f"{pfx}ksp_type"] = "gmres" # "cg", "bicgstab"
opts[f"{pfx}pc_type"]  = "sor"   # "lu"

opts[f"{pfx}pc_factor_mat_solver_type"]  = "mumps"

ksp.setFromOptions()

niters, converged = solver.solve(w)
#############

# saving and visualization
if os.path.exists("out_poisson"):
    shutil.rmtree("out_poisson")
file = io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w")
file.write_mesh(msh)
file.write_function(w)
file.close()
#
