#
import numpy as np

import ufl
from ufl import ds, dx, grad, inner, dot, variable, diff, derivative
from ufl import sin, cos, tan, log, exp, pi

import dolfinx
from dolfinx import fem, io, mesh, plot
from dolfinx.fem import Function, FunctionSpace, Constant

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

#import pyvista

# local
from pfbase import *
import nlsolvers

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

w = Function(V)
w_ = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)

F = poisson_weak_form(w, w_, f, -1.0) - inner(g, w_) * ds

"""
 Standard Newton Solve
"""

from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

problem = NonlinearProblem(F, w, bcs)
solver = NewtonSolver(MPI.COMM_WORLD, problem)
solver.rtol = 1e-6

niters, converged = solver.solve(w)
print("Converged = ", converged, " in ", niters, " iterations.")

"""
 Newton Solve
"""
w.x.array[:] = 1.0

problem = nlsolvers.NewtonPDEProblem(F, w, bcs)

solver = dolfinx.cpp.nls.petsc.NewtonSolver(MPI.COMM_WORLD)
solver.setF(problem.F, problem.vector())
solver.setJ(problem.J, problem.matrix())
solver.set_form(problem.form)

solver.rtol = 1e-6
niters, converged = solver.solve(w.vector)
print("Converged = ", converged, " in ", niters, " iterations.")

"""
 SNES
"""
w.x.array[:] = 1.0

problem = nlsolvers.SnesPDEProblem(F, w, bcs)

solver = PETSc.SNES().create()
solver.setFunction(problem.F, problem.vector())
solver.setJacobian(problem.J, problem.matrix())

solver.setTolerances(rtol = 1e-6, max_it = 20)

solver.solve(None, w.vector)
converged = solver.getConvergedReason()
niters = solver.getIterationNumber()

print("Converged = ", converged, " in ", niters, " iterations.")

#
