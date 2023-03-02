#
import numpy as np

import ufl
from ufl import TestFunction
from ufl import dx, grad, inner

import dolfinx
from dolfinx import fem, io, mesh, plot, la
from dolfinx.fem import form, Function, FunctionSpace, Constant

from mpi4py import MPI
from petsc4py import PETSc

import nlsolvers

msh = mesh.create_unit_square(MPI.COMM_WORLD, 12, 15)
V = FunctionSpace(msh, ("Lagrange", 1))
u = Function(V)
v = TestFunction(V)
F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(grad(u), grad(v)) * dx - inner(u, v) * dx

u_bc = Function(V)
bc = fem.dirichletbc(u_bc, fem.locate_dofs_geometrical(V, lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))))

bcs = [bc,]

u_bc.x.array[:] = 1.0
u.x.array[:] = 0.9

"""
 Newton Solve
"""

problem = nlsolvers.NewtonPDEProblem(F, u, bcs)

solver = dolfinx.cpp.nls.petsc.NewtonSolver(MPI.COMM_WORLD)
solver.setF(problem.F, problem.vector())
solver.setJ(problem.J, problem.matrix())
solver.set_form(problem.form)
n, converged = solver.solve(u.vector)

print("Converged = ", converged, " in ", n, " iterations.")

"""
 SNES
"""

u.x.array[:] = 1.0

problem = nlsolvers.SnesPDEProblem(F, u, bcs)

snes = PETSc.SNES().create()
snes.setFunction(problem.F, problem.vector())
snes.setJacobian(problem.J, problem.matrix())

snes.setTolerances(rtol = 1e-9, max_it = 10)
ksp = snes.getKSP()
ksp.setType("preonly")
ksp.setTolerances(rtol = 1e-9)

pc = ksp.getPC()
pc.setType("lu")

snes.solve(None, u.vector)
converged = snes.getConvergedReason()
n = snes.getIterationNumber()

print("Converged = ", converged, " in ", n, " iterations.")
#
