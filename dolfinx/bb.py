#
import numpy as np

import ufl
from ufl import TestFunction
from ufl import dx, grad, inner

import dolfinx
from dolfinx import fem, mesh
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
 SNES
"""

problem = nlsolvers.SnesPDEProblem(F, u, bcs)

snes = PETSc.SNES().create()
snes.setFunction(problem.F, problem.vector())
snes.setJacobian(problem.J, problem.matrix())

opts = PETSc.Options()
opts['snes_linesearch_type'] = 'basic'
opts['snes_monitor'] = None
opts['snes_linesearch_monitor'] = None
snes.setFromOptions()

snes.setTolerances(rtol = 1e-9, max_it = 10)
ksp = snes.getKSP()
ksp.setType("preonly")
ksp.setTolerances(rtol = 1e-9)

pc = ksp.getPC()
pc.setType("lu")

snes.solve(None, u.vector)
niters, converged = snes.getIterationNumber(), snes.converged

print("Converged = ", converged, " in ", niters, " iterations.")
#
