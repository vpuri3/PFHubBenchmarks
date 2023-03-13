#
import numpy as np
import time, os, shutil

import ufl
from ufl import TestFunction
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

# local
import pfbase

"""
Solve Poisson equation
        -\Delta u = f on \Omega   = [0, 2] \cross [0, 1]
                u = 0 on \Gamma_D = {(0, y) U (2, y)}
  n \cdot \grad u = g on \Gamma_N = {(x, 0) U (x, 1)}

f = 10 * exp(-((x-0.5)**2 + (y - 0.5)**2) / 0.02 )
g = sin(5 * x)
"""

###################################
# Mesh
###################################
Nx = 32
Ny = 32
Nz = 32

Lx = 1.0
Ly = 1.0
Lz = 1.0

msh = mesh.create_box(comm = MPI.COMM_WORLD,
                      points = ((0.0, 0.0, 0.0), (Lx, Ly, Lz)), n = (Nx, Ny, Nz),
                      #cell_type = mesh.CellType.hexahedron,
                      cell_type = mesh.CellType.tetrahedron,
                     )

###################################
# Function Space
###################################
W = FunctionSpace(msh, ("Lagrange", 1))
w  = Function(W)
w_ = TestFunction(W)

x  = ufl.SpatialCoordinate(msh)

###################################
# boundary condition
###################################

bdry_x = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], Lx))
bdry_y = lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], Ly))
bdry_z = lambda x: np.logical_or(np.isclose(x[2], 0.0), np.isclose(x[2], Lz))

faces_x = mesh.locate_entities_boundary(msh, dim = 2, marker = bdry_x)
faces_y = mesh.locate_entities_boundary(msh, dim = 2, marker = bdry_y)
faces_z = mesh.locate_entities_boundary(msh, dim = 2, marker = bdry_z)

dof_x = fem.locate_dofs_topological(V = W, entity_dim = 2, entities = faces_x)
dof_y = fem.locate_dofs_topological(V = W, entity_dim = 2, entities = faces_y)
dof_z = fem.locate_dofs_topological(V = W, entity_dim = 2, entities = faces_z)

# dirichlet
bc_x = fem.dirichletbc(value = ScalarType(0), dofs = dof_x, V = W)
bc_y = fem.dirichletbc(value = ScalarType(0), dofs = dof_y, V = W)
bc_z = fem.dirichletbc(value = ScalarType(0), dofs = dof_z, V = W)

bcs = [
        bc_x,
        bc_y,
        bc_z,
       ]

# equation
x = ufl.SpatialCoordinate(msh)

f = sin(2 * pi * x[0]) * sin(3 * pi * x[1]) * sin(4 * pi * x[2])
g = Constant(msh, 0.0) # neumann data

#############
# Linear Problem
#############

#petsc_options = {
#    "ksp_type" : "gmres",
#    "pc_type" : "sor",
#    }
#dw = ufl.TrialFunction(W)
#a = inner(grad(w_), grad(dw)) * dx
#L = inner(w_, f) * dx
#problem = fem.petsc.LinearProblem(a, L, bcs = bcs, petsc_options = petsc_options)
#w = problem.solve()

#############
# NonlinearProblem
#############

#w.x.array[:] = 1.0

F = inner(grad(w_), grad(w)) * dx - inner(w_, f) * dx # - inner(w_, g) * ds

problem = pfbase.SnesPDEProblem(F, w, bcs)

snes = PETSc.SNES().create()
snes.setFunction(problem.F, problem.vector())
snes.setJacobian(problem.J, problem.matrix())

opts = PETSc.Options()

snes.setTolerances(atol = 1e-6, max_it = 10)

opts[f"snes_linesearch_type"] = "basic"
opts[f"snes_monitor"] = None
opts[f"snes_converged_reason"] = None
#opts[f"snes_linesearch_monitor"] = None

ksp = snes.getKSP()
ksp.setTolerances(atol = 1e-8, max_it = 1000)

opts[f"ksp_type"] = "gmres"
#opts[f"ksp_monitor"] = None
opts[f"pc_type"]  = "sor"

pc  = ksp.getPC()

snes.setFromOptions()

snes.solve(None, w.vector)
niters, converged = snes.getIterationNumber(), snes.converged
kspiter = snes.getLinearSolveIterations()
resid = snes.getFunctionNorm()

if MPI.COMM_WORLD.rank == 0:
    print("SNES solver converged = ", converged, " in ", niters,
          " Newton iterations, and ", kspiter, " Krylov iterations, ",
          "with residual ", resid)

#############

# saving and visualization
if os.path.exists("out_poisson"):
    if MPI.COMM_WORLD.rank == 0:
        shutil.rmtree("out_poisson")
file = io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w")
file.write_mesh(msh)
file.write_function(w)
file.close()
#
