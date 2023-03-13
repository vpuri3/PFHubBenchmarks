import numpy as np
import time, os, shutil

import ufl
from ufl import TestFunction, dx, grad, inner

import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem import form, Function, FunctionSpace, Constant

from mpi4py import MPI
from petsc4py import PETSc

# local
import pfbase

""" Problem Setup """

Nx = 60
Ny = 75
msh = mesh.create_unit_square(MPI.COMM_WORLD, Nx, Ny,
                              cell_type = mesh.CellType.quadrilateral,
                              #cell_type = mesh.CellType.triangle,
                              #diagonal=mesh.DiagonalType.crossed

                              #ghost_mode = mesh.GhostMode.shared_facet
                              #ghost_mode = mesh.GhostMode.none
                              #ghost_mode = mesh.GhostMode.shared_vertex
                              )

Nx = 50
Ny = 50
Nz = 50
msh = mesh.create_unit_cube(MPI.COMM_WORLD, Nx, Ny, Nz,
                            #cell_type = mesh.CellType.tetrahedron,
                            cell_type = mesh.CellType.hexahedron,
                            )

V = FunctionSpace(msh, ("Lagrange", 1))
u = Function(V)
v = TestFunction(V)

u_bc = Function(V)
u_bc.x.array[:] = 1.0

bc_cond = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
dofs = fem.locate_dofs_geometrical(V, bc_cond)
bc = fem.dirichletbc(u_bc, dofs)

F = 5.0 * v * dx - ufl.sqrt(u * u) * inner(grad(u), grad(v)) * dx - inner(u, v) * dx

bcs = [bc,]

"""
SNES Solver
"""

if MPI.COMM_WORLD.rank == 0:
    print("#=================================================#")
    print("SNES Solve")
    print("#=================================================#")

u.x.array[:] = 0.9

problem = pfbase.SnesPDEProblem(F, u, bcs)

snes = PETSc.SNES().create()
snes.setFunction(problem.F, problem.vector())
snes.setJacobian(problem.J, problem.matrix())

# KSP  opts: https://petsc.org/release/docs/manualpages/KSP/KSPType/
# SNES opts: https://petsc.org/release/docs/manualpages/SNES/SNESLineSearchType/

opts = PETSc.Options()

snes.setTolerances(atol = 1e-6, max_it = 10)

opts[f"snes_linesearch_type"] = "basic"
opts[f"snes_monitor"] = None
opts[f"snes_converged_reason"] = None
#opts[f"snes_linesearch_monitor"] = None

ksp = snes.getKSP()
#ksp.setTolerances(atol = 1e-8, max_it = 1000)

opts[f"ksp_type"] = "gmres"
#opts[f"ksp_monitor"] = None
#opts[f"ksp_gmres_restart"] = 100
opts[f"pc_type"]  = "sor"

pc  = ksp.getPC()

snes.setFromOptions()

t0 = time.time()
snes.solve(None, u.vector)
tt = time.time() - t0

niters, converged = snes.getIterationNumber(), snes.converged
kspiter = snes.getLinearSolveIterations()
resid = snes.getFunctionNorm()

if MPI.COMM_WORLD.rank == 0:
    print("SNES solver converged = ", converged, " in ", niters, " Newton iterations, and ", kspiter, " Krylov iterations, and residual ", resid)
    print("Time taken: ", tt)

"""
Newton Solver
"""

if MPI.COMM_WORLD.rank == 0:
    print("#=================================================#")
    print("Newton Solve")
    print("#=================================================#")

opts.clear()
u.x.array[:] = 0.9

problem = pfbase.NewtonPDEProblem(F, u, bcs)

newton = dolfinx.cpp.nls.petsc.NewtonSolver(MPI.COMM_WORLD)
newton.setF(problem.F, problem.vector())
newton.setJ(problem.J, problem.matrix())
newton.set_form(problem.form)

newton.report = True
newton.error_on_nonconvergence = False
newton.max_it = 20
newton.atol = 1e-6

opts = PETSc.Options()

ksp = newton.krylov_solver
#ksp.setTolerances(atol = 1e-8, max_it = 100)
ksp_pfx = ksp.getOptionsPrefix()

opts[f"{ksp_pfx}ksp_type"] = "gmres"
#opts[f"{ksp_pfx}ksp_monitor"] = None
#opts[f"{ksp_pfx}ksp_gmres_restart"] = 100
opts[f"{ksp_pfx}pc_type"]  = "sor"
newton.relaxation_parameter = 1.0
opts[f"{ksp_pfx}pc_type"]  = "sor"

ksp.setFromOptions()

t0 = time.time()
n, converged = newton.solve(u.vector)
tt = time.time() - t0

if MPI.COMM_WORLD.rank == 0:
    print("Newton solver converged = ", converged, " in ", n, " Newton iterations.")
    print("Time taken: ", tt)

#
