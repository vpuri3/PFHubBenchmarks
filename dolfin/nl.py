#
from dolfin import *

import numpy as np
import time

###################################
# Optimization options for the finite element form compiler
###################################
parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math'
parameters["form_compiler"]["quadrature_degree"] = 3

###################################
# Mesh
###################################
Nx = 60
Ny = 75
mesh = UnitSquareMesh(Nx, Ny, 'crossed')
mesh = UnitSquareMesh(Nx, Ny)

#Nx = 20
#Ny = 20
#Nz = 10
#mesh = UnitCubeMesh(Nx, Ny, Nz)

V = FunctionSpace(mesh, "Lagrange", 1)
u = Function(V)
v = TestFunction(V)
du = TrialFunction(V)

u_bc = Function(V)
u_bc.interpolate(Constant(0.9))
bc_loc = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
bc = DirichletBC(V, u_bc, bc_loc)

F = 5.0 * v * dx - sqrt(u * u) * inner(grad(u), grad(v)) * dx - inner(u, v) * dx
J = derivative(F, u, du)
bcs = [bc,]

problem = NonlinearVariationalProblem(F, u, bcs, J)

"""
SNES Solver
"""
u.vector()[:] = 0.9

snes = NonlinearVariationalSolver(problem)

snes.parameters['nonlinear_solver'] = 'snes'
nlparams = snes.parameters['snes_solver']
nlparams['line_search'] = 'basic'

nlparams['report'] = True
nlparams['error_on_nonconvergence'] = False
nlparams['absolute_tolerance'] = 1e-6
nlparams['maximum_iterations'] = 10

nlparams['linear_solver'] = 'gmres'
nlparams['preconditioner'] = 'sor'

nlparams['krylov_solver']['maximum_iterations'] = 1000
nlparams['krylov_solver']['error_on_nonconvergence'] = False
#nlparams['krylov_solver']['monitor_convergence'] = True

niters, converged = snes.solve()

if MPI.rank(mesh.mpi_comm()) == 0:
    print("SNES converged =", converged, ", niters =", niters)

"""
Newton Solver
"""
u.vector()[:] = 0.9

solver = NonlinearVariationalSolver(problem)

solver.parameters['nonlinear_solver'] = 'newton'
nlparams = solver.parameters['newton_solver']

nlparams['report'] = True
nlparams['error_on_nonconvergence'] = False
nlparams['absolute_tolerance'] = 1e-6
nlparams['maximum_iterations'] = 10

nlparams['linear_solver'] = 'gmres'
nlparams['preconditioner'] = 'sor'

nlparams['krylov_solver']['maximum_iterations'] = 1000
nlparams['krylov_solver']['error_on_nonconvergence'] = False
#nlparams['krylov_solver']['monitor_convergence'] = True

niters, converged = solver.solve()

if MPI.rank(mesh.mpi_comm()) == 0:
    print("Newton converged =", converged, ", niters =", niters)

#
