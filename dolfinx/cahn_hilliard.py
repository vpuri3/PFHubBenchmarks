#
import numpy as np

import ufl
from ufl import ds, dx, grad, inner, dot, variable, diff, derivative
from ufl import sin, cos, tan, log, exp

import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem import Function, FunctionSpace, Constant

from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from dolfinx.nls.petsc import NewtonSolver

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType, SNES

# vis
from dolfinx import io, plot

"""
Solve Cahn-Hilliard Equation

  d/dt c = \div(M \grad(mu))
      mu = f'(c) - kappa * Delta c

  on \Omega = [0, 1]^2
  n \cdot \grad  c = 0 on \partial\Omega
  n \cdot \grad mu = 0 on \partial\Omega

f = 100 * c^2 * (1-c)^2
kappa = 1e-2
M = 1.0
"""

# mesh
Nx = 96
Ny = 96

Lx = 1.0
Ly = 1.0

msh = mesh.create_rectangle(comm = MPI.COMM_WORLD,
                            points = ((0.0, 0.0), (Lx, Ly)), n = (Nx, Ny),
                            cell_type = mesh.CellType.triangle,
                            )

P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
W  = FunctionSpace(msh, P1 * P1)
x  = ufl.SpatialCoordinate(msh)

c , mu  = w  = Function(W)
c0, mu0 = w0 = Function(W)
c_, mu_ = w_ = ufl.TestFunction(W)
dc, dmu = dw = ufl.TestFunction(W)

# parameters
M = Constant(msh, 1.0)
kappa = Constant(msh, 1e-2)

_c = variable(c)
f = 100.0 * _c**2 * (1 - _c)**2
dfdc = ufl.diff(f, _c)

# initial conditions
ic_c  = lambda x: 0.63 + 0.02 * (0.5 - np.random.rand(x.shape[1]))
ic_mu = lambda x: np.zeros(x.shape[1])

w.x.array[:] = 0
w.sub(0).interpolate(ic_c)
w.sub(1).interpolate(ic_mu)

w0.x.array[:] = 0
w0.sub(0).interpolate(ic_c)
w0.sub(1).interpolate(ic_mu)

# Scatter forward the solution vector to update ghost values
w.x.scatter_forward()
w0.x.scatter_forward()

# boundary conditions
bcs = []

# weak form
dt = Constant(msh, 1e-4)

"""
d/dt c  = div(M * grad(\mu))
    \mu = F'(c) - \kappa * lapl(c)
"""

# CH - c
Fc_lhs =  c_ * ((c - c0) / dt) * dx
Fc_rhs = -inner(grad(c_), M * grad(mu)) * dx

Fc = Fc_lhs - Fc_rhs

# CH - mu
Fmu_lhs = mu_ * mu * dx
Fmu_rhs = mu_ * dfdc * dx + kappa * inner(grad(mu_), grad(c)) * dx

Fmu = Fmu_lhs - Fmu_rhs

# CH
F = Fc + Fmu

# solve
J = derivative(F, w, dw)
problem = NonlinearProblem(F, w, bcs)#, J = J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)

solver.convergence_criterion = "incremental"
solver.rtol = 1e-6

ksp  = solver.krylov_solver
opts = PETSc.Options()
ksp_pfx  = ksp.getOptionsPrefix()
opts[f"{ksp_pfx}ksp_type"] = "preonly"
opts[f"{ksp_pfx}pc_type"]  = "lu"
opts[f"{ksp_pfx}pc_factor_mat_solver_type"]  = "mumps"
ksp.setFromOptions()

file = io.XDMFFile(MPI.COMM_WORLD, "ch/out.xdmf", "w")
file.write_mesh(msh)

###################################
# time integration
###################################

t = Constant(msh, 0.0)
tprev = 0.0

dt_min = 1e-2
dt.value = 1e-6

T = 50 * dt

#V0, dofs = W.sub(0).collapse()

while (float(t) < float(T)):
    t += dt
    r = solver.solve(w)
    print(f"Step {int(float(t)/float(dt))}: num iterations: {r[0]}")
    w0.x.array[:] = w.x.array

    c, mu = w.sub(0), w.sub(1)
    file.write_function(c, t)

file.close()
#
