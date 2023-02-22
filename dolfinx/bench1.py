#
import numpy as np
import time

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

# local
from pfbase import *

"""
Solve Cahn-Hilliard Equation

  d/dt c = \div(M \grad(mu))
      mu = f'(c) - kappa * Delta c

  on \Omega = [0, 1]^2
  n \cdot \grad  c = 0 on \partial\Omega
  n \cdot \grad mu = 0 on \partial\Omega

f = 100 * c^2 * (1-c)^2
lambda = 1e-2
M = 1.0
"""

###################################
# Create or read mesh
###################################
Lx = Ly = 200.0
Nx = Ny = 100

msh = mesh.create_rectangle(comm = MPI.COMM_WORLD,
                            points = ((0.0, 0.0), (Lx, Ly)), n = (Nx, Ny),
                            cell_type = mesh.CellType.triangle,
                            )

###################################
# Model Setup - need
#   dt, w, w0, F, J, bcs
###################################
dt = Constant(msh, 1e-1)

# parameters
c_alpha = Constant(msh, 0.3)
c_beta = Constant(msh, 0.7)
rho_s = Constant(msh, 5.0)
kappa = Constant(msh, 2.0)
M = Constant(msh, 5.0)

# FEM setup
P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
W  = FunctionSpace(msh, P1 * P1)
x  = ufl.SpatialCoordinate(msh)

w  = Function(W)
w0 = Function(W)
w_ = ufl.TestFunction(W)
dw = ufl.TestFunction(W)

c , mu  = w.sub(0),  w.sub(1)
c0, mu0 = w0.sub(0), w0.sub(1)
c_, mu_ = w_
dc, dmu = dw

# Initial conditions
ic_c0 = 0.5
ic_epsilon = 0.05

def ic_c(x, c0 = ic_c0, epsilon = ic_epsilon):
    val = c0 + epsilon*(np.cos(0.105*x[0])*np.cos(0.11*x[1])
        +(np.cos(0.13*x[0])*np.cos(0.087*x[1]))**2
        + np.cos(0.025*x[0] - 0.15*x[1])*np.cos(0.07*x[0] - 0.02*x[1]))

    return val

ic_mu = lambda x: np.zeros(x.shape[1])

w.x.array[:] = 0
c.interpolate(ic_c)
c.interpolate(ic_mu)

w0.interpolate(w)

w.x.scatter_forward()
w0.x.scatter_forward()

# Free Energy
_c = variable(c)
f_chem = rho_s * (_c - c_alpha)**2 * (c_beta - _c)**2
dfdc = diff(f_chem, _c)

F = cahn_hilliard_weak_form(w[0], w[1], w_[0], w_[1], w0[0], dt, M, kappa, dfdc)
J = derivative(F, w, dw)
bcs = [] # noflux bc

###################################
# Nonlinear solver setup
###################################

#dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)

problem = NonlinearProblem(F, w, bcs)#, J = J)
solver = NewtonSolver(MPI.COMM_WORLD, problem)

# https://fenicsproject.discourse.group/t/snes-solver-fails-when-using-the-line-search-in-fenicsx/7505/9
#solver = SNES(MPI.COMM_WORLD, problem)

solver.convergence_criterion = "incremental" # 'residual'
solver.rtol = 1e-6

solver.report = True
solver.error_on_nonconvergence = False

ksp  = solver.krylov_solver
opts = PETSc.Options()
pfx  = ksp.getOptionsPrefix()
opts[f"{pfx}ksp_type"] = "preonly"
opts[f"{pfx}pc_type"]  = "lu"
opts[f"{pfx}pc_factor_mat_solver_type"]  = "mumps"
ksp.setFromOptions()

#petsc_options = {
#    "ksp_type" : "preonly",
#    "pc_type" : "lu",
#    }

#nlparams['report'] = True
#nlparams['absolute_tolerance'] = 1e-6
#nlparams['maximum_iterations'] = 10
#
##
##nlparams['line_search'] = 'bt'      # WORKS
##nlparams['line_search'] = 'cp'      # WORKS
#nlparams['line_search'] = 'basic'    # WORKS
##nlparams['line_search'] = 'nleqerr' # WORKS
##nlparams['line_search'] = 'l2'      # FAILS
#
## 
#nlparams['linear_solver'] = 'gmres'
#nlparams['preconditioner'] = 'sor'
#
##nlparams['linear_solver'] = 'gmres'
##nlparams['linear_solver'] = 'bicgstab'
##nlparams['linear_solver'] = 'minres'
#
##nlparams['preconditioner'] = 'none'
##nlparams['preconditioner'] = 'sor'
##nlparams['preconditioner'] = 'petsc_amg'
##nlparams['preconditioner'] = 'hypre_amg'
#
#nlparams['krylov_solver']['maximum_iterations'] = 1000
##nlparams['krylov_solver']['monitor_convergence'] = True
#

###################################
# analysis setup
###################################
#file = io.XDMFFile(MPI.COMM_WORLD, "bench1/out.xdmf", "w")
#file.write_mesh(msh)
#
#file.write_function(c, t)

def total_solute(c):
    val = c * dx
    return fem.assemble_scalar(fem.form(val))

def total_free_energy(f_chem, kappa, c):
    val = f_chem*dx + kappa/2.0*inner(grad(c), grad(c))*dx
    return fem.assemble_scalar(fem.form(val))

###################################
# time integration
###################################

t = Constant(msh, 0.0)
tprev = 0.0

benchmark_output = []
end_time = Constant(msh, 1e3) # 1e6
iteration_count = 0
dt_min = 1e-2
dt.value = 1e-1

t1 = time.time()

while float(t) < float(end_time):

    tprev = float(t)

    iteration_count += 1
    if MPI.COMM_WORLD.rank == 0:
        print(f'Iteration #{iteration_count}. Time: {float(t)}, dt: {float(dt)}')
    else:
        pass

    # set IC
    w0.interpolate(w)

    # solve
    t.value = tprev + float(dt)
    niters, converged = solver.solve(w)

    while not converged:
        if float(dt) < dt_min + 1E-8:
            if MPI.COMM_WORLD.rank == 0:
                print("dt too small. exiting.")
            #postprocess()
            exit()

        dt.value = max(0.5*float(dt), dt_min)
        t.value = tprev + float(dt)
        w.interpolate(w0)

        if MPI.COMM_WORLD.rank == 0:
            print(f'REPEATING Iteration #{iteration_count}. Time: {float(t)}, dt: {float(dt)}')
        niters, converged = solver.solve(w)

    # Simple rule for adaptive timestepping
    if (niters < 5):
        dt.value = 2 * float(dt)
    else:
        dt.value = max(0.5*float(dt), dt_min)

    ############
    # Analysis
    ############
    c, mu = w.sub(0), w.sub(1)

    #file.write_function(c, t)

    F_total = total_free_energy(f_chem, kappa, c)
    C_total = total_solute(c)
    benchmark_output.append([float(t), F_total, C_total])

t2 = time.time()
spent_time = t2 - t1
if MPI.COMM_WORLD.rank == 0:
    print(f'Time spent is {spent_time}')
else:
    pass

#file.close()
####################################
## post process
####################################
#if df.MPI.rank(mesh.mpi_comm()) == 0:
#    np.savetxt('results/bench1' + '_out.csv',
#            np.array(benchmark_output),
#            fmt='%1.10f',
#            header="time,total_free_energy,total_solute",
#            delimiter=',',
#            comments=''
#            )
#else:
#    pass
