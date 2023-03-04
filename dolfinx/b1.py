#
import numpy as np
import time, os, shutil

import ufl
from ufl import TestFunction
from ufl import ds, dx, grad, inner, dot, variable, diff, derivative
from ufl import sin, cos, tan, log, exp, pi

import dolfinx
from dolfinx import fem, mesh
from dolfinx.fem import form, Function, FunctionSpace, Constant

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

# vis
from dolfinx import io, plot

# local
import pfbase

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
                            #cell_type = mesh.CellType.quadrilateral,
                            )

###################################
# Model Setup - need
#   t, dt, w, w0, F, bcs
###################################

t  = Constant(msh, 0.0)
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
w_ = TestFunction(W)

c , mu  = w.sub(0),  w.sub(1)
c0, mu0 = w0.sub(0), w0.sub(1)
c_, mu_ = w_

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
mu.interpolate(ic_mu)

w0.interpolate(w)

w.x.scatter_forward()
w0.x.scatter_forward()

# weak form
_c = variable(c)
f_chem = rho_s * (_c - c_alpha)**2 * (c_beta - _c)**2
dfdc = diff(f_chem, _c)

F = cahn_hilliard_weak_form(w[0], w[1], w_[0], w_[1], w0[0], dt, M, kappa, dfdc)
#F = diffusion_weak_form()

bcs = [] # noflux bc

###################################
# Nonlinear solver setup
###################################

problem = problem_types.SnesPDEProblem(F, w, bcs)

solver = PETSc.SNES().create()
#solver.setType("vinewtonrsls")
solver.setFunction(problem.F, problem.vector())
#solver.setObjective(problem.F) # needed for line search ??
solver.setJacobian(problem.J, problem.matrix())
solver.setTolerances(rtol = 1e-6, atol = 1e-6, max_it = 20)

# KSP  opts: https://petsc.org/release/docs/manualpages/KSP/KSPType/
# SNES opts: https://petsc.org/release/docs/manualpages/SNES/SNESLineSearchType/

opts = PETSc.Options()

snes_pfx = solver.prefix
opts[f"{snes_pfx}snes_linesearch_type"] = "basic" # "bt" "cp" "basic" "nleqerr" "l2"
opts[f"{snes_pfx}snes_monitor"] = None
opts[f"{snes_pfx}snes_linesearch_monitor"] = None

solver.setFromOptions()

ksp = solver.getKSP()

ksp_pfx = ksp.prefix
opts[f"{ksp_pfx}ksp_monitor"] = None
opts[f"{ksp_pfx}ksp_type"] = "gmres" # "gmres", "cg", "bicgstab", "minres"
opts[f"{ksp_pfx}pc_type"]  = "sor"
ksp.setTolerances(rtol = 1e-6, max_it = int(Nx * Ny / 10))

ksp.setFromOptions()

pc = ksp.getPC()
pc.setType("sor") # "none", "lu", "sor", "petsc_amg", "hypre_amg"

#solver.report = True
#solver.error_on_nonconvergence = False
#solver.relaxation_parameter = 1.0 # default 1.0
#solver.convergence_criterion = "residual" # "residual", "incremental"
##nlparams['krylov_solver']['monitor_convergence'] = True

###################################
# analysis setup
###################################
if os.path.exists("out_bench1"):
    if MPI.COMM_WORLD.rank == 0:
        shutil.rmtree("out_bench1")
file = io.XDMFFile(MPI.COMM_WORLD, "out_bench1/bench1.xdmf", "w")
file.write_mesh(msh)

file.write_function(c, t)

def total_solute(c):
    frm = c * dx
    val = fem.assemble_scalar(form(frm))
    return MPI.COMM_WORLD.allreduce(val, op=MPI.SUM)

def total_free_energy(f_chem, kappa, c):
    frm = f_chem * dx + kappa / 2.0 * inner(grad(c), grad(c)) * dx
    val = fem.assemble_scalar(form(frm))
    return MPI.COMM_WORLD.allreduce(val, op=MPI.SUM)

###################################
# time integration
###################################

tprev = 0.0

benchmark_output = []
end_time = Constant(msh, 5e1) # 1e6
iteration_count = 0
dt_min = 1e-2
dt.value = 1e-1

t1 = time.time()

while float(t) < float(end_time):

    tprev = float(t)

    iteration_count += 1
    if MPI.COMM_WORLD.rank == 0:
        print(f'Iteration #{iteration_count}. Time: {float(t)}, dt: {float(dt)}')

    # set IC
    w0.interpolate(w)

    # solve
    t.value = tprev + float(dt)

    solver.solve(None, w.vector)
    niters, converged = solver.getIterationNumber(), solver.converged

    while not converged:
        if float(dt) < dt_min + 1E-8:
            if MPI.COMM_WORLD.rank == 0:
                print("dt too small. exiting.")
            #postprocess()
            exit()

        dt.value = max(0.5 * float(dt), dt_min)
        t.value  = tprev + float(dt)
        w.interpolate(w0)

        if MPI.COMM_WORLD.rank == 0:
            print(f'REPEATING Iteration #{iteration_count}. Time: {float(t)}, dt: {float(dt)}')
        solver.solve(None, w.vector)
        niters, converged = solver.getIterationNumber(), solver.converged

    # Simple rule for adaptive timestepping
    if (niters < 10):
        dt.value = 2 * float(dt)
    else:
        dt.value = max(0.5*float(dt), dt_min)

    #if MPI.COMM_WORLD.rank == 0:
        #print("Converged in ", niters, "SNES iterations")

    ############
    # Analysis
    ############
    c, mu = w.sub(0), w.sub(1)

    file.write_function(c, t)

    F_total = total_free_energy(f_chem, kappa, c)
    C_total = total_solute(c)
    benchmark_output.append([float(t), F_total, C_total])

    if MPI.COMM_WORLD.rank == 0:
        print("Total solute: ", C_total, "TFE, ", F_total)

# end time loop

t2 = time.time()
spent_time = t2 - t1
if MPI.COMM_WORLD.rank == 0:
    print(f'Time spent is {spent_time}')
else:
    pass

file.close()
####################################
## post process
####################################
if MPI.COMM_WORLD.rank == 0:
    np.savetxt('out_bench1/bench1' + '_out.csv',
            np.array(benchmark_output),
            fmt='%1.10f',
            header="time,total_free_energy,total_solute",
            delimiter=',',
            comments=''
            )
#
