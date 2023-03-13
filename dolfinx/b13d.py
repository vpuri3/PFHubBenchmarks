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

# vis
from dolfinx import io

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
# Mesh
###################################
Lx = 100.0
Ly = 100.0
Lz = 100.0

Nx = 50
Ny = 50
Nz = 50

#msh = mesh.create_rectangle(comm = MPI.COMM_WORLD,
#                            points = ((0.0, 0.0), (Lx, Ly)), n = (Nx, Ny),
#                            cell_type = mesh.CellType.triangle,
#                            #cell_type = mesh.CellType.quadrilateral,
#                            #diagonal = mesh.DiagonalType.crossed
#
#                            #ghost_mode = mesh.GhostMode.none
#                            )

msh = mesh.create_box(comm = MPI.COMM_WORLD,
                      points = ((0.0, 0.0, 0.0), (Lx, Ly, Lz)), n = (Nx, Ny, Nz),
                      #cell_type = mesh.CellType.hexahedron,
                      cell_type = mesh.CellType.tetrahedron,
                     )
###################################
# Function Space
###################################
P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
W  = FunctionSpace(msh, P1 * P1)
x  = ufl.SpatialCoordinate(msh)

w  = Function(W)
w0 = Function(W)
w_ = TestFunction(W)

c , mu  = w.sub(0),  w.sub(1)
#c0, mu0 = w0.sub(0), w0.sub(1)
#c_, mu_ = w_

###################################
# Initial conditions
###################################
ic_c0 = 0.5
ic_epsilon = 0.05

def ic_c(x, c0 = ic_c0, epsilon = ic_epsilon):
    val = c0 + epsilon*(
          np.cos(0.105*x[0])*np.cos(0.11*x[1])
        +(np.cos(0.13 *x[0])*np.cos(0.087*x[1]))**2
        + np.cos(0.025*x[0] - 0.15*x[1])*np.cos(0.07*x[0] - 0.02*x[1]))

    return val

ic_mu = lambda x: np.zeros(x.shape[1])

w.x.array[:] = 0
c.interpolate(ic_c)
mu.interpolate(ic_mu)

w0.interpolate(w)

w.x.scatter_forward()
w0.x.scatter_forward()

###################################
# Model Setup - return
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

# weak form
_c = variable(c)
f_chem = rho_s * (_c - c_alpha)**2 * (c_beta - _c)**2
dfdc = diff(f_chem, _c)

###
def cahn_hilliard(c, mu, c_, mu_, c0, dt, M, kappa, dfdc):

    # """
    # d/dt c  = div(M * grad(\mu))
    #     \mu = F'(c) - \kappa * lapl(c)
    # """

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

    return F

F = cahn_hilliard(w[0], w[1], w_[0], w_[1], w0[0], dt, M, kappa, dfdc)

bcs = [] # noflux bc

###################################
# Nonlinear solver setup
###################################

nl = "snes"
#nl = "newton"

# KSP  opts: https://petsc.org/release/docs/manualpages/KSP/KSPType/
# PC   opts: https://petsc.org/main/overview/linear_solve_table/
# SNES opts: https://petsc.org/release/docs/manualpages/PC/PCType/

if nl == "snes":
    problem = pfbase.SnesPDEProblem(F, w, bcs)

    snes = PETSc.SNES().create()
    snes.setFunction(problem.F, problem.vector())
    snes.setJacobian(problem.J, problem.matrix())
    snes.setTolerances(atol = 1e-6, rtol = 1e-10, max_it = 20) # rtol=1e-6

    opts = PETSc.Options()

    opts[f"snes_linesearch_type"] = "bt" # "bt" "cp" "basic" "nleqerr" "l2"
    opts[f"snes_converged_reason"] = None
    #opts[f"snes_view"] = None
    opts[f"snes_monitor"] = None
    #opts[f"snes_linesearch_monitor"] = None

    ksp = snes.getKSP()

    #opts[f"ksp_monitor"] = None
    opts[f"ksp_type"] = "gmres" # "gmres", "cg", "bicgstab", "minres"
    #opts[f"ksp_gmres_restart"] = 100
    opts[f"pc_type"]  = "sor" # "sor", "gamg", "asm", "hypre", "jacobi", "bjacobi"
    opts[f"pc_type"]  = "jacobi" # "gamg",
    #ksp.setTolerances(atol = 1e-6, rtol = 1e-10, max_it = int(Nx * Ny * Nz / 10))
    ksp.setTolerances(max_it = int(Nx * Ny * Nz / 10))
    
    #ksp.setFromOptions()
    snes.setFromOptions()

elif nl == "newton":
    problem = pfbase.NewtonPDEProblem(F, w, bcs)
    
    newton = dolfinx.cpp.nls.petsc.NewtonSolver(MPI.COMM_WORLD)
    newton.setF(problem.F, problem.vector())
    newton.setJ(problem.J, problem.matrix())
    newton.set_form(problem.form)
    
    #problem = NonlinearProblem(F, w, bcs)
    #newton = NewtonSolver(MPI.COMM_WORLD, problem)

    newton.report = True
    newton.convergence_criterion = "residual" # "incremental" 'residual'
    newton.error_on_nonconvergence = False
    newton.max_it = 20
    newton.atol = 1e-6
    
    opts = PETSc.Options()
    
    ksp = newton.krylov_solver
    #ksp.setTolerances(atol = 1e-6, max_it = 1000)
    ksp_pfx = ksp.getOptionsPrefix()
    
    opts[f"{ksp_pfx}ksp_type"] = "gmres"
    #opts[f"{ksp_pfx}ksp_monitor"] = None
    #opts[f"{ksp_pfx}ksp_gmres_restart"] = 100
    opts[f"{ksp_pfx}pc_type"]  = "sor"
    newton.relaxation_parameter = 1.0
    opts[f"{ksp_pfx}pc_type"]  = "sor"
    
    ksp.setFromOptions()

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
end_time = Constant(msh, 1e2) # 1e6
iteration_count = 0
dt_min = 1e-2
dt.value = 1e-1

t1 = time.time()

while float(t) < float(end_time):

    tprev = float(t)

    iteration_count += 1
    if MPI.COMM_WORLD.rank == 0:
        print(f'#======================#')
        print(f'Iteration #{iteration_count}. Time: {float(t)}, dt: {float(dt)}')

    # set IC
    w0.x.array[:] = w.x.array[:]

    # solve
    t.value = tprev + float(dt)

    if nl == "snes":
        snes.solve(None, w.vector)
        converged, niters = snes.converged, snes.getIterationNumber()
        resid = snes.getFunctionNorm()
    elif nl == "newton":
        niters, converged = newton.solve(w.vector)

    while not converged:
        if float(dt) < dt_min + 1E-8:
            if MPI.COMM_WORLD.rank == 0:
                print("dt too small. exiting.")
            #postprocess()
            exit()

        dt.value = max(0.5 * float(dt), dt_min)
        t.value  = tprev + float(dt)
        w.x.array[:] = w0.x.array[:]

        if MPI.COMM_WORLD.rank == 0:
            print(f'REPEATING Iteration #{iteration_count}. Time: {float(t)}, dt: {float(dt)}')
        if nl == "snes":
            snes.solve(None, w.vector)
            niters, converged = snes.getIterationNumber(), snes.converged
            resid = snes.getFunctionNorm()
        elif nl == "newton":
            niters, converged = newton.solve(w.vector)

    # assert residual is less than tolerance
    #if MPI.COMM_WORLD.rank == 0:
        #print("Residual: ", resid)

    # Simple rule for adaptive timestepping
    if (niters < 5):
        dt.value = 2 * float(dt)
    elif (niters > 10):
        dt.value = max(0.5*float(dt), dt_min)

    ############
    # Analysis
    ############
    c, mu = w.sub(0), w.sub(1)

    file.write_function(c, t)

    F_total = total_free_energy(f_chem, kappa, c)
    C_total = total_solute(c)
    benchmark_output.append([float(t), F_total, C_total])

    if MPI.COMM_WORLD.rank == 0:
        print("C_total: ", C_total, "TFE, ", F_total)

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
