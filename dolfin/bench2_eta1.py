#
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import time

from pfbase import *

save_solution = False
case = 'b'

###################################
# Optimization options for the finite element form compiler
###################################
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math'
df.parameters["form_compiler"]["quadrature_degree"] = 3

###################################
# Create or read mesh
###################################
Lx = Ly = 200.0
Nx = Ny = 100
mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly), Nx, Ny, 'crossed')

###################################
# Model Setup - need
#   dt, w, w0, F, J, bcs
###################################
dt = df.Constant(1e-1)

NUM_ETA = 1

# parameters
c_alpha = df.Constant(0.3)
c_beta = df.Constant(0.7)
rho = df.Constant(np.sqrt(2.0))
kappa_c = df.Constant(3.0)
kappa_eta = df.Constant(3.0)
M = df.Constant(5.0)
ww = df.Constant(1.0)
alpha = df.Constant(5.0)
L = df.Constant(5.0)

# FEM setup
P1 = df.FunctionSpace(mesh, 'P', 1)
PE = P1.ufl_element()
ME = [PE, PE]
for i in range(NUM_ETA):
    ME.append(PE)
ME = df.MixedElement(ME)
W  = df.FunctionSpace(mesh,  ME)

w  = df.Function(W)
dw = df.TrialFunction(W)
w_ = df.TestFunction(W)

# Initial conditions
c0 = 0.5
epsilon = 0.05
epsilon_eta = 0.1
psi = 1.5

w0 = df.Function(W)
w_ic = ICB2_jank1(c0, epsilon, epsilon_eta, psi, degree=2)
w0.interpolate(w_ic)

# Free energy functinoal
c, mu, eta = df.split(w)

c = variable(c)
eta = variable(eta)

def hinterp(u):
    return u**3 * (6*u**2 - 15*u + 10)

def double_well(u):
    W = u**2 * (1 - u)**2
    return W

# double well important for stability!

#f_chem = rho_s * (c - c_alpha)**2 * (c_beta - c)**2 # WORKING
#f_chem = rho**2 * (c - c_alpha)**2 * (1 - eta) + rho**2 * (c - c_beta)**2 # MADE UP CHEM POTENTIAL IS BAD
#f_chem = rho**2 * (c - c_alpha)**2 * (1 - eta) + rho**2 * (c - c_beta)**2 * eta + ww * double_well(eta)
f_chem = rho**2 * (c - c_alpha)**2 * (1 - hinterp(eta)) + rho**2 * (c - c_beta)**2  * hinterp(eta) + ww * double_well(eta)

dfdc = df.diff(f_chem, c)
dfde = df.diff(f_chem, eta)

Fc = cahn_hilliard_weak_form(w[0], w[1], w_[0], w_[1], w0[0], dt, M, kappa_c, dfdc)
Fe = allen_cahn_weak_form(w[2], w_[2], w0[2], dt, L, kappa_eta, dfde)
F = Fc + Fe

###############
J = df.derivative(F, w, dw)
bcs = [] # noflux bc

###################################
# Nonlinear solver setup
###################################
problem = df.NonlinearVariationalProblem(F, w, bcs, J)
solver = df.NonlinearVariationalSolver(problem)

solver.parameters['nonlinear_solver'] = 'snes'
solver.parameters['snes_solver']['report'] = True
solver.parameters['snes_solver']['line_search'] = 'bt'
solver.parameters['snes_solver']['error_on_nonconvergence'] = False
solver.parameters['snes_solver']['absolute_tolerance'] = 1e-6
solver.parameters['snes_solver']['maximum_iterations'] = 10

#solver.parameters['snes_solver']['linear_solver'] = 'gmres'
solver.parameters['snes_solver']['linear_solver'] = 'bicgstab'

#solver.parameters['snes_solver']['preconditioner'] = 'none'
solver.parameters['snes_solver']['preconditioner'] = 'sor'
solver.parameters['snes_solver']['krylov_solver']['maximum_iterations'] = 1000

###################################
# analysis setup
###################################
if save_solution:
    cfile = df.XDMFFile("results/concentration.xdmf")
    for f in [cfile, ]:
        f.parameters['flush_output'] = True
        f.parameters['rewrite_function_mesh'] = False

    cfile.write(w.sub(0), 0.0)

def total_free_energy(f_chem, kappa_c, kappa_eta, w):
    E = df.assemble(f_chem*dx + kappa_c/2.0*inner(grad(c), grad(c))*dx)

    for i in range(NUM_ETA):
        eta = w[2+i]
        E += df.assemble(kappa_eta/2.0 * inner(grad(eta), grad(eta)) * dx)

    return E

###################################
# time integration
###################################

# Ensure everything is reset
t = df.Constant(0.0)
tprev = 0.0
w.interpolate(w_ic)
w0.interpolate(w_ic)

benchmark_output = []
#end_time = df.Constant(1e6)
end_time = df.Constant(5e2)
iteration_count = 0
dt_min = 1e-2
dt.assign(1e-2)
t1 = time.time()

while float(t) < float(end_time) + df.DOLFIN_EPS:

    tprev = float(t)

    iteration_count += 1
    if df.MPI.rank(mesh.mpi_comm()) == 0:
        df.info(f'Iteration #{iteration_count}. Time: {float(t)}, dt: {float(dt)}')
    else:
        pass

    # set IC
    w0.assign(w)

    # solve
    t.assign(tprev + float(dt))
    niters, converged = solver.solve()

    while not converged:
        dt.assign(max(0.5*float(dt), dt_min))
        t.assign(tprev + float(dt))
        niters, converged = solver.solve()

    # Simple rule for adaptive timestepping
    if (niters < 5):
        dt.assign(2*float(dt))
    else:
        dt.assign(max(0.5*float(dt), dt_min))

    ############
    # Analysis
    ############

    if save_solution:
        cfile.write(c, float(t))

    F_total = total_free_energy(f_chem, kappa_c, kappa_eta, w)
    benchmark_output.append([float(t), F_total])

t2 = time.time()
spent_time = t2 - t1
if df.MPI.rank(mesh.mpi_comm()) == 0:
    df.info(f'Time spent is {spent_time}')
else:
    pass

###################################
# post process
###################################
if df.MPI.rank(mesh.mpi_comm()) == 0:
    np.savetxt('results/2' + case + '_out.csv',
            np.array(benchmark_output),
            fmt='%1.10f',
            header="time,total_free_energy",
            delimiter=',',
            comments=''
            )
else:
    pass
