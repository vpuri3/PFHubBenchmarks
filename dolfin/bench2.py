#
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import time

from pfbase import *

save_solution = True

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

NUM_ETA = 4

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
#W  = df.FunctionSpace(mesh,  ME, constrained_domain=PeriodicBoundary(Lx, Ly))

w  = df.Function(W)
dw = df.TrialFunction(W)
w_ = df.TestFunction(W)

# Initial conditions
c0 = 0.5
epsilon = 0.05
epsilon_eta = 0.1
psi = 1.5

w0 = df.Function(W)
w_ic = InitialConditionsBench2(c0, epsilon, epsilon_eta, psi, degree=2)
w0.interpolate(w_ic)

# Free energy functional
c, _, eta1, eta2, eta3, eta4 = df.split(w)
c    = df.variable(c)
eta1 = df.variable(eta1)
eta2 = df.variable(eta2)
eta3 = df.variable(eta3)
eta4 = df.variable(eta4)

def double_well(u1, u2, u3, u4, alpha):
    W = (u1**2 * (1 - u1)**2 +
         u2**2 * (1 - u2)**2 +
         u3**2 * (1 - u3)**2 +
         u4**2 * (1 - u4)**2)

    W += alpha * (
            u1**2 * u2**2 + u1**2 * u3**2 + u1**2 * u4**2 +
            u2**2 * u3**2 + u2**2 * u4**2 +
            u3**2 * u4**2)

    return W

def hinterp(u1, u2, u3, u4):
    return (u1**3 * (6*u1**2 - 15*u1 + 10) +
            u2**3 * (6*u2**2 - 15*u2 + 10) +
            u3**3 * (6*u3**2 - 15*u3 + 10) +
            u4**3 * (6*u4**2 - 15*u4 + 10))

f_alpha = rho**2 * (c - c_alpha)**2
f_beta  = rho**2 * (c - c_beta)**2
f_chem  = (f_alpha * (1 - hinterp(eta1, eta2, eta3, eta4)) +
           f_beta  * hinterp(eta1, eta2, eta3, eta4) +
           ww * double_well(eta1, eta2, eta3, eta4, alpha))

dfdc  = df.diff(f_chem, c)
dfde1 = df.diff(f_chem, eta1)
dfde2 = df.diff(f_chem, eta2)
dfde3 = df.diff(f_chem, eta3)
dfde4 = df.diff(f_chem, eta4)

Fc  = cahn_hilliard_weak_form(w[0], w[1], w_[0], w_[1], w0[0], dt, M, kappa_c, dfdc)
Fe1 = allen_cahn_weak_form(w[2], w_[2], w0[2], dt, L, kappa_eta, dfde1, df.Constant(0))
Fe2 = allen_cahn_weak_form(w[3], w_[3], w0[3], dt, L, kappa_eta, dfde2, df.Constant(0))
Fe3 = allen_cahn_weak_form(w[4], w_[4], w0[4], dt, L, kappa_eta, dfde3, df.Constant(0))
Fe4 = allen_cahn_weak_form(w[5], w_[5], w0[5], dt, L, kappa_eta, dfde4, df.Constant(0))

F = Fc + Fe1 + Fe2 + Fe3 + Fe4

###############
J = df.derivative(F, w, dw)
bcs = [] # noflux bc

###################################
# Nonlinear solver setup
###################################
df.set_log_level(df.LogLevel.ERROR)

problem = df.NonlinearVariationalProblem(F, w, bcs, J)
solver  = df.NonlinearVariationalSolver(problem)

#solver.parameters['nonlinear_solver'] = 'newton'
#nlparams  = solver.parameters['newton_solver']

solver.parameters['nonlinear_solver'] = 'snes'
nlparams  = solver.parameters['snes_solver']

nlparams['report'] = True
nlparams['error_on_nonconvergence'] = False
nlparams['absolute_tolerance'] = 1e-6
nlparams['maximum_iterations'] = 10

#
#nlparams['line_search'] = 'bt'      # WORKS (29s for end_time=100, 32 cores)
nlparams['line_search'] = 'cp'       # FABULOUS (22s)
#nlparams['line_search'] = 'basic'   # WORKS (24s)
#nlparams['line_search'] = 'nleqerr' # WORKS (42s)
#nlparams['line_search'] = 'l2'      # FAILS

# 
nlparams['linear_solver'] = 'gmres'
nlparams['preconditioner'] = 'sor'

#nlparams['linear_solver'] = 'gmres'
#nlparams['linear_solver'] = 'bicgstab'
#nlparams['linear_solver'] = 'minres'

#nlparams['preconditioner'] = 'none'
#nlparams['preconditioner'] = 'sor'
#nlparams['preconditioner'] = 'petsc_amg'
#nlparams['preconditioner'] = 'hypre_amg'

nlparams['krylov_solver']['maximum_iterations'] = 1000
#nlparams['krylov_solver']['monitor_convergence'] = True

###################################
# analysis setup
###################################
if save_solution:
    dirname = "results/bench2/"
    filename0 = dirname + "conc"
    filename1 = dirname + "eta1"
    filename2 = dirname + "eta2"
    filename3 = dirname + "eta3"
    filename4 = dirname + "eta4"
    #cfile = df.XDMFFile(filename + ".xdmf")
    #for f in [cfile, ]:
    #    f.parameters['flush_output'] = True
    #    f.parameters['rewrite_function_mesh'] = False

    #cfile.write(w.sub(0), 0.0)

    file0 = df.File(filename0 + ".pvd")
    file1 = df.File(filename1 + ".pvd")
    file2 = df.File(filename2 + ".pvd")
    file3 = df.File(filename3 + ".pvd")
    file4 = df.File(filename4 + ".pvd")

def total_solute(c):
    return df.assemble(c * dx)

def total_free_energy(f_chem, kappa_c, kappa_eta, w):
    E = df.assemble(f_chem*dx + kappa_c/2.0*inner(grad(c), grad(c))*dx)

    for i in range(NUM_ETA):
        eta = w[2+i]
        E += df.assemble(kappa_eta/2.0 * inner(grad(eta), grad(eta)) * dx)

    return E

def postprocess():
    if df.MPI.rank(mesh.mpi_comm()) == 0:
        np.savetxt('results/bench2' + '_out.csv',
                np.array(benchmark_output),
                fmt='%1.10f',
                header="time,total_free_energy,total_solute",
                delimiter=',',
                comments=''
                )
    else:
        pass
    return

###################################
# time integration
###################################

# Ensure everything is reset
t = df.Constant(0.0)
tprev = 0.0
w.interpolate(w_ic)
w0.interpolate(w_ic)

benchmark_output = []
end_time = df.Constant(100) # 1E6
iteration_count = 0
dt_min = 1e-4
dt.assign(1e-2)
t1 = time.time()

while float(t) < float(end_time) + df.DOLFIN_EPS:

    tprev = float(t)

    iteration_count += 1
    if df.MPI.rank(mesh.mpi_comm()) == 0:
        print(f'Iteration #{iteration_count}. Time: {float(t)}, dt: {float(dt)}')
    else:
        pass

    # set IC
    w0.assign(w)

    # solve
    t.assign(tprev + float(dt))
    niters, converged = solver.solve()

    while not converged:
        #if float(dt) < dt_min + 1E-8:
        #    if df.MPI.rank(mesh.mpi_comm()) == 0:
        #        print("dt too small. exiting.")
        #    postprocess()
        #    exit()

        dt.assign(max(0.5*float(dt), dt_min))
        t.assign(tprev + float(dt))
        w.assign(w0)

        if df.MPI.rank(mesh.mpi_comm()) == 0:
            print(f'REPEATING Iteration #{iteration_count}. Time: {float(t)}, dt: {float(dt)}')
        niters, converged = solver.solve()

    # Simple rule for adaptive timestepping
    if (niters < 5):
        dt.assign(2*float(dt))
    else:
        dt.assign(max(0.5*float(dt), dt_min))

    ############
    # Analysis
    ############
    c, _, eta1, eta2, eta3, eta4 = w.split()

    if save_solution:
        file0 << (c, t)
        file1 << (eta1, t)
        file2 << (eta2, t)
        file3 << (eta3, t)
        file4 << (eta4, t)
        #cfile.write(c, float(t))

    F_total = total_free_energy(f_chem, kappa_c, kappa_eta, w)
    C_total = total_solute(c)
    benchmark_output.append([float(t), F_total, C_total])

t2 = time.time()
spent_time = t2 - t1
if df.MPI.rank(mesh.mpi_comm()) == 0:
    print(f'Time spent is {spent_time}')
else:
    pass

###################################
# post process
###################################
postprocess()
