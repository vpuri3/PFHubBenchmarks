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
Lx = Ly = 960.0
Nx = Ny = 350
mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly), Nx, Ny, 'crossed')

###################################
# Model Setup - need
#   dt, w, w0, F, J, bcs
###################################
dt = df.Constant(1e-1)

# parameters
_W0 = df.Constant(1.0)
m = df.Constant(4)
epsilon_m = df.Constant(0.05)
theta0 = df.Constant(0.0)
tau0 = df.Constant(1.0)
D = df.Constant(10.0)
Delta = df.Constant(-0.3) # undercooling

# FEM setup
P1 = df.FunctionSpace(mesh, 'P', 1)
PE = P1.ufl_element()
ME = [PE, PE]
ME = df.MixedElement(ME)
W  = df.FunctionSpace(mesh,  ME)

w  = df.Function(W)
dw = df.TrialFunction(W)
w_ = df.TestFunction(W)

# Initial Conditions
rIC = 8.0
wIC = 1.0
vin = 1.0
vout = -1.0

w0 = df.Function(W)
w_ic = InitialConditionsBench3(float(Delta), rIC, wIC, vin, vout, degree=2)

w.interpolate(w_ic)
w0.interpolate(w_ic)

# free energy functional
U , phi  = df.split(w)
U0, phi0 = df.split(w0)

lam = D * tau0 / (0.6267 * _W0**2)

f_chem = -1/2 * phi**2 + 1/4 * phi**4 + lam * U * phi * (
        1 - 2/3 * phi**2 + 1/5 * phi**4)

## TODO - fix presence of NaNs. all we need is the angle
g = df.grad(phi)
theta = df.atan(g[1] / g[0]) # df.atan2 - no attribute error :/
a = 1.0 #+ epsilon_m * df.cos(m * (theta - theta0))
a = 1.0 #- epsilon_m

_W  = _W0 * a
tau = tau0 * a**2

dfdp = (phi - lam * U * (1 - phi**2)) * (1 - phi**2) #- 2*epsilon_m*df.inner(g, g)

L_u  = df.Constant(1.0)
f1_u = df.Constant((0.0, 0.0))
f2_u = df.Constant(0.0)
f_p = df.Constant(0.0)

# TODO- time-derivative
ddt_phi_U = w_[0] * (phi - phi0) / dt # discr LHS
ddt_phi_U = allen_cahn_RHS_IBP(w[1], w_[0], 1/tau, _W**2, -dfdp, f_p) # discr RHS

Fu = diffusion_weak_form(w[0], w_[0], w0[0], dt, L_u, D, f1_u, f2_u)
Fu -= 0.5 * ddt_phi_U * dx

Fp = allen_cahn_weak_form(w[1], w_[1], w0[1], dt, 1/tau, _W**2, -dfdp, f_p)

F = Fu + Fp

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
#nlparams['line_search'] = 'bt'      # WORKS (40s, 128 core, T=500)
#nlparams['line_search'] = 'cp'      # WORKS (25s) - (62s) for T=1500
nlparams['line_search'] = 'basic'    # WORKS (27s) - (58s) for T=1500
#nlparams['line_search'] = 'nleqerr' # WORKS (45s)
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
    dirname = "results/bench3/"
    filename0 = dirname + "U"
    filename1 = dirname + "phi"
    #cfile = df.XDMFFile(filename + ".xdmf")
    #for f in [cfile, ]:
    #    f.parameters['flush_output'] = True
    #    f.parameters['rewrite_function_mesh'] = False

    #cfile.write(w.sub(0), 0.0)

    file0 = df.File(filename0 + ".pvd")
    file1 = df.File(filename1 + ".pvd")

def total_free_energy(phi, f_chem, _W):
    E = df.assemble(
            (f_chem + 0.5 * _W**2 * df.inner(df.grad(phi), df.grad(phi))) * dx
            )

    return E

def solid_fraction(phi):
    return df.assemble(0.5 * (phi + 1.0) * dx) / (Lx * Ly)

def postprocess():
    if df.MPI.rank(mesh.mpi_comm()) == 0:
        print("Post-Processing")
        np.savetxt('results/bench3' + '_out.csv',
                np.array(benchmark_output),
                fmt='%1.10f',
                header="time,total_free_energy,solid_fraction",
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
end_time = df.Constant(100) # 1500
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
    U, phi = w.split()

    if save_solution:
        file0 << (U, t)
        file1 << (phi, t)
        #cfile.write(c, float(t))

    F_total = total_free_energy(phi, f_chem, _W)
    S_total = solid_fraction(phi)
    benchmark_output.append([float(t), F_total, S_total])

    if F_total < 0.0:
        if df.MPI.rank(mesh.mpi_comm()) == 0:
            print("TFE: {F_total} < 0")
        postprocess()
        exit()

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
#
