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
#   dt, w_ic, w, w0, F, J, bcs
###################################
dt = df.Constant(1e-1)

# parameters
c_alpha = df.Constant(0.3)
c_beta = df.Constant(0.7)
rho_s = df.Constant(5.0)
kappa = df.Constant(2.0)
M = df.Constant(5.0)

# FEM setup
P = df.FunctionSpace(mesh, 'P', 1)
Pelem = P.ufl_element()
W = df.FunctionSpace(mesh,  df.MixedElement([Pelem, Pelem]))

w  = df.Function(W)
dw = df.TrialFunction(W)
w_ = df.TestFunction(W)

# Initial conditions
epsilon = 0.05
c0 = 0.5

w0 = df.Function(W)
w_ic = InitialConditionsBench1(c0, epsilon, degree=2)
w0.interpolate(w_ic)

w_ = df.TestFunction(W)
dw = df.TrialFunction(W)

# Free Energy
c , mu  = df.split(w)
#c_, mu_ = df.split(w_)
#c0, _  = df.split(w0)

c = variable(c)
f_chem = rho_s * (c - c_alpha)**2 * (c_beta - c)**2
dfdc = df.diff(f_chem, c)

F = cahn_hilliard_weak_form(w[0], w[1], w_[0], w_[1], w0[0], dt, M, kappa, dfdc)
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
#nlparams['line_search'] = 'bt'      # WORKS
#nlparams['line_search'] = 'cp'      # WORKS
nlparams['line_search'] = 'basic'    # WORKS
#nlparams['line_search'] = 'nleqerr' # WORKS
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
    filename = "results/bench1/conc"
    #cfile = df.XDMFFile(filename + ".xdmf")
    #for f in [cfile, ]:
    #    f.parameters['flush_output'] = True
    #    f.parameters['rewrite_function_mesh'] = False

    #cfile.write(w.sub(0), 0.0)

    cfile = df.File(filename + ".pvd", "compressed")
    #cfile << mesh

def total_solute(c):
    return df.assemble(c * dx)

def total_free_energy(f_chem, kappa, c):
    return df.assemble(f_chem*dx + kappa/2.0*inner(grad(c), grad(c))*dx)

###################################
# time integration
###################################

# Ensure everything is reset
t = df.Constant(0.0)
tprev = 0.0
w.interpolate(w_ic)
w0.interpolate(w_ic)

benchmark_output = []
end_time = df.Constant(1e3) # 1e6
iteration_count = 0
dt_min = 1e-2
dt.assign(1e-1)

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
    c, _ = w.split()

    if save_solution:
        cfile << (c, t)
        #cfile.write(c, float(t))

    F_total = total_free_energy(f_chem, kappa, c)
    C_total = total_solute(c)
    benchmark_output.append([float(t), F_total, C_total])

    if df.MPI.rank(mesh.mpi_comm()) == 0:
        print("C_total: ", C_total, "TFE: ", F_total)

t2 = time.time()
spent_time = t2 - t1
if df.MPI.rank(mesh.mpi_comm()) == 0:
    print(f'Time spent is {spent_time}')
else:
    pass

###################################
# post process
###################################
if df.MPI.rank(mesh.mpi_comm()) == 0:
    np.savetxt('results/bench1' + '_out.csv',
            np.array(benchmark_output),
            fmt='%1.10f',
            header="time,total_free_energy,total_solute",
            delimiter=',',
            comments=''
            )
else:
    pass
