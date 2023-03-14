#
import dolfin as df
import numpy as np
import time

from pfbase import *
from ufl import split, dx, inner, grad, variable, diff

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
Lx = Ly = 100.0
Nx = Ny = 100
mesh = df.RectangleMesh(df.Point(0.0, 0.0), df.Point(Lx, Ly), Nx, Ny, 'crossed')

###################################
# Model Setup - need
#   dt, w, w0, F, J, bcs
###################################
dt = df.Constant(1e-1)

# parameters
c_alpha = df.Constant(0.3)
c_beta = df.Constant(0.7)
kappa = df.Constant(2.0)
rho = df.Constant(5.0)
M = df.Constant(5.0)
k = df.Constant(0.09)
epsilon = df.Constant(90.0)

# FEM setup
P1 = df.FunctionSpace(mesh, 'P', 1)
PE = P1.ufl_element()
ME = [PE, PE, PE]
ME = df.MixedElement(ME)
W  = df.FunctionSpace(mesh,  ME)

w  = df.Function(W)
dw = df.TrialFunction(W)
w_ = df.TestFunction(W)

# Initial conditions
cc0 = 0.5
cc1 = 0.04

w0 = df.Function(W)
w_ic = InitialConditionsBench6(cc0, cc1, degree=2)
w0.interpolate(w_ic)

# Free energy functional
c, _, phi = df.split(w)
c   = df.variable(c)
phi = df.variable(phi)

f_chem = rho * (c - c_alpha)**2 * (c_beta - c)**2
f_elec = k * c * phi / 2.0

dfdc = df.diff(f_chem, c) + k * phi

## weak form
Fc = cahn_hilliard_weak_form(w[0], w[1], w_[0], w_[1], w0[0], dt, M, kappa, dfdc)
Fp = poisson_weak_form(w[2], w_[2], -k * c / epsilon, df.Constant(1.0))

F= Fc + Fp

# BC
tol = 1E-12
def boundary_left(x, on_boundary):
    return on_boundary and df.near(x[0], 0, tol)

def boundary_right(x, on_boundary):
    return on_boundary and df.near(x[0], Lx, tol)

phi_right = df.Expression(("sin(x[1]/7)"), degree=2)

_, _, Wphi = W.split()
bc_left  = df.DirichletBC(Wphi, df.Constant(0.0), boundary_left)
bc_right = df.DirichletBC(Wphi, phi_right, boundary_right)

bcs = [bc_left, bc_right] # no-flux on top, bottom boundary

###############
J = df.derivative(F, w, dw)

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
# bactracig (bt) diverges with only Laplace eqn
#nlparams['line_search'] = 'bt'      # WORKS (7s) for np=32, T=3.0
nlparams['line_search'] = 'cp'       # (8s) #
#nlparams['line_search'] = 'basic'   # (7s)
#nlparams['line_search'] = 'nleqerr' # (15s)
#nlparams['line_search'] = 'l2'      # FAILING

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

nlparams['krylov_solver']['maximum_iterations'] = 5000
#nlparams['krylov_solver']['monitor_convergence'] = True

###################################
# analysis setup
###################################
if save_solution:
    dirname = "results/bench6/"
    filename0 = dirname + "conc"
    filename1 = dirname + "phi"
    #cfile = df.XDMFFile(filename + ".xdmf")
    #for f in [cfile, ]:
    #    f.parameters['flush_output'] = True
    #    f.parameters['rewrite_function_mesh'] = False

    #cfile.write(w.sub(0), 0.0)

    file0 = df.File(filename0 + ".pvd")
    file1 = df.File(filename1 + ".pvd")

def total_solute(c):
    return df.assemble(c * dx)

def total_free_energy(f_chem, f_elec, kappa):
    E = df.assemble((
        f_chem +
        f_elec +
        kappa / 2.0 * inner(grad(c), grad(c))
        )*dx)

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
end_time = df.Constant(3) # 400.0
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
    c, _, phi = w.split()

    if save_solution:
        file0 << (c, t)
        file1 << (phi, t)
        #cfile.write(c, float(t))

    F_total = total_free_energy(f_chem, f_elec, kappa)
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
if df.MPI.rank(mesh.mpi_comm()) == 0:
    np.savetxt('results/bench6' + '_out.csv',
            np.array(benchmark_output),
            fmt='%1.10f',
            header="time,total_free_energy,total_solute",
            delimiter=',',
            comments=''
            )
else:
