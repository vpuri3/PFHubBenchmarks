#
import dolfin as df
import numpy as np
import time

from pfbase import *

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

# parameters
c_alpha = df.Constant(0.3)
c_beta = df.Constant(0.7)
rho_s = df.Constant(5.0)
kappa = df.Constant(2.0)
M = df.Constant(5.0)

# FEM setup
P = df.FunctionSpace(mesh, 'P', 1)
Pelem = P.ufl_element()
W = df.FunctionSpace(mesh, df.MixedElement([Pelem, Pelem]))

w  = df.Function(W)
w0 = df.Function(W)
w_ = df.TestFunction(W)
dw = df.TrialFunction(W)

# Initial conditions
class InitialConditionsBench1(df.UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c0 = args[0]
        self.epsilon = args[1]
    def eval(self, values, x):
        # indices: (c, mu) - (0, 1)

        values[0] = self.c0 + self.epsilon*(np.cos(0.105*x[0])*np.cos(0.11*x[1])
                +(np.cos(0.13*x[0])*np.cos(0.087*x[1]))**2
                + np.cos(0.025*x[0] - 0.15*x[1])*np.cos(0.07*x[0] - 0.02*x[1]))
        values[1] = 0.0

    def value_shape(self):
        return (2,)

ic_epsilon = 0.05
ic_c0 = 0.5
w_init = InitialConditionsBench1(ic_c0, ic_epsilon, degree=2)
w.interpolate(w_init)
w0.interpolate(w_init)

# Free Energy
c , mu  = df.split(w)

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
#nlparams = solver.parameters['newton_solver']

solver.parameters['nonlinear_solver'] = 'snes'
nlparams = solver.parameters['snes_solver']
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

###################################
# analysis setup
###################################
def total_solute(c):
    return df.assemble(c * dx)

def total_free_energy(f_chem, kappa, c):
    return df.assemble(f_chem*dx + kappa/2.0*inner(grad(c), grad(c))*dx)

filename = "out_b1"
file = HDF5File(MPI.comm_world, filename + ".h5", "w")
file.write(mesh, "mesh")
tsave = []

###################################
# time integration
###################################

# Ensure everything is reset
t = df.Constant(0.0)
tprev = 0.0
w.interpolate(w_init)

# save IC
c, mu = w.sub(0), w.sub(1)
file.write(c , "c" , float(t))
file.write(mu, "mu", float(t))
tsave.append(float(t))

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

    if df.MPI.rank(mesh.mpi_comm()) == 0:
        print("Converged in ", niters, "Newton iterations")

    ############
    # Analysis
    ############
    c, mu = w.sub(0), w.sub(1)

    F_total = total_free_energy(f_chem, kappa, c)
    C_total = total_solute(c)
    benchmark_output.append([float(t), F_total, C_total])

    if df.MPI.rank(mesh.mpi_comm()) == 0:
        print("C_total: ", C_total, "TFE: ", F_total)

    # save solution
    tt = time.time()
    file.write(c , "c" , float(t))
    file.write(mu, "mu", float(t))
    tsave.append(float(t))
    tt = time.time() - tt
    if MPI.comm_world.rank == 0:
        print("Time taken in saving is:", tt)

t2 = time.time()
spent_time = t2 - t1

if df.MPI.rank(mesh.mpi_comm()) == 0:
    print(f'Time spent is {spent_time}')

if df.MPI.rank(mesh.mpi_comm()) == 0:
    np.savetxt(filename + ".csv",
               np.array(tsave),
               fmt='%1.10f',
               header="time",
               delimiter=',',
               comments=''
              )
#
