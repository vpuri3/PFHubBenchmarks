#
# https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/2D_elasticity.py.html
#
from dolfin import *
import matplotlib.pyplot as plt

###################################
# Optimization options for the finite element form compiler
###################################
parameters["form_compiler"]["cpp_optimize"] = True
parameters['form_compiler']['cpp_optimize_flags'] = '-O3 -ffast-math'
parameters["form_compiler"]["quadrature_degree"] = 3

###################################
# Create or read mesh
###################################
L = 25.
H = 1.
Nx = 250
Ny = 10
mesh = RectangleMesh(Point(0.,0.), Point((L, H)), Nx, Ny, "crossed")

###################################
# Model Setup - need
#   dt, w, F, J, bcs
###################################


# fem setup
V = VectorFunctionSpace(mesh, 'Lagrange', degree=1)

u  = Function(V)
du = TrialFunction(V)
u_ = TestFunction(V)

# parameters
E = Constant(1e5)
nu = Constant(0.3)

mu = E / 2. / (1. + nu)
lmda = E * nu / (1. + nu) / (1. - 2. * nu)

rho_g = 1e-3
f = Constant((0, -rho_g))

# boundary condition
def left(x, on_boundary):
    return near(x[0], 0.)

bc_left = DirichletBC(V, Constant((0., 0.)), left)

bcs = [bc_left,]

# weak form

def eps(v): # strain
    return sym(grad(v))

def sigma(v):
    return lmda * tr(eps(v)) * Identity(2) + 2. * mu * eps(v)

a = inner(sigma(du), eps(u_)) * dx
l = inner(f, u_) * dx

F = inner(sigma(u), eps(u_)) * dx - inner(f, u_) * dx
J = derivative(F, u, du)

###################################
# Solve
###################################

""" Vanilla Solver """
#solve(a == l, u, bcs)

""" LinearVariational Solver """
#u.vector().zero()
#
#problem = LinearVariationalProblem(a, l, u, bcs)
#solver = LinearVariationalSolver(problem)
#
#linparams = solver.parameters
#linparams['linear_solver'] = 'gmres'
#linparams['preconditioner'] = 'hypre_amg' # SOR was causing problems
#
#kspparams = linparams['krylov_solver']
#kspparams['relative_tolerance'] = 1e-6
#kspparams['maximum_iterations'] = Nx * Ny * 2
#
#kspparams['report'] = True
#kspparams['monitor_convergence'] = True
#
#solver.solve()

""" NonlinearVariational Solver """
u.vector().zero()

problem = NonlinearVariationalProblem(F, u, bcs, J)
solver = NonlinearVariationalSolver(problem)

solver.parameters['nonlinear_solver'] = 'snes'
nlparams  = solver.parameters['snes_solver']

nlparams['report'] = True
nlparams['error_on_nonconvergence'] = False
nlparams['absolute_tolerance'] = 1e-6
nlparams['maximum_iterations'] = 5

#nlparams['line_search'] = 'bt'      #
#nlparams['line_search'] = 'cp'      #
nlparams['line_search'] = 'basic'    #
#nlparams['line_search'] = 'nleqerr' #
#nlparams['line_search'] = 'l2'      #

# 
nlparams['linear_solver'] = 'gmres' # gmres, cg
nlparams['preconditioner'] = 'hypre_amg' # sor, amg, petsc_amg, hypre_amg

#nlparams['krylov_solver']['maximum_iterations'] = 1000
#nlparams['krylov_solver']['monitor_convergence'] = True

niters, converged = solver.solve()
print(f"niters, converged: {niters}, {converged}")
###################################
# Analysis
###################################
plot(1e3 * u, mode="displacement")
plt.show(block=False)

#
