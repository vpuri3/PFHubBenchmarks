#
from dolfin import *
from mpi4py import MPI as pyMPI
import time

import numpy as np

#######################################################################
# Sub domain for Periodic boundary condition
#######################################################################
class PeriodicBoundary(SubDomain):
    def __init__(self, Lx, Ly, *args, **kwargs):
        super.__init__(*args, **kwargs)
        self.Lx = Lx
        self.Ly = Ly
    # Left and bottom boundaries are "target domain" G
    def inside(self, x, on_boundary):
        # return True if on left or bottom boundaries
        # return bool((near(x[0], 0) or near(x[1], 0))  and on_boundary)
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 1) and (1, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and 
                (not ((near(x[0], 0) and near(x[1], self.Ly)) or 
                        (near(x[0], self.Lx) and near(x[1], 0)))) and on_boundary)
    # Map top and right boundaries (H) to bottom and left boundaries (G)
    def map(self, x, y):
        # map top right corner
        if near(x[0], self.Lx) and near(x[1], self.Ly):
            y[0] = x[0] - self.Lx
            y[1] = x[1] - self.Ly
        # map right edge
        elif near(x[0], self.Lx):
            y[0] = x[0] - self.Lx
            y[1] = x[1]
        # map top edge
        else:
            y[0] = x[0]
            y[1] = x[1] - self.Ly

#######################################################################
# Parallel Evaluation
# https://fenicsproject.discourse.group/t/problem-with-evaluation-at-a-point-in-parallel/1188/5
#######################################################################
def mpi4py_comm(comm):
    ''' get mpi4py communicator '''
    try:
        return comm.tompi4py()
    except AttributeError:
        return comm

def peval(f, x):
    ''' parallel synced eval '''
    try:
        yloc = f(x)
    except RuntimeError:
        yloc = np.inf * np.ones(f.value_shape())

    yloc = np.array([yloc])
    yglo = np.zeros_like(yloc)

    comm = mpi4py_comm(f.function_space().mesh().mpi_comm())
    comm.Allreduce(yloc, yglo, op=pyMPI.MIN)

    return yglo

#######################################################################
# Interpolation
#######################################################################

def sample(u, Nx, Ny, Lx, Ly):
    '''
    return array of size [Nx], [Ny] [Nx, Ny]
    '''

    if MPI.comm_world.rank == 0:
        print(f'Sampling field - {Nx * Ny} points')

    tt = time.time()

    xs = np.linspace(0, Lx, Nx)
    ys = np.linspace(0, Ly, Ny)

    us = np.zeros((Nx, Ny))

    for j in range(Ny):
        y = ys[j]
        for i in range(Nx):
            x = xs[i]
            us[i,j] = peval(u, np.array([x,y]))

    tt = time.time() - tt
    if MPI.comm_world.rank == 0:
        print(f'Sampling done - time taken: {tt}')

    return xs, ys, us

#
# below methods dont do well in parallel
#
# https://fenicsproject.discourse.group/t/integrate-over-some-but-not-all-dimensions/5201/9
#

class UxExpression(UserExpression):
    """
    u(x) = u(x, y=ypos)
    """
    def __init__(self, u2d, ypos):
        super().__init__(xpos)
        self.u2d  = u2d
        self.ypos = ypos

        self.point = np.array([0.,0.])

    def eval(self, values, x):
        self.point[0] = x[0]
        self.point[1] = self.xpos

        values[0] = peval(self.u2d, self.point)

    def value_shape(self):
        return ()

class UyExpression(UserExpression):
    """
    u(y) = u(x=xpos, y)
    """
    def __init__(self, u2d, xpos):
        super().__init__(xpos)
        self.u2d  = u2d
        self.xpos = xpos

        self.point = np.array([0.,0.])

    def eval(self, values, x):
        self.point[0] = self.xpos
        self.point[1] = x[0]

        values[0] = peval(self.u2d, self.point)

    def value_shape(self):
        return ()

class UdxExpression(UserExpression):
    """
    U(y) = \int_0^Lx u(x,y) dx
    """
    def __init__(self, u2d, V1d):
        super().__init__()
        self.u2d = u2d
        self.V1d = V1d

    def eval(self, values, x):
        Ux = interpolate(UxExpression(self.u2d, x[0]), self.V1d)
        values[0] = assemble(Uy * dx)

    def value_shape(self):
        return ()

class UdyExpression(UserExpression):
    """
    U(x) = \int_0^Ly u(x,y) dy
    """
    def __init__(self, u2d, V1d):
        super().__init__()
        self.u2d = u2d
        self.V1d = V1d

    def eval(self, values, x):
        Uy = interpolate(UyExpression(self.u2d, x[0]), self.V1d)
        values[0] = assemble(Uy * dx)

    def value_shape(self):
        return ()

#######################################################################
# Initial Conditions
#######################################################################
class InitialConditionsBench1(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c0 = args[0]
        self.epsilon = args[1]
    def eval(self, values, x):
        # indices
        # c, mu
        # 0,  1

        values[0] = self.c0 + self.epsilon*(np.cos(0.105*x[0])*np.cos(0.11*x[1])
                +(np.cos(0.13*x[0])*np.cos(0.087*x[1]))**2
                + np.cos(0.025*x[0] - 0.15*x[1])*np.cos(0.07*x[0] - 0.02*x[1]))
        values[1] = 0.0

    def value_shape(self):
        return (2,)

class ICB2_jank1(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c0 = args[0]
        self.epsilon = args[1]
        self.epsilon_eta = args[2]
        self.psi = args[3]
    def eval(self, values, x):
        # indices
        # c, mu, eta<1:4>,
        # 0,  1,      3:6,

        values[0] = self.c0 + self.epsilon*(np.cos(0.105*x[0])*np.cos(0.11*x[1])
                +(np.cos(0.13*x[0])*np.cos(0.087*x[1]))**2
                + np.cos(0.025*x[0] - 0.15*x[1])*np.cos(0.07*x[0] - 0.02*x[1]))
        values[1] = 0.0

        for i in range(1):
            ii = i + 1.0
            epsilon_eta = self.epsilon_eta
            psi = self.psi
            values[2+i] = epsilon_eta * (
                    np.cos((0.01*ii)*x[0] - 4.0) * np.cos((0.007+0.01*ii)*x[1]) +
                    np.cos((0.11+0.01*ii)*x[0]) * np.cos((0.11+0.01*ii)*x[1]) +
                    psi * (
                        np.cos((0.046+0.001*ii)*x[0] - (0.0405+0.001*ii)*x[1]) *
                        np.cos((0.031+0.001*ii)*x[0] - (0.004+0.001*ii)*x[1])
                        )**2
                    )**2

    def value_shape(self):
        return (3,)

class ICB2_jank2(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c0 = args[0]
        self.epsilon = args[1]
        self.epsilon_eta = args[2]
        self.psi = args[3]
    def eval(self, values, x):
        # indices
        # c, mu, eta<1:4>,
        # 0,  1,      3:6,

        values[0] = self.c0 + self.epsilon*(np.cos(0.105*x[0])*np.cos(0.11*x[1])
                +(np.cos(0.13*x[0])*np.cos(0.087*x[1]))**2
                + np.cos(0.025*x[0] - 0.15*x[1])*np.cos(0.07*x[0] - 0.02*x[1]))
        values[1] = 0.0

        for i in range(2):
            ii = i + 1.0
            epsilon_eta = self.epsilon_eta
            psi = self.psi
            values[2+i] = epsilon_eta * (
                    np.cos((0.01*ii)*x[0] - 4.0) * np.cos((0.007+0.01*ii)*x[1]) +
                    np.cos((0.11+0.01*ii)*x[0]) * np.cos((0.11+0.01*ii)*x[1]) +
                    psi * (
                        np.cos((0.046+0.001*ii)*x[0] - (0.0405+0.001*ii)*x[1]) *
                        np.cos((0.031+0.001*ii)*x[0] - (0.004+0.001*ii)*x[1])
                        )**2
                    )**2


    def value_shape(self):
        return (4,)

class InitialConditionsBench2(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c0 = args[0]
        self.epsilon = args[1]
        self.epsilon_eta = args[2]
        self.psi = args[3]
    def eval(self, values, x):
        # indices
        # c, mu, eta<1:4>,
        # 0,  1,      3:6,

        values[0] = self.c0 + self.epsilon*(np.cos(0.105*x[0])*np.cos(0.11*x[1])
                +(np.cos(0.13*x[0])*np.cos(0.087*x[1]))**2
                + np.cos(0.025*x[0] - 0.15*x[1])*np.cos(0.07*x[0] - 0.02*x[1]))
        values[1] = 0.0

        for i in range(4):
            ii = i + 1.0
            values[2+i] = self.epsilon_eta * (
                    np.cos((0.01*ii)*x[0] - 4.0) * np.cos((0.007+0.01*ii)*x[1]) +
                    np.cos((0.11+0.01*ii)*x[0]) * np.cos((0.11+0.01*ii)*x[1]) +
                    self.psi * (
                        np.cos((0.046+0.001*i)*x[0] - (0.0405+0.001*i)*x[1]) *
                        np.cos((0.031+0.001*i)*x[0] - (0.004+0.001*i)*x[1])
                        )**2
                    )**2

    def value_shape(self):
        return (6,)

class InitialConditionsBench3(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Delta = args[0]
        self.r = args[1]
        self.w = args[2]
        self.vin  = args[3]
        self.vout = args[4]
    def eval(self, values, x):
        # indices
        # U, phi
        # 0,   1

        values[0] = self.Delta

        r = np.sqrt(x[0]**2 + x[1]**2)

        if r < (self.r - 0.5 * self.w):
            values[1] = self.vin
        elif r > (self.r + 0.5 * self.w):
            values[1] = self.vout
        else:
            values[1] = self.vout + 0.5*(self.vin - self.vout) * (
                    1.0 + np.cos(np.pi * (r - self.r + 0.5 * self.w) / self.w)
                    )

    def value_shape(self):
        return (2,)

class InitialConditionsBench6(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c0 = args[0]
        self.c1 = args[1]
    def eval(self, values, x):
        # indices
        # c, mu, phi,
        # 0,  1,   2,

        values[0] = self.c0 + self.c1*(np.cos(0.2*x[0])*np.cos(0.11*x[1])
                +(np.cos(0.13*x[0])*np.cos(0.087*x[1]))**2
                + np.cos(0.025*x[0] - 0.15*x[1])*np.cos(0.07*x[0] - 0.02*x[1]))
        values[1] = 0.0
        values[2] = 0.0

    def value_shape(self):
        return (3,)

class LangevinNoise(UserExpression):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

        self.amp = args[0] # amplitude

        #np.random.seed(2)

    def eval(self, values, x):

        values[0] = self.amp * np.random.uniform(-1/2, 1/2)
        values[1] = self.amp * np.random.uniform(-1/2, 1/2)
        values[2] = self.amp * np.random.uniform(-1/2, 1/2)

    def value_shape(self):
        return (3,)

#######################################################################
# weak forms
#######################################################################
def cahn_hilliard_weak_form(c, mu, c_, mu_, c0, dt, M, kappa, dfdc):

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

def allen_cahn_RHS_IBP(eta, eta_, L, kappa, dfdeta, f):

    # """
    # -L * (F'(\eta) - \kappa * lapl(\eta)) + f
    # """

    rhs  = -L * ( eta_ * dfdeta + kappa * inner(grad(eta_), grad(eta)))
    rhs += eta_ * f

    return rhs

def allen_cahn_weak_form(eta, eta_, eta0, dt, L, kappa, dfdeta, f):

    # """
    # d/dt \eta  = -L * (F'(\eta) - \kappa * lapl(\eta)) + f
    # """

    lhs  = 1/dt * eta_ * (eta - eta0) * dx
    rhs  = -L * ( eta_ * dfdeta + kappa * inner(grad(eta_), grad(eta))) * dx
    rhs += eta_ * f * dx

    F_AC = lhs - rhs

    return F_AC

def poisson_weak_form(u, u_, f, M):

    # """
    # div(M * grad(u)) = f
    # """

    F_lhs = -inner(grad(u_), M * grad(u)) * dx
    F_rhs =  u_ * f * dx

    F = F_lhs - F_rhs

    return F

def diffusion_weak_form_RHS(u, u_, L, D, f1, f2):

    # """
    # L * div(D * grad(u) + f1) + f2
    # """

    rhs  = -inner(grad(L * u_), D * grad(u) + f1) * dx
    rhs += u_ * f2 * dx

    return rhs

def diffusion_weak_form(u, u_, u0, dt, L, D, f1, f2):

    # """
    # d/dt u = L * div(D * grad(u) + f1) + f2
    # """

    lhs  = 1/dt * u_ * (u - u0) * dx

    rhs  = -inner(grad(L * u_), D * grad(u) + f1) * dx
    rhs += u_ * f2 * dx

    F = lhs - rhs

    return F

def euler_bwd_weak_form(u, u_, u0, dt, f):

    # """
    # d/dt u = f
    # """

    lhs = 1/dt * u_ * (u - u0) * dx
    rhs = u_ * f * dx

    F = lhs - rhs

    return F

#######################################################################
# misc
#######################################################################

# order parameter interpolation function and derivative
def h(u):
    return u**3 * (6*u**2 - 15*u + 10)

# only active in the interface region
def dh(u):
    return 3*u**2 * (6*u**2 - 15*u + 10) + u**3 * (12*u - 15)

#######################################################################
#
