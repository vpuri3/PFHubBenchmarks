#
import numpy as np

import ufl
from ufl import ds, dx, grad, inner, dot
from ufl import sin, cos, tan, exp

import dolfinx
from dolfinx import fem, io, mesh, plot

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

#######################################################################
# Initial Conditions
#######################################################################
"""
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
"""

#######################################################################
# time loop
#######################################################################
# use lambdas for postprocessing functions
# with default values
def time_loop(w, w0, dt, dt_min):
    return w
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
