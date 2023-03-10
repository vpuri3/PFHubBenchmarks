#
import numpy as np

import ufl
from ufl import TestFunction
from ufl import ds, dx, grad, inner, dot, variable, diff, derivative

import dolfinx
from dolfinx.fem import form, Function, FunctionSpace, Constant
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_matrix, create_vector, set_bc)

#=====================================================================#
# misc
#=====================================================================#

# order parameter interpolation function and derivative
def h(u):
    return u**3 * (6*u**2 - 15*u + 10)

# only active in the interface region
def dh(u):
    return 3*u**2 * (6*u**2 - 15*u + 10) + u**3 * (12*u - 15)

#=====================================================================#

def cahn_hilliard_WF(c, mu, c_, mu_, c0, dt, M, kappa, dfdc):

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

def allen_cahn_WF(eta, eta_, eta0, dt, L, kappa, dfdeta, f):

    # """
    # d/dt \eta  = -L * (F'(\eta) - \kappa * lapl(\eta)) + f
    # """

    lhs  = 1/dt * eta_ * (eta - eta0) * dx
    rhs  = -L * ( eta_ * dfdeta + kappa * inner(grad(eta_), grad(eta))) * dx
    rhs += eta_ * f * dx

    F_AC = lhs - rhs

    return F_AC

def poisson_WF(u, u_, f, M):

    # """
    # div(M * grad(u)) = f
    # """

    F_lhs = -inner(grad(u_), M * grad(u)) * dx
    F_rhs =  u_ * f * dx

    F = F_lhs - F_rhs

    return F

def diffusion_WF_RHS(u, u_, L, D, f1, f2):

    # """
    # L * div(D * grad(u) + f1) + f2
    # """

    rhs  = -inner(grad(L * u_), D * grad(u) + f1) * dx
    rhs += u_ * f2 * dx

    return rhs

def diffusion_WF(u, u_, u0, dt, L, D, f1, f2):

    # """
    # d/dt u = L * div(D * grad(u) + f1) + f2
    # """

    lhs  = 1/dt * u_ * (u - u0) * dx

    rhs  = -inner(grad(L * u_), D * grad(u) + f1) * dx
    rhs += u_ * f2 * dx

    F = lhs - rhs

    return F

def euler_bwd_WF(u, u_, u0, dt, f):

    # """
    # d/dt u = f
    # """

    lhs = 1/dt * u_ * (u - u0) * dx
    rhs = u_ * f * dx

    F = lhs - rhs

    return F

#=====================================================================#
#
