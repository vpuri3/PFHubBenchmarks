#
import numpy as np

import ufl
from ufl import TestFunction
from ufl import ds, dx, grad, inner, dot, variable, diff, derivative

import dolfinx
from dolfinx.fem import form, Function, FunctionSpace, Constant
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_matrix, create_vector, set_bc)

# use lambdas for postprocessing functions
# with default values

def time_loop(w, w0, dt, dt_min):
    return w

