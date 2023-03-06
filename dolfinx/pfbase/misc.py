#
# https://gitlab.com/newfrac/newfrac-fenicsx-training/-/blob/main/notebooks/python/utils.py
import dolfinx
import ufl
from ufl import TestFunction, TrialFunction, inner, dx

from petsc4py import PETSc

def project(v, target_func, bcs = []):

    V = target_func.function_space
    dx = dx(V.mesh)

    # variational problem for projection
    w = TestFunction(V)

    # assemble linear system

    return

def eval_on_points():
    return

if __name__ == "__main__":
    import dolfinx
    from MPI4py import MPI
    from petsc4py import PETSc
