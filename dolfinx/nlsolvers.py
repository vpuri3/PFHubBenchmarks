#
# https://github.com/FEniCS/dolfinx/blob/18a57210eb78705bce2accc153a0a73c69dc75c5/python/test/unit/nls/test_newton.py#L154
from ufl import TrialFunction, derivative

import dolfinx
from dolfinx import fem, la
from dolfinx.fem import form, Function, FunctionSpace, Constant
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_matrix, create_vector, set_bc)

from petsc4py import PETSc

#######################################################################
# Nonlinear PDE Problem for Newton Solver
#######################################################################

class NewtonPDEProblem:
    """ Nonlinear problem class for PDE problems """

    def __init__(self, F, u, bcs):
        V = u.function_space
        du = TrialFunction(V)
        self.L = form(F)
        self.a = form(derivative(F, u, du))
        self.bcs = bcs

    def form(self, x):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x, b):
        """ Assemble residual vector """
        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, self.L)
        apply_lifting(b, [self.a], bcs = [self.bcs], x0 = [x], scale = -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs, x, -1.0)

    def J(self, x, A):
        """ Assemble Jacobian matrix """
        A.zeroEntries()
        assemble_matrix(A, self.a, bcs = self.bcs)
        A.assemble()

    def matrix(self):
        return create_matrix(self.a)

    def vector(self):
        return create_vector(self.L)

#######################################################################
# Nonlinear PDE Problem for SNES solver
#######################################################################

class SnesPDEProblem:
    """ Nonlinear problem class for PDE problems """

    def __init__(self, F, u, bcs):
        V = u.function_space
        du = TrialFunction(V)
        self.L = form(F)
        self.a = form(derivative(F, u, du))
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u


    def F(self, snes, x, F):
        """ Assemble residual vector """

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                  mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)

        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs = [self.bcs], x0 = [x], scale = -1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        """ Assemble Jacobian matrix """
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs = self.bcs)
        J.assemble()

    def matrix(self):
        return create_matrix(self.a)
 
    def vector(self):
        V = self.u.function_space
        #return create_vector(self.L)
        return la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
#
