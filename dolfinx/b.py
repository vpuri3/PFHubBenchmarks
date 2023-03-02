#
import numpy as np

import ufl
from ufl import TestFunction, TrialFunction, derivative
from ufl import dx, grad, inner

import dolfinx
from dolfinx import fem, io, mesh, plot, la
from dolfinx.fem import form, Function, FunctionSpace, Constant
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_matrix, create_vector, set_bc)

from mpi4py import MPI
from petsc4py import PETSc
from petsc4py.PETSc import ScalarType

"""
 Newton Solve
"""

class NonlinearPDEProblem:
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
        apply_lifting(b, [self.a], bcs = [bcs], x0 = [x], scale = -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs, x, -1.0)

    def J(self, x, A):
        """ Assemble Jacobian matrix """
        A.zeroEntries()
        assemble_matrix(A, self.a, bcs = bcs)
        A.assemble()

    def matrix(self):
        return create_matrix(self.a)

    def vector(self):
        return create_vector(self.L)

msh = mesh.create_unit_square(MPI.COMM_WORLD, 12, 12)
V = FunctionSpace(msh, ("Lagrange", 1))
u = Function(V)
v = TestFunction(V)
F = inner(10.0, v) * dx - inner(grad(u), grad(v)) * dx

bc = fem.dirichletbc(PETSc.ScalarType(1.0), fem.locate_dofs_geometrical(V, lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))), V)

bcs = [bc,]

problem = NonlinearPDEProblem(F, u, bcs)

solver = dolfinx.cpp.nls.petsc.NewtonSolver(MPI.COMM_WORLD)
solver.setF(problem.F, problem.vector())
solver.setJ(problem.J, problem.matrix())
solver.set_form(problem.form)
n, converged = solver.solve(u.vector)

print("Converged = ", converged, " in ", n, " iterations.")

"""
 SNES
"""
class SNESProblem:
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
        apply_lifting(F, [self.a], bcs = [bcs], x0 = [x], scale = -1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, bcs, x, -1.0)

    def J(self, snes, x, J, P):
        """ Assemble Jacobian matrix """
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs = bcs)
        J.assemble()

    def matrix(self):
        return create_matrix(self.a)
 
    def vector(self):
        V = self.u.function_space
        #return create_vector(self.L)
        return la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)


msh = mesh.create_unit_square(MPI.COMM_WORLD, 12, 15)
V = FunctionSpace(msh, ("Lagrange", 1))
u = Function(V)
v = TestFunction(V)
F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(grad(u), grad(v)) * dx - inner(u, v) * dx

u_bc = Function(V)
bc = fem.dirichletbc(u_bc, fem.locate_dofs_geometrical(V, lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))))

bcs = [bc,]

u_bc.x.array[:] = 1.0
u.x.array[:] = 0.9

problem = SNESProblem(F, u, bcs)

snes = PETSc.SNES().create()
snes.setFunction(problem.F, problem.vector())
snes.setJacobian(problem.J, problem.matrix())

snes.setTolerances(rtol = 1e-9, max_it = 10)
ksp = snes.getKSP()
ksp.setType("preonly")
ksp.setTolerances(rtol = 1e-9)

pc = ksp.getPC()
pc.setType("lu")

snes.solve(None, u.vector)
converged = snes.getConvergedReason()
n = snes.getIterationNumber()

print("Converged = ", converged, " in ", n, " iterations.")
#
