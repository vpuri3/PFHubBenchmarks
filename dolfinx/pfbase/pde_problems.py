#
"""
REFERENCES
    - https://github.com/FEniCS/dolfinx/blob/18a57210eb78705bce2accc153a0a73c69dc75c5/python/test/unit/nls/test_newton.py#L154
    - https://gitlab.com/newfrac/newfrac-fenicsx-training/-/blob/main/notebooks/python/nonlinear_pde_problem.py
    - https://gitlab.com/newfrac/newfrac-fenicsx-training/-/blob/main/notebooks/python/snes_problem.py
"""
from ufl import TrialFunction, derivative

import dolfinx
from dolfinx import fem, la
from dolfinx.fem import form, Function, FunctionSpace, Constant
from dolfinx.fem.petsc import (apply_lifting, assemble_matrix, assemble_vector,
                               create_matrix, create_vector, set_bc)

from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from petsc4py import PETSc

## KSP, SNES Options
# https://petsc.org/release/docs/manualpages/KSP/KSPType/
# https://petsc.org/release/docs/manualpages/SNES/SNESLineSearchType/

## SNES SOLVER
# https://petsc.org/release/docs/manual/snes/
# https://petsc.org/release/docs/manualpages/SNES/SNESLineSearchType/
# https://petsc.org/release/docs/manual/ksp/#krylov-methods

#=====================================================================#
# Nonlinear PDE Problem Types
#=====================================================================#

###
# dolfinx.NewtonSolver
###

class NewtonPDEProblem:
    """
    Nonlinear problem class compatible with dolfinx.NewtonSolver
    """

    def __init__(self, F, u, bcs = []):
        """
        This class set up structures for solving a nonlinear problem
        using Newton's method.

        Parameters
        ==========
        F: Residual form
        u: Solution function
        bcs: List of boundary conditions
        """

        V = u.function_space
        du = TrialFunction(V)
        self.L = form(F)
        self.a = form(derivative(F, u, du))
        self.bcs = bcs

    def form(self, x):
        """
        This function is called before the residual or Jacobian is computed
        inside the Newton step.  It is usually used to update ghost values.

        Parameters
        ==========
        x: Vector containing the latest solution.
        """
        # scatter ghost values
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def F(self, x, b):
        """
        Assemble residual F into the vector b

        Parameters
        ==========
        x: Vector containing the latest solution
        b: Vector to assemble the residual into
        """

        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, self.L)
        # apply boundary conditions
        apply_lifting(b, [self.a], bcs = [self.bcs], x0 = [x], scale = -1.0)
        # gather ghost values in b
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs, x, -1.0)

    def J(self, x, A):
        """
        Assemble Jacobian matrix

        Parameters
        ==========
        x: Vector containing the latest solution
        A: Matrix to assemble the Jacobian into
        """
        A.zeroEntries()
        assemble_matrix(A, self.a, bcs = self.bcs)
        A.assemble()

    def matrix(self):
        return create_matrix(self.a)

    def vector(self):
        return create_vector(self.L)

###
# PETSc.SNES
###

class SnesPDEProblem:
    """
    Nonlinear problem class compatible with PETSc.SNES solver
    """

    def __init__(self, F, u, bcs = []):
        """
        This class set up structures for solving a nonlinear problem
        using Newton's method

        Parameters
        ==========
        F: Residual form
        u: Solution function
        bcs: List of boundary conditions
        """

        V = u.function_space
        du = TrialFunction(V)
        self.L = form(F)
        self.a = form(derivative(F, u, du))
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u # vector containing the latest solution

    def F(self, snes, x, b):
        """
        Assemble residual F into vector x

        Parameters
        ==========
        snes: the snes object
        x: Vector containing the latest solution
        b: Vector to assemble the residual into
        """

        # assign the vector to the function
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                      mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.vector) # u <- x
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT,
                                  mode=PETSc.ScatterMode.FORWARD)

        with b.localForm() as b_local:
            b_local.set(0.0)

        assemble_vector(b, self.L)
        apply_lifting(b, [self.a], bcs = [self.bcs], x0 = [x], scale = -1.0)
        b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs, x, -1.0)

    def J(self, snes, x, J, P):
        """
        Assemble Jacobian matrix

        Parameters
        ==========
        snes: the snes object
        x: vector containing the latest solution
        J: matrix to assemble the jacobian into
        P: PETSc.Mat
        """
        J.zeroEntries()
        assemble_matrix(J, self.a, bcs = self.bcs)
        J.assemble()

    def matrix(self):
        return create_matrix(self.a)
 
    def vector(self):
        V = self.u.function_space
        return create_vector(self.L)
        #return la.create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)

#=====================================================================#

if __name__ == "__main__":
    import numpy as np
    
    import ufl
    from ufl import TestFunction, dx, grad, inner

    import dolfinx
    from dolfinx import fem, mesh
    from dolfinx.fem import form, Function, FunctionSpace, Constant
    
    from mpi4py import MPI
    from petsc4py import PETSc
    
    """ Problem Setup """
    
    msh = mesh.create_unit_square(MPI.COMM_WORLD, 12, 15)
    V = FunctionSpace(msh, ("Lagrange", 1))
    u = Function(V)
    v = TestFunction(V)
    F = inner(5.0, v) * dx - ufl.sqrt(u * u) * inner(grad(u), grad(v)) * dx - inner(u, v) * dx

    u_bc = Function(V)
    u_bc.x.array[:] = 1.0

    bc_cond = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))
    dofs = fem.locate_dofs_geometrical(V, bc_cond)
    bc = fem.dirichletbc(u_bc, dofs)

    bcs = [bc,]

    """
    SNES Solver
    """

    u.x.array[:] = 0.9

    problem = SnesPDEProblem(F, u, bcs)

    snes = PETSc.SNES().create()
    snes.setFunction(problem.F, problem.vector())
    snes.setJacobian(problem.J, problem.matrix())

    # KSP  opts: https://petsc.org/release/docs/manualpages/KSP/KSPType/
    # SNES opts: https://petsc.org/release/docs/manualpages/SNES/SNESLineSearchType/

    snes.setTolerances(rtol = 1e-6, max_it = 10)
    ksp = snes.getKSP()
    ksp.setType("gmres")
    ksp.setTolerances(rtol = 1e-6, max_it = 100)

    pc = ksp.getPC()
    pc.setType("sor")

    opts = PETSc.Options()

    snes_pfx = snes.prefix
    opts[f"{snes_pfx}snes_linesearch_type"] = "basic"
    opts[f"{snes_pfx}snes_monitor"] = None
    opts[f"{snes_pfx}snes_linesearch_monitor"] = None

    snes.setFromOptions()

    snes.solve(None, u.vector)
    niters, converged = snes.getIterationNumber(), snes.converged

    if MPI.COMM_WORLD.rank == 0:
        print("SNES solver converged = ", converged, " in ", niters, " iterations.")

    """
    Newton Solver
    """

    u.x.array[:] = 0.9

    problem = NewtonPDEProblem(F, u, bcs)

    newton = dolfinx.cpp.nls.petsc.NewtonSolver(MPI.COMM_WORLD)
    newton.setF(problem.F, problem.vector())
    newton.setJ(problem.J, problem.matrix())
    newton.set_form(problem.form)
    n, converged = newton.solve(u.vector)

    if MPI.COMM_WORLD.rank == 0:
        print("Newton solver converged = ", converged, " in ", n, " iterations.")
    #
