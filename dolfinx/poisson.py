#
import numpy as np

import ufl
from ufl import ds, dx, grad, inner, dot
from ufl import sin, cos, tan, exp

import dolfinx
from dolfinx import fem, io, mesh, plot

from mpi4py import MPI
from petsc4py.PETSc import ScalarType

#import pyvista

"""
Solve Poisson equation
        -\Delta u = f on \Omega   = [0, 2] \cross [0, 1]
                u = 0 on \Gamma_D = {(0, y) U (2, y)}
  n \cdot \grad u = g on \Gamma_N = {(x, 0) U (x, 1)}

f = 10 * exp(-((x-0.5)**2 + (y - 0.5)**2) / 0.02 )
g = sin(5 * x)
"""

# mesh
Nx = 32
Ny = 16

Lx = 2.0
Ly = 1.0

msh = mesh.create_rectangle(comm = MPI.COMM_WORLD,
                            points = ((0.0, 0.0), (Lx, Ly)), n = (Nx, Ny),
                            cell_type = mesh.CellType.triangle,
                            )

V = fem.FunctionSpace(msh, ("Lagrange", 1))

# boundary condition
BC_dirichlet = lambda x: np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], Lx))
BC_neumann   = lambda x: np.logical_or(np.isclose(x[1], 0.0), np.isclose(x[1], Ly))

facets_dirichlet = mesh.locate_entities_boundary(msh, dim = 1, marker = BC_dirichlet)

dofs_dirichlet = fem.locate_dofs_topological(V = V, entity_dim = 1,
                                             entities = facets_dirichlet)

bc_dirichlet = fem.dirichletbc(value = ScalarType(0), dofs = dofs_dirichlet, V = V)

bcs = [
        bc_dirichlet,
       ]

# equation
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)

f = 10 * exp(-((x[0]-0.5)**2 + (x[1] - 0.5)**2) / 0.02 )
g = sin(5 * x[0])

a = inner(grad(u), grad(v)) * dx
L = inner(f, v) * dx + inner(g, v) * ds

# solve
petsc_options = {
    "ksp_type" : "gmres",
    "pc_type" : "sor",
    }
#petsc_options = {
#    "ksp_type" : "preonly",
#    "pc_type" : "lu",
#    }

problem = fem.petsc.LinearProblem(a, L, bcs = bcs, petsc_options = petsc_options)
uh = problem.solve()

# saving and visualization
file = io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w")
file.write_mesh(msh)
file.write_function(uh)
file.close()

#cells, types, x = plot.create_vtk_mesh(V)
#grid = pyvista.UnstructuredGrid(cells, types, x)
#grid.point_data["u"] = uh.x.array.real
#grid.set_active_scalars("u")
#
#plotter = pyvista.Plotter()
#plotter.add_mesh(grid, show_edges = True)
#warped = grid.warp_by_scalar()
#plotter.add_mesh(warped)
#plotter.show()
#
