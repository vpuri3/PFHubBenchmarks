#
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import time

from pfbase import *

filename = "out_b1"
hdf = HDF5File(MPI.comm_world, filename + ".h5", "r")

tsave = np.loadtxt(filename + ".csv", skiprows=1)
Nsave = tsave.size

mesh = Mesh()
hdf.read(mesh, "mesh", True)
V = FunctionSpace(mesh, 'P', 1)

cs  = []
mus = []

for i in range(Nsave):
    c  = Function(V)
    mu = Function(V)

    hdf.read(c , f"c/vector_{i}")
    hdf.read(mu, f"mu/vector_{i}")

    cs.append(c)
    mus.append(mu)
#
