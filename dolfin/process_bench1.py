#
from dolfin import *
import numpy as np

import os, shutil

def process_bench1(dirname):
    hdf = HDF5File(MPI.comm_world, dirname + "conc.h5", "r")

    #ts = np.loadtxt(dirname + "stats.csv", skiprows=1)
    #Nt = ts.size

    mesh = Mesh()
    hdf.read(mesh, "mesh", True)

    V = FunctionSpace(mesh, 'P', 1)

    cs = []

    for i in range(Nt):
        c = Function(V)

        hdf.read(c, f"xi/vector_{i}")

        cs.append(c)

    if MPI.comm_world.rank == 0:
        print("done reading HDF5")

    #return mesh, ts, cs
    return mesh, cs

dirname = "./results/bench1/"
mesh, cs = process_case(dirname)
#mesh, ts, cs = process_case(dirname)

file = File("c.pvd" , "compressed")
file << mesh

for i in range(ts.size):
    file << cs[i]
