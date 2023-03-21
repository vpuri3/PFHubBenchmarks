#
from dolfin import *
import numpy as np
import pandas as pd

import os, shutil

def process_bench1(dirname):
    hdf = HDF5File(MPI.comm_world, dirname + "conc.h5", "r")

    stats = pd.read_csv(dirname + "stats.csv")
    times = np.array(stats.time)
    Nt = times.size

    mesh = Mesh()
    hdf.read(mesh, "mesh", True)

    V = FunctionSpace(mesh, 'P', 1)

    cs = []

    for i in range(Nt):
        c = Function(V)

        hdf.read(c, f"c/vector_{i}")

        cs.append(c)

    if MPI.comm_world.rank == 0:
        print("done reading HDF5")

    return mesh, times, cs, stats

dirname = "./results/bench1/"
mesh, times, cs, stats = process_bench1(dirname)

file = File("./results/bench1/" + "c.pvd" , "compressed")
file << mesh

for i in range(times.size):
    if MPI.comm_world.rank == 0:
        print(f"writing step {i}")
    file << cs[i]
