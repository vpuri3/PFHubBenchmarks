# PFHubBenchmarks
PFHub benchmark problems implemented with FEniCS. Problem details https://pages.nist.gov/pfhub/

# Installation and Running

Install the code using Miniconda.
```bash
conda create -n PF
conda activate PF
conda install -c conda-forge fenics dolfin-adjoint mpich pyvista matplotlib
```

Run the code in parallel
```bash
mpirun -np 8 python dolfin/bench<1,2,3,6>.py
```
or in serial

```bash
python dolfin/bench<1,2,3,6>.py
```
