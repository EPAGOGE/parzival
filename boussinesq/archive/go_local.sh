#!/bin/bash
# Local MPI run, 8 ranks on 10 cores. Free. Self-logging.
export PATH="$(brew --prefix open-mpi)/bin:$PATH"
export DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib:$(brew --prefix open-mpi)/lib:$(brew --prefix hdf5)/lib"
export OMP_NUM_THREADS=1
cd /Users/epagogellc/parzival/boussinesq
mpirun -n 8 /Users/epagogellc/parzival/.venv-dedalus/bin/python dedalus_axisym.py --scenario \
  --Nz 1024 --Nr 3072 --A 100 --ic-power 4 --zpow 1 --r0 0.4 \
  --tmax 0.0068 --ckpt-sim-dt 1e-4 --ckpt-max-writes 30 \
  --run-id mpi1024 --out ../runs/mpi1024.json
echo "exit=$?" > ../runs/MPI_EXIT
