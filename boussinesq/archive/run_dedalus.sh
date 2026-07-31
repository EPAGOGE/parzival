#!/bin/bash
# Dedalus runtime env: serial (OMP=1, threading degrades Dedalus), brew dylibs.
export OMP_NUM_THREADS=1
export PATH="$(brew --prefix open-mpi)/bin:$PATH"
export DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib:$(brew --prefix open-mpi)/lib:$(brew --prefix hdf5)/lib"
cd ~/parzival/boussinesq
exec ~/parzival/.venv-dedalus/bin/python "$@"
