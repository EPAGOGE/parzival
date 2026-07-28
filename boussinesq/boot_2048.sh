#!/bin/bash
# N=2048 tier-one Boussinesq run, MPI-parallel, with a hard idle-burn guard.
#
# GUARD: when the solver exits -- success, crash, or b^2 break -- the box runs
# `shutdown -h now`. Launched with the default shutdown behaviour that means the
# instance STOPS, so compute billing ends automatically the moment the science
# ends. Results stay on EBS (pennies/hr) to be pulled, then terminated by hand.
# It cannot idle-burn: nothing has to be remembered for the meter to stop.
set -u
cd /home/ubuntu
export DEBIAN_FRONTEND=noninteractive

# --- deps (apt for MPI, conda-forge for Dedalus: the path proven on the c6i) ---
sudo apt-get update -qq                                   >boot.log 2>&1
sudo apt-get install -y -qq libopenmpi-dev openmpi-bin    >>boot.log 2>&1
curl -fsSL -o mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh >>boot.log 2>&1
bash mf.sh -b -p /home/ubuntu/mf                          >>boot.log 2>&1
/home/ubuntu/mf/bin/conda install -y -c conda-forge dedalus mpi4py >>boot.log 2>&1

if ! /home/ubuntu/mf/bin/python3 -c "import dedalus.public, mpi4py" 2>/dev/null; then
  echo "DEDALUS INSTALL FAILED $(date -u)" > INSTALL_FAILED
  sudo shutdown -h +5            # do not sit idle on a failed install
  exit 1
fi

# --- one MPI rank per PHYSICAL core (hyperthread ranks hurt spectral codes) ---
NP=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)
[ "$NP" -lt 1 ] && NP=1
echo "launching N=2048 on $NP ranks $(date -u)" > LAUNCHED

mkdir -p runs
OMP_NUM_THREADS=1 /home/ubuntu/mf/bin/mpirun -n "$NP" --oversubscribe \
  /home/ubuntu/mf/bin/python3 dedalus_bsq.py \
  --Nx 2048 --Nz 2048 --A 4 --stop 1.56 \
  --run-id M2048 --checkpoint-wall 600 \
  --out runs/big_N2048.json > runs/big_N2048.log 2>&1

echo "solver exited rc=$? $(date -u)" > FINISHED
sync
sudo shutdown -h now             # <-- the guard: meter stops, data survives
