#!/bin/bash
# Standing certification watcher. Reruns the backwards proof every 20 min,
# appends one line per tick, and makes any regression impossible to miss.
export PATH="$(brew --prefix open-mpi)/bin:$PATH"
export DYLD_LIBRARY_PATH="$(brew --prefix fftw)/lib:$(brew --prefix open-mpi)/lib:$(brew --prefix hdf5)/lib"
V=/Users/epagogellc/parzival/.venv-dedalus/bin/python
cd /Users/epagogellc/parzival/boussinesq
while true; do
  OUT=$($V verify.py 2>&1)
  TOT=$(echo "$OUT" | grep -o '[0-9]*/15 pass' | head -1)
  TS=$(date '+%m-%d %H:%M')
  if echo "$TOT" | grep -q '^15/15'; then
    echo "$TS  PASS  $TOT" >> verify_watch.log
    echo "PASS $TS $TOT" > VERIFY_STATUS
  else
    echo "$TS  *** REGRESSION ***  $TOT" >> verify_watch.log
    { echo "FAIL $TS $TOT"; echo "$OUT" | grep FAIL; } > VERIFY_STATUS
    cp VERIFY_STATUS "REGRESSION_$(date +%H%M).txt"
  fi
  sleep 1200
done
