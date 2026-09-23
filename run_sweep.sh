#!/bin/bash
# Resolution sweep for the continuous-reservoir experiment.
# Runs 60 s, 30 s and 10 s steps (2 s was already done) under one protocol.
# GPU-bound; shares the machine with whatever else is running.
cd ~/Quantum-Brain-Encoding
echo "sweep started $(date +%T)"
for W in 30 15 5; do
  echo "=== $((W * 2))s steps (step-windows $W) ==="
  ~/qbe-venv/bin/python -u run_reservoir_fine.py --step-windows "$W" 2>&1 | grep --line-buffered -v Warning
done
echo SWEEPDONE
