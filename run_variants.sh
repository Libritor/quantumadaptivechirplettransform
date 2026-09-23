#!/bin/bash
# Extract CHB-MIT features for the idea #2 and #3 variants.
cd ~/Quantum-Brain-Encoding
V=~/qbe-venv/bin/python
echo "=== cubic phase (argmax) ==="
$V -u prep_chbmit_qact.py --refine 4 --select argmax --rates3 -40 -20 0 20 40 --suffix _cubic 2>&1 | grep -E "QACT:|saved"
echo "=== skew envelope (argmax) ==="
$V -u prep_chbmit_qact.py --refine 4 --select argmax --skews -4 -2 0 2 4 --suffix _skew 2>&1 | grep -E "QACT:|saved"
echo "=== cubic + skew (argmax) ==="
$V -u prep_chbmit_qact.py --refine 4 --select argmax --rates3 -30 0 30 --skews -3 0 3 --suffix _cubicskew 2>&1 | grep -E "QACT:|saved"
echo "=== proposal k=32 ==="
$V -u prep_chbmit_qact.py --refine 4 --select proposal --candidates 32 --suffix _prop32 2>&1 | grep -E "QACT:|saved"
echo VARIANTSDONE
