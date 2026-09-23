#!/usr/bin/env bash
# Rebuild CHB-MIT QACT features with the full-parity engine, then rerun the
# pre-registered classification comparison (A4/A5 in compare_qact_allfeat.py).
set -e
cd "$(dirname "$0")"
PY=~/qbe-venv/bin/python
echo "=== parity features, sampled selection ==="
$PY prep_chbmit_qact.py --refine 4 --suffix _parity
echo "=== parity features, argmax control ==="
$PY prep_chbmit_qact.py --refine 4 --suffix _parity_argmax --select argmax
echo "=== classification comparison ==="
$PY compare_qact_allfeat.py
