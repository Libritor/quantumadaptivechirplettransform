#!/usr/bin/env bash
# Rebuild CHB-MIT QACT features with the full-parity engine, then rerun every
# pre-registered classification comparison on them:
#   compare_qact.py --variant parity   top-k selection (where QACT led by +1.9..+3.0)
#   compare_qact_allfeat.py            all features (A4/A5)
#   confirm_siena_parity.py            independent replication on Siena
set -e
cd "$(dirname "$0")"
PY=~/qbe-venv/bin/python
echo "=== parity features, sampled selection ==="
$PY prep_chbmit_qact.py --refine 4 --suffix _parity
echo "=== parity features, argmax control ==="
$PY prep_chbmit_qact.py --refine 4 --suffix _parity_argmax --select argmax
echo "=== top-k comparison (P1-P3) ==="
$PY compare_qact.py --variant parity
echo "=== all-feature comparison (A1-A5) ==="
$PY compare_qact_allfeat.py
echo "=== Siena replication ==="
$PY confirm_siena_parity.py
