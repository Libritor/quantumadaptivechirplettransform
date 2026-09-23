#!/bin/bash
# Pre-registered: PRIMARY = chbmitamp (decides the verdict). SECONDARY = chbmit (descriptive).
export QBE_DEVICE=cuda
M="qsvc pqk qlr hybrid hybrid_z logreg svm_rbf"
for tag in chbmitamp chbmit; do
  ~/qbe-venv/bin/python -u tune2.py --tag $tag --norm both --budget 30 --outer 6 --inner 3 \
     --jobs 12 --max-tune 3000 --models $M --out results/tuning_$tag.json > results/tuning_$tag.log 2>&1
done
echo ALLDONE >> results/tuning_chbmit.log
