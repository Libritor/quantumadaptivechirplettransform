#!/bin/bash
# Train/evaluate all persons, 4 at a time on the GPU, then run the pre-registered analysis.
cd ~/Quantum-Brain-Encoding
mkdir -p results/seq
PERSONS="chb01 chb02 chb03 chb04 chb05 chb06 chb07 chb08 chb09 chb10 chb11 chb12 chb13 chb14 chb15 chb16 chb17 chb18 chb19 chb20 chb22 chb23 chb24"
echo $PERSONS | tr ' ' '\n' | xargs -P 4 -I{} sh -c \
  '[ -f results/seq/{}.json ] && grep -q qlstm_twin results/seq/{}.json && exit 0; ~/qbe-venv/bin/python -u train_seq.py --person {} >> results/seq/{}.log 2>&1'
~/qbe-venv/bin/python analyze_seq.py | tee results/seq_analysis.txt
