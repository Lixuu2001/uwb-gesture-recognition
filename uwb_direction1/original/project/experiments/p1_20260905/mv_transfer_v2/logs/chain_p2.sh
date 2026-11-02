#!/bin/sh
# Wait for the stage P1 queue to exit, then run the follow-up batch.
PY="/d/D_conda/envs/pytorch/python.exe"
while pgrep -f "train_student.py --arms B-GS" >/dev/null 2>&1; do sleep 20; done
"$PY" train_student.py --arms A-TS --folds 1 2 3 4 5 6 7 8 --model-seed 2026 >> logs/p2_ats.log 2>&1
"$PY" train_student.py --arms A-US A-MV B-SELF B-MV A-SC A-GS B-GS --folds 1 2 3 4 5 6 7 8 --model-seed 2027 >> logs/p2_seed2027.log 2>&1
echo CHAIN_DONE
