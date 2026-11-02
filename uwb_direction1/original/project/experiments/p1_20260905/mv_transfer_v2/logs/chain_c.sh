#!/bin/sh
# Wait for the seed-2027 queue to exit, then run the method C diagnostic.
PY="/d/D_conda/envs/pytorch/python.exe"
while pgrep -f "train_student.py --arms A-US A-MV" >/dev/null 2>&1; do sleep 20; done
"$PY" predictability.py --folds 1 2 3 4 5 6 7 8 >> logs/predictability.log 2>&1
echo C_DIAGNOSTIC_DONE
