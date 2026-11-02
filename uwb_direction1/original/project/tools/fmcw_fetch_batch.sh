#!/bin/bash
# Sequential batch fetch of nominal (e1) members for LOSO experiments.
# Each member ~650-750 MB compressed -> 2.4 GB npz. Resume-safe.
PY=D:/D_conda/envs/pytorch/python.exe
OUT=D:/uwb_gestures/fmcw_frame_labeled
for u in 4 5 6 7 8 9 10 11 12 2 1; do
  m="fulldataset/user${u}_e1.npz"
  if [ -f "$OUT/user${u}_e1.npz" ]; then echo "skip user$u (exists)"; continue; fi
  echo "=== fetching user$u at $(date +%T) ==="
  "$PY" tools/fmcw_zenodo_fetch.py fetch "$m" "$OUT" || echo "FAILED user$u"
done
echo "BATCH COMPLETE"
