"""Canonical label-free cache recovery, fixed to existing byte-level identities.

The Torch loop below was executed successfully during 2026-11-03 recovery.
This convenience wrapper was compiled, not re-run against the active cache.
Never fits a classifier or reads y/labels.
"""
from pathlib import Path
import argparse,hashlib
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for data in iter(lambda:f.read(4194304),b''):h.update(data)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--data',default='/workspace/uwb-restored-20261103');a.add_argument('--ids',default='/workspace/uwb-accuracy-loop-20261029/run01/source_ids.npz');a.add_argument('--out',default='/workspace/uwb-timemil-20261029/run02/source_maps.npy');p=a.parse_args()
 expected='ad38801896b65e676569c379eb03750ed61ecbf920557e3e78de351216683a57'
 out=Path(p.out)
 if out.exists():
  assert sha(out)==expected,'Existing cache mismatch; preserve and inspect, do not overwrite'
  print('Existing canonical cache SHA verified; no writes or fitting');return
 assert sha(p.ids)=='d932f8910326ebf9c114c36e90026e5bf1cf8fdc71233816c9b389dabe3bdc50'
 rawpath=Path(p.data)/'cache/X_clutterremoved.npy'
 assert sha(rawpath)=='3004e7a2aa44cb4ffd763a6ef3bf30c85d2b7dc0499140c195c48fa3ed6ce519'
 import torch,numpy as np
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 with np.load(p.ids) as z:ids=z['ids'];users=z['users']
 assert ids.shape==users.shape==(6900,) and len(np.unique(ids))==6900
 assert set(users.tolist())==set(range(1,7))
 raw=np.load(rawpath,mmap_mode='r');assert raw.shape==(9593,3,90,189) and raw.dtype==np.float32
 out.parent.mkdir(parents=True,exist_ok=True)
 maps=np.lib.format.open_memmap(out,mode='w+',dtype='float32',shape=(6900,90,567))
 with torch.no_grad():
  for start in range(0,6900,64):
   x=torch.from_numpy(np.array(raw[ids[start:start+64]],copy=True))
   x=x/x.square().mean((2,3),keepdim=True).sqrt().clamp_min(1e-8)
   maps[start:start+64]=x.permute(0,2,1,3).reshape(len(x),90,567).numpy()
 maps.flush();del maps
 assert sha(out)==expected,'Recovered cache mismatch retained for inspection'
 print('All6900 canonical maps recovered with exact original SHA; no labels or fitting')
if __name__=='__main__':main()
