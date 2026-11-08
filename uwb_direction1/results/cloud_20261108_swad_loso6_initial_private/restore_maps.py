"""Recreate ONLY a missing deterministic RMS input cache; never train or overwrite data."""
from pathlib import Path
import argparse,hashlib,json,os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def main():
    a=argparse.ArgumentParser();a.add_argument('--raw',required=True);a.add_argument('--ids',required=True);a.add_argument('--out',required=True);a.add_argument('--materialize-missing',action='store_true');args=a.parse_args()
    expected='ad38801896b65e676569c379eb03750ed61ecbf920557e3e78de351216683a57';out=Path(args.out)
    if out.exists():assert sha(out)==expected;print(json.dumps(dict(state='existing_cache_exact_no_write',sha256=expected)));return
    assert args.materialize_missing,'Explicit missing-cache materialization required'
    p=Path('/workspace/uwb-irm-loso6-20261107/PHASE_IDENTITY.json')
    if p.exists():
        old=json.loads(p.read_text());proc=Path('/proc')/str(old['pid'])/'stat'
        if proc.exists():
            q=proc.read_text();q=q[q.rfind(')')+2:].split();boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            assert q[0]=='Z' or boot!=old['boot_id'] or int(q[19])!=old['startticks'],'Active original phase: do not rebuild its inputs'
    assert sha(args.raw)=='3004e7a2aa44cb4ffd763a6ef3bf30c85d2b7dc0499140c195c48fa3ed6ce519'
    assert sha(args.ids)=='d932f8910326ebf9c114c36e90026e5bf1cf8fdc71233816c9b389dabe3bdc50'
    import numpy as np
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    with np.load(args.ids) as z:ids=z['ids'];users=z['users']
    assert len(ids)==6900 and set(users)==set(range(1,7))
    raw=np.load(args.raw,mmap_mode='r');assert raw.shape==(9593,3,90,189) and raw.dtype==np.float32
    out.parent.mkdir(parents=True,exist_ok=True);maps=np.lib.format.open_memmap(out,mode='w+',dtype='float32',shape=(6900,90,567))
    for b in range(0,6900,64):
        x=torch.from_numpy(np.array(raw[ids[b:b+64]],copy=True))
        x=x/x.square().mean((2,3),keepdim=True).sqrt().clamp_min(1e-8)
        maps[b:b+64]=x.permute(0,2,1,3).reshape(len(x),90,567).numpy()
    maps.flush();del maps
    assert sha(out)==expected,'Rebuild differs: preserve output for inspection; never accept a changed cache silently'
    print(json.dumps(dict(state='missing_cache_rebuilt_and_exact_SHA_verified',sha256=expected,gesture_labels_read=False,new_models_trained=False)))
if __name__=='__main__':main()
