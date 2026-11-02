"""Single-clip CPU architecture timing, separate from accuracy evaluation.

Original model weights are unavailable, so its random weights measure only
architecture cost. New checkpoint is actual fold1 legacy. No device/energy,
collection, clutter removal, or original trained-model latency claim.
"""
import argparse
import importlib.util
import json
from numbers import Integral
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

from audit_restored import digest,dump,expected_parts
from benchmark_cpu_encoder import extract
from circular_signal import SignalNet,encode,get_batch


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ('data','source','runs','benchmark','out'):ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Use a new output path')
    if json.loads((args.runs/'queue_status.json').read_text())['state']!='complete':
        raise ValueError('Wait for training to finish; avoid known training contention')
    torch.set_num_threads(3);torch.use_deterministic_algorithms(True);torch.manual_seed(2026)
    base=args.source/'experiments/p1_20260905'
    module=base/'cross_user_meta/model.py'
    spec=importlib.util.spec_from_file_location('_original_cpu_latency',module)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    ns={'torch':torch,'nn':nn,'np':np,'Integral':Integral,
        'SUBSETS':((0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2))}
    extract(args.source/'experiments/new_routes/multiview.py',{'SubsetModel','physical_indices'},ns)
    original=ns['SubsetModel'](old.Classifier([2],'concat','mean')).eval()
    funcs={'np':np}
    extract(base/'vendor/uwb_osr/data.py',{'_analytic','doppler_map'},funcs)
    extract(base/'pipeline.py',{'norm','transform'},funcs)
    small=SignalNet().eval()
    checkpoint=args.runs/'fold1/legacy/best.pt'
    small.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
    ids=np.array(json.loads(args.benchmark.read_text())['source_ids'][:8])
    y=np.load(args.data/'cache/y_clutterremoved.npy');users=np.load(args.data/'cache/subj_clutterremoved.npy')
    assert np.isin(ids,expected_parts(y,users,1)['train']).all()
    raw=np.array(np.load(args.data/'cache/X_clutterremoved.npy',mmap_mode='r')[ids],copy=True)
    def prepare_old(x):
        a=funcs['transform'](x,'off').astype('float16')
        b=funcs['transform'](x,'only').astype('float16')
        return torch.from_numpy(np.concatenate((a,b),axis=-1).astype('float32'))
    def prepare_new(x):
        # Full six-channel preparation matches deployed research implementation,
        # even though legacy only uses two of those channels. No optimized timing.
        return get_batch(encode(x),np.arange(len(x)),'legacy')
    old_x=[prepare_old(raw[i:i+1]) for i in range(8)]
    new_x=[prepare_new(raw[i:i+1]) for i in range(8)]
    funcs_run={'original_forward':lambda i:original(old_x[i],(0,1,2)),
               'compact_forward':lambda i:small(new_x[i]),
               'original_preprocess_and_forward':lambda i:original(prepare_old(raw[i:i+1]),(0,1,2)),
               'compact_preprocess_and_forward':lambda i:small(prepare_new(raw[i:i+1]))}
    timings={k:[] for k in funcs_run}
    for i in range(8):
        for fn in funcs_run.values():fn(i)
    for repeat in range(5):
        for i in range(8):
            # Alternating order limits a fixed warm-cache ordering bias.
            names=list(funcs_run)
            if (repeat+i)%2:names.reverse()
            for name in names:
                start=time.perf_counter();out=funcs_run[name](i)
                assert out.shape==(1,12) and torch.isfinite(out).all()
                timings[name].append((time.perf_counter()-start)*1000)
    result={'state':'measured','device':'cpu','torch':torch.__version__,'threads':3,'batch':1,
        'warmups_per_mode':8,'trials_per_mode':40,'source_sample_ids':ids.tolist(),
        'original_weights':'random initialization; architecture timing only, not old trained-model reproduction',
        'compact_weights':'actual fold1 legacy checkpoint','compact_checkpoint_sha256':digest(checkpoint),
        'original_parameters':sum(p.numel() for p in original.parameters()),
        'compact_parameters':sum(p.numel() for p in small.parameters()),
        'milliseconds':{k:{'median':float(np.median(v)),'p90':float(np.quantile(v,.9)),'trials':v} for k,v in timings.items()},
        'excludes':['collection','file IO','static clutter removal','model loading'],
        'notes':'de-cluttered source clips already in RAM; local CPU microbenchmark; not an embedded-device energy or throughput claim',
        'script_sha256':digest(__file__)}
    dump(args.out,result)
    print(json.dumps({k:v for k,v in result.items() if k!='milliseconds'}))
    print(json.dumps({k:{s:v for s,v in d.items() if s!='trials'} for k,d in result['milliseconds'].items()}))


if __name__=='__main__':main()
