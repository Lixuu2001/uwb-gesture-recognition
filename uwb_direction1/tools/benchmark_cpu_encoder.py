"""Measure original three-view encoder training on real SOURCE rows, CPU only.

This is a cost/gradient probe, not a classifier-quality experiment. No historical
checkpoint is loaded, and no validation/test examples enter the optimizer.
"""
import argparse
import ast
import importlib.util
import json
from numbers import Integral
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from audit_restored import digest, dump, expected_parts


def extract(path, names, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes) == len(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--batch',type=int,default=8)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Use a new benchmark output path')
    torch.set_num_threads(3)
    torch.manual_seed(2026)
    base=args.source/'experiments/p1_20260905'
    module_path=base/'cross_user_meta/model.py'
    spec=importlib.util.spec_from_file_location('_cpu_cost_original_model',module_path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
    ns={'torch':torch,'nn':nn,'np':np,'Integral':Integral,
        'SUBSETS':((0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2))}
    multi=args.source/'experiments/new_routes/multiview.py'
    extract(multi,{'SubsetModel','physical_indices'},ns)
    right=old.Classifier([2],'concat','mean')
    model=ns['SubsetModel'](right).cpu().train()
    del right
    transform_ns={'np':np}
    vendor=base/'vendor/uwb_osr/data.py'
    pipeline=base/'pipeline.py'
    extract(vendor,{'_analytic','doppler_map'},transform_ns)
    extract(pipeline,{'norm','transform'},transform_ns)
    y=np.load(args.data/'cache/y_clutterremoved.npy')
    users=np.load(args.data/'cache/subj_clutterremoved.npy')
    train=expected_parts(y,users,1)['train']
    ids=np.random.default_rng(2026).choice(train,args.batch,replace=False)
    raw=np.load(args.data/'cache/X_clutterremoved.npy',mmap_mode='r')
    started=time.perf_counter()
    # Preserve historical float16 storage rounding before converting to CPU float32.
    amp=transform_ns['transform'](raw[ids],'off').astype('float16')
    phase=transform_ns['transform'](raw[ids],'only').astype('float16')
    x=torch.from_numpy(np.concatenate((amp,phase),axis=-1).astype('float32'))
    target=torch.from_numpy(y[ids]).long()
    preprocess_seconds=time.perf_counter()-started
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
    times,losses=[],[]
    for step in range(4):
        started=time.perf_counter()
        logits=model(x,(0,1,2))
        loss=F.cross_entropy(logits,target)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grads=[p.grad for p in model.encoder.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads)>0
        optimizer.step()
        times.append(time.perf_counter()-started);losses.append(loss.item())
        print(json.dumps({'step':step,'seconds':times[-1],'source_loss':losses[-1]}),flush=True)
    median=float(np.median(times[1:]))
    record={'state':'benchmark_complete','scope':'four optimizer steps on fixed source batch; no quality claim',
            'source_sample_ids':ids.tolist(),'source_subjects':users[ids].tolist(),
            'fold':1,'test_user':1,'validation_user':2,'batch':args.batch,'torch':torch.__version__,
            'threads':3,'device':'cpu','input_shape':list(x.shape),
            'parameters':sum(p.numel() for p in model.parameters()),
            'preprocessing_seconds':preprocess_seconds,'training_step_seconds':times,
            'median_measured_step_seconds_excluding_first':median,
            'rough_source_pass_seconds_at_this_batch':median*np.ceil(len(train)/args.batch),
            'source_pass_estimate_excludes_validation_and_io':True,
            'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'finite_nonzero_encoder_gradients':True,'losses_on_fixed_training_batch':losses,
            'source_sha256':{str(p):digest(p) for p in [module_path,multi,vendor,pipeline,base/'train.py',base/'vendor/uwb_osr/model.py',base/'vendor/config.py']},
            'script_sha256':digest(__file__)}
    args.out.parent.mkdir(parents=True,exist_ok=True);dump(args.out,record)
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()
