"""One fixed AUG arm, sharing the unchanged original warmup training loop."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import torch
import numpy as np
from augment import AugmentReader,prepare_cache,sampler_seed

HERE=Path(__file__).resolve().parent; BASE=HERE.parent; OLD=BASE/'cross_user_meta/runs_s2026'
spec=importlib.util.spec_from_file_location('_mild_time_original_trainer',BASE/'cross_user_meta/trainer.py')
T=importlib.util.module_from_spec(spec); sys.modules[spec.name]=T; spec.loader.exec_module(T)
original_epoch=T.warmup_epoch

def mild_epoch(model,optimizer,sampler,reader,labels,criterion,config):
    seed=sampler_seed(copy.deepcopy(sampler.state_dict()),config['augmentation_fold'])
    reader.configure(seed)
    row=original_epoch(model,optimizer,sampler,reader,labels,criterion,config)
    row.update(augmentation_seed=seed,augmentation_trace=reader.trace.hexdigest(),
               augmentation_counts={str(k):v for k,v in reader.counts.items()})
    return row

def verify_old(source,config):
    old=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    if old['source']!=source: raise ValueError('Original code/data/runtime differs; cannot reuse controls')
    if old['config']!=config: raise ValueError('Original configuration differs')
    environment=json.loads((OLD/'ENVIRONMENT.json').read_text(encoding='utf-8'))
    numpy_same=environment['packages']['numpy']==np.__version__
    ledger=[]
    for fold in range(1,9):
        p=OLD/f'fold{fold}/warmup/result.json'; r=json.loads(p.read_text(encoding='utf-8'))
        if r['identity']['source']!=source or r['identity']['config']!=config:
            raise ValueError('Control run source/config mismatch')
        for name,h in r['files'].items():
            if T.protocol.digest(p.parent/name)!=h: raise ValueError('Control artifacts changed')
        ledger.append({'fold':fold,'path':str(p),'result_sha256':T.protocol.digest(p),
                       'initial_state_sha256':r['identity']['initial_state_sha256'],
                       'selected_state_sha256':r['selected_state_sha256'],
                       'best_epoch':r['best_epoch'],'files':r['files']})
    return {'controls_reused':False,'matched_original_source':True,'matched_config':True,
            'numpy_version':np.__version__,'old_numpy_version':environment['packages']['numpy'],
            'numpy_matches':numpy_same,'decision':'Rerun CONTROL and AUG in current runtime; old runs are provenance references only',
            'controls':ledger}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--mode',choices=['CONTROL','AUG','PAIRED'],default='PAIRED')
    ap.add_argument('--folds',nargs='+',type=int,default=list(range(1,9)))
    ap.add_argument('--out',type=Path,default=HERE/'runs')
    args=ap.parse_args(); args.out.mkdir(exist_ok=True)
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    old=json.loads((OLD/'manifest.json').read_text(encoding='utf-8'))
    source=T.source_identity(); config=copy.deepcopy(old['config'])
    ledger=verify_old(source,config)
    cache,cachemeta=prepare_cache()
    source['augmentation']=cachemeta
    for p in [Path(__file__),HERE/'augment.py',BASE/'pipeline.py',BASE/'vendor/uwb_osr/data.py']:
        source['code'][str(p)]=T.protocol.digest(p)
    config.update(augmentation={'mode':args.mode,'probability':.5,'raw_warp_amounts':[-2,2],
                                'derive_phase_after_warp':True,'seed_policy':'SHA256 of source sampler state/fold/fixed seed20260908'})
    if args.smoke:
        config.update(warmup_epochs=1,warmup_min=1,epochs=1,min_epochs=1,smoke_steps=3,smoke=True)
    T.freeze_manifest(args.out,source,config)
    T.protocol.dump(args.out/'CONTROL_REUSE.json',ledger)
    T.warmup_epoch=mild_epoch
    try:
        for fold in args.folds:
            for mode in (['CONTROL','AUG'] if args.mode=='PAIRED' else [args.mode]):
                foldconfig=dict(config,augmentation_fold=fold,active_mode=mode)
                reader,y,s,parts=T.protocol.open_fold(fold)
                matrix,meta=T.smoothing.prepare_matrix(fold,OLD/'matrices',source['data'])
                reader=AugmentReader(reader,cache,mode=='AUG')
                root=args.out/mode
                T.protocol.dump(root/f'fold{fold}'/'split.json',{'ids':{k:v.tolist() for k,v in parts.items()},
                                    'training_subjects':np.unique(s[parts['train']]).tolist(),'val':fold%8+1,'test':fold})
                T.protocol.dump(args.out/'queue_status.json',{'state':'running','fold':fold,'mode':mode,'folds':args.folds})
                result=T.run_arm(root,fold,'warmup',None,reader,y,s,parts,matrix,meta,source,foldconfig)
                if result['identity']['initial_state_sha256']!=ledger['controls'][fold-1]['initial_state_sha256']:
                    raise ValueError('New arm initialization differs from control')
    except BaseException as error:
        T.protocol.dump(args.out/'queue_status.json',{'state':'failed','error':str(error)})
        raise
    T.protocol.dump(args.out/'queue_status.json',{'state':'complete','folds':args.folds,'mode':args.mode,'smoke':args.smoke})
    print('MILD_TIME_QUEUE_COMPLETE',flush=True)

if __name__=='__main__': main()
