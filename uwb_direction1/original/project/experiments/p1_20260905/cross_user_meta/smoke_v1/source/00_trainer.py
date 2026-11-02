"""Sequential six-user warmup and matched exact-meta A0--A3 experiments.

Only the selected Right classifier is deployed. Full epoch checkpoints restore
optimizer, sampler and all RNG states; every resume is source/config bound.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location('_cross_meta_training_'+name, HERE/(name+'.py'))
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


protocol, state, models, meta, smoothing = [load(n) for n in ('protocol','train_state','model','meta_step','smoothing')]


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def source_identity():
    paths = [HERE/(n+'.py') for n in ('trainer','train_state','protocol','model','meta_step','smoothing')]
    paths += [protocol.BASE/'train.py', protocol.BASE/'vendor/config.py', protocol.BASE/'vendor/uwb_osr/model.py']
    return {'code': {str(p): protocol.digest(p) for p in paths}, 'data': smoothing.data_identity(),
            'runtime': {'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name()}}


def freeze_manifest(out, source, config):
    expected = {'source': source, 'config': config}
    path = out/'manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != expected:
            raise ValueError('Output manifest differs from current sources/configuration')
    else:
        out.mkdir(parents=True, exist_ok=True)
        for i, source_path in enumerate(source['code']):
            p = Path(source_path); target = out/'source'/f'{i:02d}_{p.name}'
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
        protocol.dump(path, expected)


@torch.no_grad()
def evaluate(model, reader, labels, ids, role, batch=32):
    model.eval(); logits = []
    for start in range(0, len(ids), batch):
        rows = ids[start:start+batch]
        x = reader.load_ids(rows, role, 'R')
        with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
            value = model(x)
        logits.append(value.float().cpu().numpy())
    value = np.concatenate(logits)
    if not np.isfinite(value).all(): raise ValueError('Nonfinite evaluation logits')
    ce = float(F.cross_entropy(torch.from_numpy(value), torch.from_numpy(labels[ids]).long()))
    return value, ce


def verified_completed(directory, identity):
    path = directory/'result.json'
    if not path.exists(): return None
    result = json.loads(path.read_text(encoding='utf-8'))
    if result['identity'] != identity: raise ValueError('Completed result identity mismatch')
    for name, digest in result['files'].items():
        if protocol.digest(directory/name) != digest: raise ValueError(f'Completed artifact changed: {directory/name}')
    print(f'VERIFIED_SKIP fold={identity["fold"]} arm={identity["arm"]}', flush=True)
    return result


def optimizer_for(model, arm, config):
    if arm == 'warmup':
        return torch.optim.Adam(model.right.parameters(), lr=config['warmup_lr'], weight_decay=config['weight_decay'])
    return torch.optim.Adam([
        {'params': model.right.encoder.parameters(), 'lr': config['continuation_lr']},
        {'params': model.right.head.parameters(), 'lr': config['continuation_lr']},
        {'params': model.fuser.parameters(), 'lr': config['aux_lr']}], weight_decay=config['weight_decay'])


def finite_step(model, optimizer, loss):
    if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
    loss.backward()
    # A nonfinite gradient is a failed run, never a silently skipped update.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
    optimizer.step()
    return float(norm)


def warmup_epoch(model, optimizer, sampler, reader, labels, criterion, config):
    model.train(); ids = sampler.next_epoch(); trace = hashlib.sha256(); total = 0.; rows_seen = 0; steps = 0
    limit = config['smoke_steps']
    for start in range(0, len(ids), config['warmup_batch']):
        if limit and steps >= limit: break
        rows = ids[start:start+config['warmup_batch']]; trace.update(rows.tobytes())
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.
        for k in range(0, len(rows), config['warmup_microbatch']):
            micro = rows[k:k+config['warmup_microbatch']]
            x = reader.load_ids(micro, 'train', 'R'); y = torch.as_tensor(labels[micro], device='cuda').long()
            with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
                loss = criterion(model(x), y) * (len(micro)/len(rows))
            if not torch.isfinite(loss): raise ValueError('Nonfinite warmup loss')
            loss.backward(); loss_value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        optimizer.step(); total += loss_value*len(rows); rows_seen += len(rows); steps += 1
    return {'train_loss': total/rows_seen, 'steps': steps, 'episode_hash': trace.hexdigest(), 'source_rows': rows_seen}


def continuation_epoch(model, optimizer, sampler, reader, labels, criterion, arm, config):
    model.train(); trace = hashlib.sha256(); total = 0.; counts = {}; diagnostic = {'support_R':0., 'support_MV':0., 'query':0.}
    steps = config['smoke_steps'] or math.ceil(len(reader.partitions['train'])/(6*config['per_user']))
    for _ in range(steps):
        ep = sampler.next(); counts[str(ep['query_subject'])] = counts.get(str(ep['query_subject']), 0)+1
        trace.update(ep['support_ids'].tobytes()); trace.update(ep['query_ids'].tobytes())
        sx = reader.load_ids(ep['support_ids'], 'train', 'MV' if arm in ('A2','A3') else 'R')
        qx = reader.load_ids(ep['query_ids'], 'train', 'R')
        sy = torch.as_tensor(labels[ep['support_ids']],device='cuda').long()
        qy = torch.as_tensor(labels[ep['query_ids']],device='cuda').long()
        optimizer.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16):
            loss, values = meta.episode_loss(model,sx,sy,qx,qy,arm,criterion,config['alpha'],config['mv_weight'])
        finite_step(model,optimizer,loss); total += float(loss.detach())
        for key, value in values.items(): diagnostic[key] += float(value) if value is not None else 0.
        del loss, values, sx, qx, sy, qy
    return {'train_loss': total/steps, 'steps': steps, 'episode_hash': trace.hexdigest(),
            'query_subject_counts': counts, 'loss_components': {k:v/steps for k,v in diagnostic.items()}}


def run_arm(out, held, arm, initial_path, reader, labels, subjects, parts, matrix, matrix_meta, source, config):
    directory = out/f'fold{held}'/arm; directory.mkdir(parents=True, exist_ok=True)
    seed_all(config['seed'])
    model = models.build_model(config['seed'])
    if initial_path is not None:
        initial = torch.load(initial_path, map_location='cpu', weights_only=False)
        model.load_state_dict(initial['state'], strict=True); del initial
    initial_hash = state.state_hash(model.state_dict())
    identity = {'config': config, 'source': source, 'fold': held, 'arm': arm,
                'matrix_sha256': matrix_meta['matrix_sha256'],
                'partitions': {k:protocol.array_hash(v) for k,v in parts.items()},
                'initial_state_sha256': initial_hash}
    finished = verified_completed(directory, identity)
    if finished is not None: return finished
    model = model.cuda(); optimizer = optimizer_for(model, arm, config)
    matrix_gpu = torch.as_tensor(matrix,device='cuda')
    criterion = lambda logits, y: smoothing.classification_loss(logits, y, matrix_gpu)
    if arm == 'warmup':
        sampler = state.ShuffleSampler(parts['train'], config['seed']+held)
        maximum, minimum, patience = (config[k] for k in ('warmup_epochs','warmup_min','warmup_patience'))
    else:
        sampler = protocol.EpisodeSampler(parts['train'],labels,subjects,config['episode_seed']+held,config['per_user'])
        maximum, minimum, patience = (config[k] for k in ('epochs','min_epochs','patience'))
    progress = directory/'progress.pt'
    if progress.exists():
        info = state.load_progress(progress,model,optimizer,sampler,identity)
        selection = state.Selection.from_state(info['selection']); history = info['history']; epoch = info['epoch']
        if protocol.digest(directory/'best.pt') != info['best_file_sha256']:
            # The best file can be one epoch ahead after a crash between saves.
            # progress embeds its matching best state, so restore that transaction.
            saved = torch.load(progress,map_location='cpu',weights_only=False)
            state.atomic_save(directory/'best.pt',saved['info']['best_checkpoint'])
            del saved
        print(f'RESUME fold={held} arm={arm} epoch={epoch}',flush=True)
    else:
        seed_all(config['seed']+100000)
        _, ce = evaluate(model,reader,labels,parts['val'],'val',config['eval_batch'])
        selection = state.Selection(ce); epoch=0
        history=[{'epoch':0,'val_ce':ce,'train_loss':None,'episode_hash':None,'steps':0,'seconds':0.}]
        state.atomic_save(directory/'best.pt',{'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':0,'val_ce':ce})
        save_epoch(progress,directory,model,optimizer,sampler,identity,selection,history,0)
    print(f'RUN fold={held} arm={arm} start_epoch={epoch} max_epochs={maximum}',flush=True)
    stop_reason = 'max_epochs'
    while epoch < maximum:
        if selection.should_stop(epoch,minimum,patience):
            stop_reason='patience'; break
        started=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        epoch += 1
        if arm == 'warmup': row=warmup_epoch(model,optimizer,sampler,reader,labels,criterion,config)
        else: row=continuation_epoch(model,optimizer,sampler,reader,labels,criterion,arm,config)
        _, ce=evaluate(model,reader,labels,parts['val'],'val',config['eval_batch'])
        row.update(epoch=epoch,val_ce=ce,seconds=time.perf_counter()-started,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated())
        history.append(row)
        if selection.observe(epoch,ce):
            state.atomic_save(directory/'best.pt',{'state':{k:v.detach().cpu() for k,v in model.state_dict().items()},'epoch':epoch,'val_ce':ce})
        save_epoch(progress,directory,model,optimizer,sampler,identity,selection,history,epoch)
        protocol.dump(directory/'status.json',{'fold':held,'arm':arm,'epoch':epoch,'best_epoch':selection.best_epoch,
                                              'best_val_ce':selection.best_ce,'last':row,'state':'running'})
        print(f'EPOCH fold={held} arm={arm} epoch={epoch} val_ce={ce:.6f} best={selection.best_epoch} seconds={row["seconds"]:.1f}',flush=True)
    if selection.should_stop(epoch,minimum,patience): stop_reason='patience'
    best=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(best['state'],strict=True)
    val_logits, val_ce=evaluate(model,reader,labels,parts['val'],'val',config['eval_batch'])
    if abs(val_ce-selection.best_ce)>1e-7: raise ValueError('Selected validation CE replay differs')
    logits,_=evaluate(model,reader,labels,parts['test'],'test',config['eval_batch'])
    state.atomic_save(directory/'right.pt',{k:v.detach().cpu() for k,v in model.right.state_dict().items()})
    with (directory/'predictions.npz.tmp').open('wb') as f:
        np.savez_compressed(f,logits=logits,y=labels[parts['test']],ids=parts['test'],
                            train_ids=parts['train'],val_ids=parts['val'],test_ids=parts['test'],
                            val_logits=val_logits,val_y=labels[parts['val']])
    (directory/'predictions.npz.tmp').replace(directory/'predictions.npz')
    result={'identity':identity,'best_epoch':selection.best_epoch,'best_val_ce':selection.best_ce,
            'selected_state_sha256':state.state_hash(best['state']),'history':history,'stop_reason':stop_reason,
            'max_epochs':maximum,'min_epochs':minimum,'patience':patience,
            'accuracy':float(np.mean(logits.argmax(1)==labels[parts['test']])),
            'training_subjects':np.unique(subjects[parts['train']]).tolist(),
            'validation_subject':held%8+1,'test_subject':held,
            'files':{n:protocol.digest(directory/n) for n in ('best.pt','right.pt','predictions.npz','progress.pt')}}
    protocol.dump(directory/'result.json',result)
    protocol.dump(directory/'status.json',{'state':'complete','fold':held,'arm':arm,'best_epoch':selection.best_epoch})
    print(f'COMPLETE fold={held} arm={arm} selected_epoch={selection.best_epoch}',flush=True)
    del best, model, optimizer, matrix_gpu; gc.collect(); torch.cuda.empty_cache()
    return result


def save_epoch(progress,directory,model,optimizer,sampler,identity,selection,history,epoch):
    # Embedding the selected state makes progress the single recovery transaction.
    best=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    info={'identity':identity,'selection':selection.state_dict(),'history':history,'epoch':epoch,
          'best_file_sha256':protocol.digest(directory/'best.pt'),'best_checkpoint':best}
    state.save_progress(progress,model,optimizer,sampler,info)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=HERE/'runs_s2026')
    parser.add_argument('--folds',type=int,nargs='+',default=list(range(1,9)))
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    config={'seed':args.seed,'episode_seed':7777,'per_user':4,'alpha':1e-4,'mv_weight':.5,
            'warmup_lr':1e-4,'continuation_lr':1e-5,'aux_lr':1e-4,'weight_decay':1e-4,
            'warmup_batch':128,'warmup_microbatch':32,'eval_batch':32,
            'warmup_epochs':1 if args.smoke else 40,'warmup_min':1 if args.smoke else 12,'warmup_patience':8,
            'epochs':1 if args.smoke else 16,'min_epochs':1 if args.smoke else 6,'patience':5,
            'smoke_steps':3 if args.smoke else 0,'smoke':args.smoke,
            'precision':'float32 weights; float16 inputs; bfloat16 autocast','attention':'MATH',
            'criterion':'CE + 0.5 KL(T=2, target=0.5*(onehot+onehot@matrix))',
            'warmup_initial_candidate':True,'continuation_initial_candidate':True}
    source=source_identity(); freeze_manifest(args.out,source,config)
    for held in args.folds:
        reader,labels,subjects,parts=protocol.open_fold(held)
        matrix,matrix_meta=smoothing.prepare_matrix(held,args.out/'matrices',source['data'])
        protocol.dump(args.out/f'fold{held}'/'split.json',{'ids':{k:v.tolist() for k,v in parts.items()},
                     'training_subjects':np.unique(subjects[parts['train']]).tolist(),'val':held%8+1,'test':held})
        for arm in ('warmup','A0','A1','A2','A3'):
            protocol.dump(args.out/'queue_status.json',{'state':'running','fold':held,'arm':arm,'folds':args.folds})
            initial=None if arm=='warmup' else args.out/f'fold{held}'/'warmup'/'best.pt'
            run_arm(args.out,held,arm,initial,reader,labels,subjects,parts,matrix,matrix_meta,source,config)
    protocol.dump(args.out/'queue_status.json',{'state':'complete','folds':args.folds,'smoke':args.smoke})
    print('QUEUE_COMPLETE',flush=True)


if __name__=='__main__': main()
