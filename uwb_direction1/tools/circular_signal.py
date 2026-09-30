"""CPU signal-level controls for wrapped versus circular analytic increments.

The analytic signal is constructed from real RF frames, not calibrated complex
radar I/Q. No physical velocity or unwrapping claim is made. All preprocessing
is per clip and label-free; the trainable CNN is initialized from scratch.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from audit_restored import digest, dump, expected_parts, metrics
from full_view_heads import atomic_save, array_hash

CHANNELS = {'legacy': (0, 2, -1, -1), 'angle_control': (0, 1, 2, 3),
            'circular': (0, 1, 4, 5)}


def analytic(x):
    n = x.shape[-1]
    h = np.zeros(n)
    h[0] = 1
    h[1:(n+1)//2] = 2
    if n % 2 == 0:
        h[n//2] = 1
    return np.fft.ifft(np.fft.fft(x.astype(np.float64), axis=-1)*h, axis=-1)


def increment_channels(d):
    """Common envelope, weighted angle, its absolute value, real/imag phasor.

    Exact zeros remain zero. Dividing d by sqrt(abs(d)) gives the same
    amplitude weight sqrt(abs(d)) as the original angle representation.
    """
    w = np.sqrt(np.abs(d))
    angle = w*np.angle(d)
    unit_weighted = np.divide(d, w, out=np.zeros_like(d), where=w > 0)
    return np.stack((w, angle, np.abs(angle), unit_weighted.real,
                     unit_weighted.imag), axis=-3)


def encode(raw):
    """[N,3,90,189] -> [N,3,6,90,63], normalization before range pooling.

    Channels: standardized abs RF, envelope, weighted angle, unsigned angle,
    weighted cosine, weighted sine. Historical first-increment repetition is
    retained. No fitting across records and no temporal resampling.
    """
    if raw.ndim != 4 or raw.shape[1:] != (3, 90, 189):
        raise ValueError('Expected [N,3,90,189] real radar frames')
    z = analytic(raw)
    z -= z.mean(axis=-2, keepdims=True)
    d = z[..., 1:, :]*z[..., :-1, :].conj()
    inc = increment_channels(d).astype(np.float32)
    inc = np.concatenate((inc[..., :1, :], inc), axis=-2)
    features = np.concatenate((np.abs(raw).astype(np.float32)[:, :, None], inc), axis=2)
    features = (features-features.mean(axis=(-2,-1), keepdims=True))/(features.std(axis=(-2,-1), keepdims=True)+1e-6)
    # Match original float16 storage rounding before fixed range averaging.
    features = features.astype(np.float16).astype(np.float32)
    return features.reshape(*features.shape[:-1],63,3).mean(-1).astype(np.float16)


def get_batch(cache, ids, arm):
    x = np.asarray(cache[ids], dtype=np.float32)
    channels = CHANNELS[arm]
    out = np.stack([x[:, :, c] if c >= 0 else np.zeros_like(x[:, :, 0])
                    for c in channels], axis=2)
    return torch.from_numpy(out)


class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(cin,cout,3,stride=2,padding=1,bias=False),
            nn.GroupNorm(4,cout), nn.GELU(),
            nn.Conv2d(cout,cout,3,padding=1,bias=False), nn.GroupNorm(4,cout), nn.GELU())
    def forward(self,x):
        return self.layers(x)


class SignalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(Block(4,16),Block(16,24),Block(24,48),
                                     nn.AdaptiveAvgPool2d((3,4)))
        self.head = nn.Sequential(nn.Flatten(),nn.Linear(3*48*3*4,128),
                                  nn.GELU(),nn.Dropout(.2),nn.Linear(128,12))
    def forward(self,x):
        n,v,c,t,r = x.shape
        if (v,c,t,r) != (3,4,90,63):
            raise ValueError('Expected [N,3,4,90,63]')
        return self.head(self.encoder(x.reshape(n*v,c,t,r)).reshape(n,v,-1))


def source_order(ids, seed, held, epoch):
    return np.random.default_rng(seed+10000*held+100000*epoch).permutation(ids)


@torch.no_grad()
def predict(model, cache, ids, arm, batch):
    model.eval()
    return torch.cat([model(get_batch(cache,ids[i:i+batch],arm))
                      for i in range(0,len(ids),batch)])


def prepare(data, out):
    out.mkdir(parents=True,exist_ok=True)
    path = data/'cache/X_clutterremoved.npy'
    raw_sha = digest(path)
    if raw_sha != '3004e7a2aa44cb4ffd763a6ef3bf30c85d2b7dc0499140c195c48fa3ed6ce519':
        raise ValueError('Unexpected original raw cache hash')
    identity = {'raw_sha256':raw_sha,'script_sha256':digest(__file__),
                'numpy':np.__version__,'shape':[9593,3,6,90,63],
                'definition':'per-clip normalized analytic increments; range mean pool factor3'}
    dest = out/'signals.npy'
    if (out/'PREPARED.json').exists():
        saved=json.loads((out/'PREPARED.json').read_text())
        if saved['identity'] != identity or digest(dest) != saved['sha256']:
            raise ValueError('Prepared data mismatch')
        return saved
    if dest.exists():
        raise ValueError('Unverified existing cache; preserve and use a new directory')
    raw = np.load(path,mmap_mode='r')
    temp = out/'signals.pending.npy'
    arr = np.lib.format.open_memmap(temp,mode='w+',dtype='float16',shape=tuple(identity['shape']))
    started=time.perf_counter()
    for i in range(0,len(raw),16):
        arr[i:i+16] = encode(raw[i:i+16])
        if i % 1024 == 0:
            print(json.dumps({'preparing':i,'total':len(raw)}),flush=True)
    arr.flush();del arr
    temp.replace(dest)
    saved={'identity':identity,'sha256':digest(dest),'seconds':time.perf_counter()-started,
           'label_free':True,'fits_across_samples':False}
    dump(out/'PREPARED.json',saved)
    return saved


def benchmark(data, out, config):
    if out.exists():
        raise ValueError('Use a new benchmark output path')
    labels=np.load(data/'cache/y_clutterremoved.npy')
    users=np.load(data/'cache/subj_clutterremoved.npy')
    ids=source_order(expected_parts(labels,users,1)['train'],config['seed'],1,1)[:config['batch']]
    raw=np.load(data/'cache/X_clutterremoved.npy',mmap_mode='r')
    start=time.perf_counter();x=encode(raw[ids]);prep=time.perf_counter()-start
    x=get_batch(x,np.arange(len(x)),'circular')
    torch.manual_seed(config['seed'])
    model=SignalNet()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    target=torch.from_numpy(labels[ids]).long()
    times=[];losses=[]
    for step in range(5):
        start=time.perf_counter()
        loss=F.cross_entropy(model(x),target)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grads=[p.grad for p in model.encoder.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads)>0
        optimizer.step();times.append(time.perf_counter()-start);losses.append(loss.item())
    median=float(np.median(times[1:]))
    train_n=len(expected_parts(labels,users,1)['train'])
    # Add 50% headroom for validation, cache I/O and checkpointing.
    estimate=1.5*median*math.ceil(train_n/config['batch'])*config['epochs']*8*len(config['arms'])
    record={'state':'cost_gradient_probe_complete','source_ids':ids.tolist(),
            'source_users':users[ids].tolist(),'parameters':sum(p.numel() for p in model.parameters()),
            'batch':config['batch'],'step_seconds':times,'median_step_seconds':median,
            'fixed_batch_training_losses':losses,'preprocess_seconds':prep,
            'encoder_gradients_finite_nonzero':True,'quality_evidence':False,
            'estimated_full_queue_hours_with_50pct_headroom':estimate/3600,
            'cost_gate_passed':estimate/3600<=config['max_estimated_hours'],
            'script_sha256':digest(__file__),'torch':torch.__version__}
    out.parent.mkdir(parents=True,exist_ok=True);dump(out,record);print(json.dumps(record),flush=True)


def fit(out,config,identity,cache,labels,users,held,arm):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        r=json.loads((out/'result.json').read_text())
        assert r['identity']==identity
        for name,sha in r['files'].items():
            assert digest(out/name)==sha
        return r
    parts=expected_parts(labels,users,held)
    for a in parts:
        for b in parts:
            if a != b: assert not np.intersect1d(parts[a],parts[b]).size
    dump(out/'split.json',{k:v.tolist() for k,v in parts.items()})
    torch.manual_seed(config['seed'])
    model=SignalNet()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['epochs'],eta_min=config['min_lr'])
    target=torch.from_numpy(labels).long()
    if (out/'progress.pt').exists():
        s=torch.load(out/'progress.pt',map_location='cpu',weights_only=False)
        if s['identity']!=identity:raise ValueError('Resume identity mismatch')
        model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer'])
        scheduler.load_state_dict(s['scheduler']);torch.set_rng_state(s['rng'])
        history,best,best_ce,best_epoch=(s[k] for k in ('history','best','best_ce','best_epoch'))
    else:
        val=predict(model,cache,parts['val'],arm,config['batch'])
        best_ce=F.cross_entropy(val,target[parts['val']]).item();best_epoch=0
        best={k:v.clone() for k,v in model.state_dict().items()}
        history=[{'epoch':0,'validation_ce':best_ce}]
    for epoch in range(history[-1]['epoch']+1,config['epochs']+1):
        if epoch>config['min_epochs'] and epoch-1-best_epoch>=config['patience']:break
        start=time.perf_counter();order=source_order(parts['train'],config['seed'],held,epoch)
        model.train();loss_sum=0.;lr=optimizer.param_groups[0]['lr']
        for i in range(0,len(order),config['batch']):
            ids=order[i:i+config['batch']]
            loss=F.cross_entropy(model(get_batch(cache,ids,arm)),target[ids])
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            loss_sum+=loss.item()*len(ids)
        scheduler.step()
        val=predict(model,cache,parts['val'],arm,config['batch'])
        ce=F.cross_entropy(val,target[parts['val']]).item()
        if not np.isfinite(ce):raise ValueError('Nonfinite validation CE')
        if ce<best_ce:
            best_ce,best_epoch=ce,epoch
            best={k:v.clone() for k,v in model.state_dict().items()}
        row={'epoch':epoch,'validation_ce':ce,'source_ce':loss_sum/len(order),'lr':lr,
             'source_ids_sha256':array_hash(order),'seconds':time.perf_counter()-start}
        history.append(row)
        atomic_save(out/'progress.pt',{'identity':identity,'model':model.state_dict(),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),
            'history':history,'best':best,'best_ce':best_ce,'best_epoch':best_epoch})
        dump(out/'progress.json',dict(row,state='training',pid=os.getpid(),best_epoch=best_epoch))
        print(json.dumps(dict(row,fold=held,arm=arm)),flush=True)
    model.load_state_dict(best)
    # All test evaluation is AFTER checkpoint selection, never optimizer input.
    val=predict(model,cache,parts['val'],arm,config['batch']).numpy()
    test=predict(model,cache,parts['test'],arm,config['batch']).numpy()
    np.testing.assert_allclose(F.cross_entropy(torch.from_numpy(val),target[parts['val']]).item(),best_ce,atol=1e-7,rtol=0)
    atomic_save(out/'best.pt',{'identity':identity,'model':best,'epoch':best_epoch,'validation_ce':best_ce})
    np.savez_compressed(out/'predictions.npz',ids=parts['test'],y=labels[parts['test']],logits=test,
                        validation_ids=parts['val'],validation_logits=val)
    result={'identity':identity,'arm':arm,'held_user':held,'selected_epoch':best_epoch,
            'selected_val_ce':best_ce,'history':history,'metrics':metrics(labels[parts['test']],test.argmax(1)),
            'files':{name:digest(out/name) for name in ('best.pt','progress.pt','predictions.npz','split.json')}}
    dump(out/'result.json',result);dump(out/'progress.json',{'state':'complete','selected_epoch':best_epoch})
    print(json.dumps({'fold':held,'arm':arm,'complete':True}),flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['benchmark','prepare','train'])
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--prepared',type=Path)
    ap.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'research/CIRCULAR_SIGNAL_PROTOCOL.json')
    args=ap.parse_args();config=json.loads(args.protocol.read_text())
    torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True)
    if args.mode=='benchmark':return benchmark(args.data,args.out,config)
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'queue.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if args.mode=='prepare':return prepare(args.data,args.out)
    if not args.prepared:raise ValueError('--prepared is required for training')
    saved=prepare(args.data,args.prepared)
    paths=[args.data/f'cache/{name}_clutterremoved.npy' for name in ('y','subj')]
    expected=['d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd',
              '5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480']
    assert [digest(p) for p in paths]==expected
    identity={'script_sha256':digest(__file__),'protocol_sha256':digest(args.protocol),
              'helper_sha256':{name:digest(Path(__file__).with_name(name)) for name in ('audit_restored.py','full_view_heads.py')},
              'prepared_sha256':saved['sha256'],'torch':torch.__version__,'numpy':np.__version__}
    if (args.out/'identity.json').exists():
        assert json.loads((args.out/'identity.json').read_text())==identity
    else:
        dump(args.out/'identity.json',identity);dump(args.out/'protocol.json',config)
    labels,users=(np.load(p) for p in paths)
    cache=np.load(args.prepared/'signals.npy',mmap_mode='r')
    results=[]
    try:
        for held in range(1,9):
            for arm in config['arms']:
                dump(args.out/'queue_status.json',{'state':'running','pid':os.getpid(),'fold':held,'arm':arm})
                results.append(fit(args.out/f'fold{held}'/arm,config,dict(identity,fold=held,arm=arm),
                                   cache,labels,users,held,arm))
        summary={a:{'per_user':[r['metrics']['macro_accuracy'] for r in results if r['arm']==a]} for a in config['arms']}
        for row in summary.values():
            row['mean_user_macro_accuracy']=float(np.mean(row['per_user']))
            row['worst_user_macro_accuracy']=float(np.min(row['per_user']))
        candidate=np.array(summary['circular']['per_user'])
        diffs={a:candidate-np.array(summary[a]['per_user']) for a in ('legacy','angle_control')}
        passed=(diffs['legacy'].mean()>=.01 and diffs['angle_control'].mean()>=.005
                and (diffs['angle_control']>0).sum()>=6 and candidate.mean()>=.9344166666666667)
        dump(args.out/'RESULTS.json',{'state':'complete','identity':identity,'summary':summary,
            'paired':{a:{'mean_pp':float(d.mean()*100),'improved_users':int((d>0).sum()),'per_user_pp':(d*100).tolist()} for a,d in diffs.items()},
            'continuation_gate_passed':bool(passed),'novelty_claim':False,'encoder_trained_from_scratch':True})
        dump(args.out/'queue_status.json',{'state':'complete','pid':os.getpid(),'runs':len(results)})
    except BaseException as e:
        dump(args.out/'queue_status.json',{'state':'failed','pid':os.getpid(),'error':str(e),'type':type(e).__name__});raise


if __name__=='__main__':main()
