"""Source-fitted range-moment branch with matched static control and saved strong baseline."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from audit_restored import digest, dump, expected_parts, metrics
from full_view_heads import atomic_save, array_hash
from circular_signal import SignalNet, get_batch as signal_batch, source_order
from probe_range_moments import moments, features


class MomentNet(nn.Module):
    def __init__(self, seq, train_ids, arm):
        super().__init__()
        base=SignalNet()
        self.encoder=base.encoder
        self.moment_branch=nn.Sequential(nn.Linear(90,32),nn.GELU(),nn.Linear(32,32),nn.GELU())
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(1728+32,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,12))
        x=features(seq[train_ids], 'static' if arm=='static' else 'ordered_absolute')
        scaler=StandardScaler().fit(x)
        self.register_buffer('moment_mean',torch.from_numpy(scaler.mean_.astype(np.float32)))
        self.register_buffer('moment_scale',torch.from_numpy(scaler.scale_.astype(np.float32)))
    def forward(self, inputs):
        x,m=inputs
        n,v,c,t,r=x.shape
        encoded=self.encoder(x.reshape(n*v,c,t,r)).reshape(n,-1)
        m=self.moment_branch((m-self.moment_mean)/self.moment_scale)
        return self.head(torch.cat([encoded,m],1))


def get_batch(cache, ids, arm):
    x=signal_batch(cache[0],ids,'legacy')
    m=features(cache[1][ids], 'static' if arm=='static' else 'ordered_absolute')
    return x,torch.from_numpy(m.astype(np.float32))


@torch.no_grad()
def predict(model, cache, ids, arm, batch):
    model.eval()
    return torch.cat([model(get_batch(cache,ids[i:i+batch],arm)) for i in range(0,len(ids),batch)])


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
    model=MomentNet(cache[1],parts['train'],arm)
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


def prepare_moments(data, out):
    raw_path=data/'cache/X_clutterremoved.npy'
    raw_sha=digest(raw_path)
    assert raw_sha=='3004e7a2aa44cb4ffd763a6ef3bf30c85d2b7dc0499140c195c48fa3ed6ce519'
    identity={'raw_sha256':raw_sha,'extractor_sha256':digest(Path(__file__).with_name('probe_range_moments.py'))}
    if (out/'PREPARED.json').exists():
        record=json.loads((out/'PREPARED.json').read_text())
        assert record['identity']==identity and digest(out/'moments.npy')==record['sha256']
        return record
    if out.exists():raise ValueError('Unverified existing moment cache; preserve it')
    raw=np.load(raw_path,mmap_mode='r');out.mkdir(parents=True)
    seq=np.concatenate([moments(raw[i:i+32]) for i in range(0,len(raw),32)])
    np.save(out/'moments.npy',seq)
    record={'identity':identity,'sha256':digest(out/'moments.npy'),'shape':list(seq.shape),'label_free':True}
    dump(out/'PREPARED.json',record);return record


def benchmark(data, prepared, out, config):
    if out.exists():raise ValueError('Preserve benchmark')
    labels=np.load(data/'cache/y_clutterremoved.npy');users=np.load(data/'cache/subj_clutterremoved.npy')
    ids=expected_parts(labels,users,1)['train']
    # Source-only RF read. Benchmark does not read a target clip or use its statistic.
    raw=np.load(data/'cache/X_clutterremoved.npy',mmap_mode='r')
    seq=np.concatenate([moments(raw[ids[i:i+32]]) for i in range(0,len(ids),32)])
    signal=np.load(prepared/'signals.npy',mmap_mode='r')
    torch.manual_seed(config['seed']);model=MomentNet(seq,np.arange(len(ids)),'ordered')
    batchids=source_order(np.arange(len(ids)),config['seed'],1,1)[:config['batch']]
    x=signal_batch(signal,ids[batchids],'legacy')
    m=torch.from_numpy(features(seq[batchids],'ordered_absolute').astype(np.float32))
    target=torch.from_numpy(labels[ids[batchids]]).long()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    times=[];losses=[]
    for step in range(5):
        start=time.perf_counter();loss=F.cross_entropy(model((x,m)),target)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        for block in (model.encoder,model.moment_branch):
            grads=[p.grad for p in block.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum().item() for g in grads)>0
        optimizer.step();times.append(time.perf_counter()-start);losses.append(loss.item())
    estimate=float(np.median(times[1:]))*math.ceil(len(ids)/config['batch'])*config['epochs']*16*1.5/3600
    record={'state':'source_gradient_cost_probe_complete','source_ids':ids[batchids].tolist(),
        'parameters':sum(p.numel() for p in model.parameters()),'step_seconds':times,'losses':losses,
        'estimated_queue_hours':estimate,'cost_gate_passed':estimate<=config['max_estimated_hours'],
        'script_sha256':digest(__file__),'quality_evidence':False,'both_branch_gradients_finite_nonzero':True}
    dump(out,record);print(json.dumps(record),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['benchmark','train'])
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--moments',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--baseline',type=Path)
    ap.add_argument('--cost',type=Path)
    ap.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'research/MOMENT_GUIDED_PROTOCOL.json')
    args=ap.parse_args();config=json.loads(args.protocol.read_text())
    torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True)
    if args.mode=='benchmark':return benchmark(args.data,args.prepared,args.out,config)
    if not all((args.moments,args.baseline,args.cost)):raise ValueError('Need moments, baseline and cost paths')
    cost=json.loads(args.cost.read_text());assert cost['cost_gate_passed'] and cost['script_sha256']==digest(__file__)
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'queue.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        saved=prepare_moments(args.data,args.moments)
        paths=[args.data/f'cache/{name}_clutterremoved.npy' for name in ('y','subj')]
        assert [digest(p) for p in paths]==['d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd','5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480']
        signal_sha=digest(args.prepared/'signals.npy')
        assert signal_sha=='1f689bba7f10bb37faf07a557c52653843e81b5907f04d17fa92dbf4dbae7c5f'
        baseline=[];basefiles={}
        for held in range(1,9):
            d=args.baseline/f'fold{held}'/'legacy';r=json.loads((d/'result.json').read_text())
            assert r['held_user']==held and r['arm']=='legacy'
            for name,h in r['files'].items():assert digest(d/name)==h
            baseline.append(r['metrics']['macro_accuracy']);basefiles[str(held)]=digest(d/'result.json')
        identity={'script_sha256':digest(__file__),'protocol_sha256':digest(args.protocol),
            'helper_sha256':{n:digest(Path(__file__).with_name(n)) for n in ['circular_signal.py','probe_range_moments.py','audit_restored.py','full_view_heads.py']},
            'signal_sha256':signal_sha,'moments_sha256':saved['sha256'],'baseline_result_sha256':basefiles,
            'torch':torch.__version__,'numpy':np.__version__,'protocol':config}
        if (args.out/'identity.json').exists():assert json.loads((args.out/'identity.json').read_text())==identity
        else:dump(args.out/'identity.json',identity)
        labels,users=(np.load(p) for p in paths)
        cache=(np.load(args.prepared/'signals.npy',mmap_mode='r'),np.load(args.moments/'moments.npy',mmap_mode='r'))
        rows=[]
        for held in range(1,9):
            for arm in config['arms']:
                dump(args.out/'queue_status.json',{'state':'running','pid':os.getpid(),'fold':held,'arm':arm,'completed_runs':len(rows)})
                rows.append(fit(args.out/f'fold{held}'/arm,config,dict(identity,fold=held,arm=arm),cache,labels,users,held,arm))
        scores={a:[r['metrics']['macro_accuracy'] for r in rows if r['arm']==a] for a in config['arms']}
        scores['legacy']=baseline
        ordered=np.array(scores['ordered']);ds=ordered-np.array(scores['static']);db=ordered-np.array(baseline)
        gate=bool(db.mean()>=.01 and ds.mean()>=.005 and (ds>0).sum()>=6 and ordered.min()>=min(baseline))
        dump(args.out/'RESULTS.json',{'state':'complete_pending_independent_audit','scores':scores,
            'means':{a:float(np.mean(s)) for a,s in scores.items()},'continue_gate_passed':gate,
            'paired_vs_legacy':db.tolist(),'paired_vs_static':ds.tolist(),'novelty_claim':False,'identity':identity})
        dump(args.out/'queue_status.json',{'state':'complete_pending_independent_audit','pid':os.getpid(),'runs':len(rows)})
    except BaseException as e:
        dump(args.out/'queue_status.json',{'state':'failed','pid':os.getpid(),'error':str(e),'type':type(e).__name__});raise


if __name__=='__main__':main()
