"""Matched source gradient-variance intervention and conditional-permutation control."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from audit_restored import digest, dump, metrics
from circular_signal import get_batch, predict, source_order
from full_view_heads import atomic_save, array_hash
from swad_core import AveragedModel, pack_average, unpack_average
from swad_source_pilot import parts_for
from circular_signal import SignalNet
from source_fishr import source_step, VarianceEMA, balanced_order

def make_model(arm, seed):
    torch.manual_seed(seed)
    return SignalNet()


def fit(out, c, identity, cache, y, users, held, arm, stop_after=None):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists():
        result = json.loads((out/'result.json').read_text())
        assert result['identity'] == identity
        for name, sha in result['files'].items():
            assert digest(out/name) == sha
        return result
    parts = parts_for(y, users, held)
    dump(out/'split.json', {k: v.tolist() for k,v in parts.items()})
    np.savez_compressed(out/'source_orders.npz', **{f'epoch_{e:02d}':balanced_order(parts['train'],users,c,held,e) for e in range(1,c['epochs']+1)})
    torch.manual_seed(c['seed'])
    net = make_model(arm,c['seed'])
    opt = torch.optim.AdamW(net.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c['epochs'], eta_min=c['min_lr'])
    target = torch.from_numpy(y).long(); domains=torch.from_numpy(users).long(); ema=VarianceEMA(c['ema_beta'])
    if (out/'progress.pt').exists():
        s = torch.load(out/'progress.pt', weights_only=False, map_location='cpu')
        assert s['identity'] == identity
        net.load_state_dict(s['model']); opt.load_state_dict(s['optimizer'])
        scheduler.load_state_dict(s['scheduler']); ema.load_state_dict(s['ema'])
        tail = unpack_average(s['tail'], net)
        best, best_ce, best_epoch, history, step = (s[k] for k in ('best','best_ce','best_epoch','history','step'))
        torch.set_rng_state(s['rng'])
    else:
        tail = AveragedModel(net)
        best_ce = F.cross_entropy(predict(net,cache,parts['val'],'legacy',c['batch']),target[parts['val']]).item()
        best = copy.deepcopy(net.state_dict()); best_epoch = step = 0
        history = [{'epoch':0, 'validation_ce':best_ce}]
    for epoch in range(history[-1]['epoch']+1, c['epochs']+1):
        started = time.perf_counter(); net.train(); segment = AveragedModel(net)
        order = balanced_order(parts['train'],users,c,held,epoch); losses=0.; objectives=0.; real_penalties=0.; shuffled_penalties=0.
        lr = opt.param_groups[0]['lr']
        for start in range(0,len(order),c['batch']):
            ids = order[start:start+c['batch']]
            stats=source_step(net,opt,get_batch(cache,ids,'legacy'),target[ids],domains[ids],ema,c,held,step+1,epoch,arm)
            step += 1
            objectives+=stats['objective']*len(ids); real_penalties+=stats['real_penalty']*len(ids); shuffled_penalties+=stats['permuted_penalty']*len(ids)
            segment.update_parameters(net, step=step)
            if epoch >= c['tail_start_epoch']: tail.update_parameters(net, step=step)
            losses += stats['ce']*len(ids)
        scheduler.step()
        ce = F.cross_entropy(predict(net,cache,parts['val'],'legacy',c['batch']),target[parts['val']]).item()
        if not np.isfinite(ce): raise ValueError('Nonfinite validation loss')
        if ce < best_ce: best_ce, best_epoch, best = ce, epoch, copy.deepcopy(net.state_dict())
        segment_path = out/f'segment_{epoch:02d}.pt'
        atomic_save(segment_path, {'average':pack_average(segment),'end_loss':ce,'epoch':epoch,'identity':identity})
        row = {'epoch':epoch,'validation_ce':ce,'source_ce':losses/len(order),'lr':lr,
               'source_ids_sha256':array_hash(order),'objective':objectives/len(order),'real_penalty':real_penalties/len(order),'permuted_penalty':shuffled_penalties/len(order),'scale':ema.scale,'step':step,'seconds':time.perf_counter()-started,
               'segment_sha256':digest(segment_path)}
        history.append(row)
        atomic_save(out/'progress.pt', {'identity':identity,'model':net.state_dict(),'optimizer':opt.state_dict(),
            'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),'ema':ema.state_dict(),
            'tail':pack_average(tail),'best':best,'best_ce':best_ce,'best_epoch':best_epoch,'history':history,'step':step})
        dump(out/'progress.json', dict(row,state='training',pid=os.getpid(),held=held,arm=arm))
        print(json.dumps(dict(row,held=held,arm=arm)),flush=True)
        if stop_after is not None and epoch >= stop_after: return None
    # Final selection completes before reading diagnostic inputs/labels for evaluation.
    states = {'erm':best, 'tail':tail.module.state_dict()}
    choice = {'erm_epoch':best_epoch,'erm_validation_ce':best_ce,'tail_steps':int(tail.n_averaged)}
    atomic_save(out/'selected.pt', {'identity':identity,'states':states,'selection':choice})
    dump(out/'selection.json', choice)
    predictions = {'ids':parts['diagnostic'],'y':y[parts['diagnostic']], 'validation_ids':parts['val']}
    scores = {}
    for kind,state in states.items():
        net.load_state_dict(state)
        predictions[kind] = predict(net,cache,parts['diagnostic'],'legacy',c['batch']).numpy()
        predictions[kind+'_validation'] = predict(net,cache,parts['val'],'legacy',c['batch']).numpy()
        scores[kind] = metrics(y[parts['diagnostic']],predictions[kind].argmax(1))
    np.savez_compressed(out/'predictions.npz', **predictions)
    names = ['source_orders.npz','split.json','progress.pt','selected.pt','predictions.npz','selection.json']
    result = {'identity':identity,'held_source_user':held,'arm':arm,'history':history,'selection':choice,'metrics':scores,
              'files':{name:digest(out/name) for name in names}}
    dump(out/'result.json',result); dump(out/'progress.json',{'state':'complete','held':held})
    return result


def baseline_check(root,c,identity):
    old=json.loads((root/'protocol.json').read_text())
    for key in ['seed','threads','users','batch','epochs','lr','weight_decay','min_lr','tail_start_epoch']:
        assert c[key]==old[key],key
    records=[]
    for h in c['users']:
        folder=root/f'held{h}';r=json.loads((folder/'result.json').read_text())
        assert r['identity']['prepared_sha256']==identity['prepared_sha256']
        for key in ['torch','numpy']:assert r['identity'][key]==identity[key]
        for name in ['circular_signal.py','swad_core.py','audit_restored.py','full_view_heads.py']:
            assert r['identity']['helpers'][name]==identity['helpers'][name]
        assert r['identity']['script']==identity['helpers']['swad_source_pilot.py']
        for name,sha in r['files'].items():assert digest(folder/name)==sha
        records.append({'held':h,'result_sha256':digest(folder/'result.json'),'files':r['files']})
    return records


def benchmark(a,c,identity,cache,y,users):
    ids=balanced_order(parts_for(y,users,3)['train'],users,c,3,1)[:c['batch']]
    x=get_batch(cache,ids,'legacy');target=torch.from_numpy(y[ids]).long();domains=torch.from_numpy(users[ids]).long();times={}
    for arm in c['arms']:
        net=make_model(arm,c['seed']);net.train();ema=VarianceEMA(c['ema_beta'])
        opt=torch.optim.AdamW(net.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);timings=[]
        for step in range(5):
            started=time.perf_counter();source_step(net,opt,x,target,domains,ema,c,3,step+1,2,arm);timings.append(time.perf_counter()-started)
        times[arm]=timings
    steps=sum(len(balanced_order(parts_for(y,users,h)['train'],users,c,h,1))//c['batch'] for h in c['users'])
    estimate=sum(float(np.median(v[1:])) for v in times.values())*c['epochs']*steps*1.75/3600
    dump(a.out,{'state':'cost_probe_only','identity':identity,'step_seconds':times,'source_ids':ids.tolist(),'parameters':265836,'estimated_hours_with_75pct_headroom':estimate,'gate_passed':estimate<=c['max_estimated_hours'],'quality_evidence':False})


def summarize(root,baseline,c,identity,cache,y,users):
    from audit_source_fishr import audit_run
    values={};audits=[]
    for arm in ['shared128']+c['arms']:
        rows={k:[] for k in ['erm','tail']}
        for h in c['users']:
            folder=baseline/f'held{h}' if arm=='shared128' else root/arm/f'held{h}'
            # Includes independent functional audit of reused baseline predictions.
            audits.append(audit_run(folder,cache,y,users,c))
            r=json.loads((folder/'result.json').read_text())
            for k in rows:rows[k].append(r['metrics'][k]['macro_accuracy'])
        values[arm]={k:{'per_user':v,'mean':float(np.mean(v)),'worst':min(v)} for k,v in rows.items()}
    comparisons={};g=c['gate'];passed=True
    u=np.array(values['real_variance']['tail']['per_user'])
    for control,margin in [('balanced_erm',g['versus_balanced_erm']),('permuted_variance',g['versus_permuted']),('shared128',g['versus_old_tail'])]:
        b=np.array(values[control]['tail']['per_user']);d=u-b
        ok=d.mean()>=margin and int((d>0).sum())>=g['wins_each'] and d.min()>=g['minimum_paired_difference_each']
        passed=passed and bool(ok)
        comparisons[control]={'mean_pp':float(d.mean()*100),'wins':int((d>0).sum()),'minimum_paired_difference_pp':float(d.min()*100),'passed':bool(ok)}
    for h in c['users']:
        states=[torch.load(root/arm/f'held{h}'/'progress.pt',weights_only=False)['ema'] for arm in c['arms']]
        assert all(x['calibration']==states[0]['calibration'] for x in states)
        early=[torch.load(root/arm/f'held{h}'/'segment_01.pt',weights_only=False)['average']['weights'] for arm in c['arms']]
        for key in early[0]:assert all(torch.equal(x[key],early[0][key]) for x in early)
    dump(root/'AUDIT.json',{'state':'passed','identity':identity,'runs':audits})
    dump(root/'SUMMARY.json',{'state':'source_pilot_complete','identity':identity,'summary':values,'paired_tail':comparisons,
         'gate_passed':passed,'novelty_claim':False,'outer_eight_fold_result':False})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['benchmark','train','audit'])
    for n in ['data','prepared','baseline','out']:ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--cost',type=Path)
    ap.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'research/SOURCE_FISHR_PROTOCOL.json')
    a=ap.parse_args();c=json.loads(a.protocol.read_text());torch.set_num_threads(c['threads']);torch.use_deterministic_algorithms(True)
    y=np.load(a.data/'cache/y_clutterremoved.npy');users=np.load(a.data/'cache/subj_clutterremoved.npy')
    cache=np.load(a.prepared/'signals.npy',mmap_mode='r')
    identity={'script':digest(__file__),'protocol':digest(a.protocol),'torch':torch.__version__,'numpy':np.__version__,
        'prepared_sha256':digest(a.prepared/'signals.npy'),
        'helpers':{n:digest(Path(__file__).with_name(n)) for n in ['source_fishr.py','linear_gradient_variance.py','audit_source_fishr.py','audit_source_sam.py','audit_temporal_scale.py','test_source_fishr.py','swad_source_pilot.py','swad_core.py','circular_signal.py','audit_restored.py','full_view_heads.py']}}
    assert identity['prepared_sha256']=='1f689bba7f10bb37faf07a557c52653843e81b5907f04d17fa92dbf4dbae7c5f'
    assert digest(a.data/'cache/y_clutterremoved.npy')=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd'
    assert digest(a.data/'cache/subj_clutterremoved.npy')=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480'
    refs=baseline_check(a.baseline,c,identity)
    identity['baseline_result_hashes']={str(r['held']):r['result_sha256'] for r in refs}
    if a.mode=='benchmark':
        if a.out.exists():raise ValueError('Use a new cost filename')
        benchmark(a,c,identity,cache,y,users);return
    if a.cost is None:raise ValueError('Require matching cost receipt')
    cost=json.loads(a.cost.read_text());assert cost['identity']==identity and cost['gate_passed']
    a.out.mkdir(parents=True,exist_ok=True)
    lock=(a.out/'queue.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.out/'identity.json').exists():assert json.loads((a.out/'identity.json').read_text())==identity
    else:dump(a.out/'identity.json',identity);dump(a.out/'protocol.json',c);dump(a.out/'BASELINE_INDEX.json',refs)
    try:
        if a.mode=='train':
            for h in c['users']:
                for arm in c['arms']:
                    dump(a.out/'queue_status.json',{'state':'training','pid':os.getpid(),'held':h,'arm':arm})
                    fit(a.out/arm/f'held{h}',c,dict(identity,held=h,arm=arm),cache,y,users,h,arm)
        dump(a.out/'queue_status.json',{'state':'auditing','pid':os.getpid()})
        summarize(a.out,a.baseline,c,identity,cache,y,users)
        dump(a.out/'queue_status.json',{'state':'complete','pid':os.getpid(),'new_trajectories':18,'reused_trajectories':6})
    except BaseException as e:
        dump(a.out/'queue_status.json',{'state':'failed','pid':os.getpid(),'type':type(e).__name__,'error':str(e)});raise


if __name__=='__main__':main()
