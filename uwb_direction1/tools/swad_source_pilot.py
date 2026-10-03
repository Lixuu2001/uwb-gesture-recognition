"""Source-user SWAD control pilot. No signals from users1/2 are requested.

One trajectory supplies ERM, fixed tail average and official LossValley SWAD.
Each epoch's segment, every update ID, RNG/optimizer/averaging queues persist.
Diagnostic labels cannot enter gradient or parameter/checkpoint selection.
"""
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
from audit_restored import digest, dump, quality_mask, metrics
from circular_signal import SignalNet, get_batch, predict, source_order
from full_view_heads import atomic_save, array_hash
from swad_core import AveragedModel, LossValley, pack_average, unpack_average, pack_valley, unpack_valley


def parts_for(y, users, held):
    allowed = [3, 4, 5, 6, 7, 8]
    if held not in allowed:
        raise ValueError('Pilot held user must be3..8')
    val = allowed[(allowed.index(held)+1) % len(allowed)]
    good = quality_mask(y, users) & np.isin(users, allowed)
    return {'train': np.flatnonzero(good & ~np.isin(users, [held, val])),
            'val': np.flatnonzero(good & (users == val)),
            'diagnostic': np.flatnonzero(good & (users == held))}


def fit(out, c, identity, cache, y, users, held, stop_after=None):
    out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists():
        result = json.loads((out/'result.json').read_text())
        assert result['identity'] == identity
        for name, sha in result['files'].items():
            assert digest(out/name) == sha
        return result
    parts = parts_for(y, users, held)
    dump(out/'split.json', {k: v.tolist() for k,v in parts.items()})
    torch.manual_seed(c['seed'])
    net = SignalNet()
    opt = torch.optim.AdamW(net.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c['epochs'], eta_min=c['min_lr'])
    target = torch.from_numpy(y).long()
    if (out/'progress.pt').exists():
        s = torch.load(out/'progress.pt', weights_only=False, map_location='cpu')
        assert s['identity'] == identity
        net.load_state_dict(s['model']); opt.load_state_dict(s['optimizer'])
        scheduler.load_state_dict(s['scheduler'])
        valley = unpack_valley(s['valley'], net); tail = unpack_average(s['tail'], net)
        best, best_ce, best_epoch, history, step = (s[k] for k in ('best','best_ce','best_epoch','history','step'))
        torch.set_rng_state(s['rng'])
    else:
        valley = LossValley(None, **{k: c[k] for k in ('n_converge','n_tolerance','tolerance_ratio')})
        tail = AveragedModel(net)
        best_ce = F.cross_entropy(predict(net,cache,parts['val'],'legacy',c['batch']),target[parts['val']]).item()
        best = copy.deepcopy(net.state_dict()); best_epoch = step = 0
        history = [{'epoch':0, 'validation_ce':best_ce}]
    for epoch in range(history[-1]['epoch']+1, c['epochs']+1):
        started = time.perf_counter(); net.train(); segment = AveragedModel(net)
        order = source_order(parts['train'], c['seed'], held, epoch); losses = 0.
        lr = opt.param_groups[0]['lr']
        for start in range(0,len(order),c['batch']):
            ids = order[start:start+c['batch']]
            loss = F.cross_entropy(net(get_batch(cache,ids,'legacy')),target[ids])
            if not torch.isfinite(loss): raise ValueError('Nonfinite source loss')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            segment.update_parameters(net, step=step)
            if epoch >= c['tail_start_epoch']: tail.update_parameters(net, step=step)
            losses += loss.item()*len(ids)
        scheduler.step()
        ce = F.cross_entropy(predict(net,cache,parts['val'],'legacy',c['batch']),target[parts['val']]).item()
        if not np.isfinite(ce): raise ValueError('Nonfinite validation loss')
        if ce < best_ce: best_ce, best_epoch, best = ce, epoch, copy.deepcopy(net.state_dict())
        # Label segment by the endpoint model CE, as the official trainer does.
        valley.update_and_evaluate(segment, None, ce, None)
        segment_path = out/f'segment_{epoch:02d}.pt'
        atomic_save(segment_path, {'average':pack_average(segment),'end_loss':ce,'epoch':epoch,'identity':identity})
        row = {'epoch':epoch,'validation_ce':ce,'source_ce':losses/len(order),'lr':lr,
               'source_ids_sha256':array_hash(order),'step':step,'seconds':time.perf_counter()-started,
               'segment_sha256':digest(segment_path),'converged':valley.is_converged,'dead_valley':valley.dead_valley}
        history.append(row)
        atomic_save(out/'progress.pt', {'identity':identity,'model':net.state_dict(),'optimizer':opt.state_dict(),
            'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),'valley':pack_valley(valley),
            'tail':pack_average(tail),'best':best,'best_ce':best_ce,'best_epoch':best_epoch,'history':history,'step':step})
        dump(out/'progress.json', dict(row,state='training',pid=os.getpid(),held=held))
        print(json.dumps(dict(row,held=held)),flush=True)
        if stop_after is not None and epoch >= stop_after: return None
    # Final selection completes before reading diagnostic inputs/labels for evaluation.
    final = copy.deepcopy(valley).get_final_model()
    states = {'erm':best, 'tail':tail.module.state_dict(), 'swad':final.module.state_dict()}
    choice = {'erm_epoch':best_epoch,'erm_validation_ce':best_ce,'swad_converged':valley.is_converged,
              'swad_dead_valley':valley.dead_valley,'swad_threshold':valley.threshold,
              'swad_start_step':final.start_step,'swad_end_step':final.end_step,
              'swad_averaged_segments':int(final.n_averaged),'tail_steps':int(tail.n_averaged)}
    atomic_save(out/'selected.pt', {'identity':identity,'states':states,'selection':choice})
    dump(out/'selection.json', choice)
    predictions = {'ids':parts['diagnostic'],'y':y[parts['diagnostic']], 'validation_ids':parts['val']}
    scores = {}
    for arm,state in states.items():
        net.load_state_dict(state)
        predictions[arm] = predict(net,cache,parts['diagnostic'],'legacy',c['batch']).numpy()
        predictions[arm+'_validation'] = predict(net,cache,parts['val'],'legacy',c['batch']).numpy()
        scores[arm] = metrics(y[parts['diagnostic']],predictions[arm].argmax(1))
    np.savez_compressed(out/'predictions.npz', **predictions)
    names = ['split.json','progress.pt','selected.pt','predictions.npz','selection.json']
    result = {'identity':identity,'held_source_user':held,'history':history,'selection':choice,'metrics':scores,
              'files':{name:digest(out/name) for name in names}}
    dump(out/'result.json',result); dump(out/'progress.json',{'state':'complete','held':held})
    return result


def audit_run(folder, cache, y, users, c):
    """Independent functional forward, plus replay of average selection from saved segments."""
    from audit_circular_signal import independent_forward
    r = json.loads((folder/'result.json').read_text()); h = r['held_source_user']
    for name,sha in r['files'].items(): assert digest(folder/name) == sha
    # Reconstruct roles independently from the declared cyclic source protocol.
    good = ~((users==6)&(y==3)) & (users>=3) & (users<=8)
    val_user = h+1 if h<8 else 3
    parts = {'train':np.flatnonzero(good & (users!=h)&(users!=val_user)),
             'val':np.flatnonzero(good & (users==val_user)), 'diagnostic':np.flatnonzero(good & (users==h))}
    split = json.loads((folder/'split.json').read_text())
    for k,ids in parts.items(): np.testing.assert_array_equal(ids,split[k])
    selected = torch.load(folder/'selected.pt',weights_only=False,map_location='cpu')
    assert selected['identity'] == r['identity']
    chosen = min(r['history'], key=lambda row:row['validation_ce'])
    assert chosen['epoch'] == selected['selection']['erm_epoch']
    progress=torch.load(folder/'progress.pt',weights_only=False,map_location='cpu')
    assert progress['identity']==r['identity']
    for k,t in progress['best'].items(): assert torch.equal(t,selected['states']['erm'][k])
    for k,t in progress['tail']['weights'].items():
        if k.startswith('module.'):
            assert torch.equal(t,selected['states']['tail'][k[7:]])
    data = np.load(folder/'predictions.npz')
    np.testing.assert_array_equal(data['ids'],parts['diagnostic'])
    np.testing.assert_array_equal(data['validation_ids'],parts['val'])
    np.testing.assert_array_equal(data['y'],y[parts['diagnostic']])
    for row in r['history'][1:]:
        order=np.random.default_rng(c['seed']+10000*h+100000*row['epoch']).permutation(parts['train'])
        assert array_hash(order)==row['source_ids_sha256']
        assert digest(folder/f"segment_{row['epoch']:02d}.pt")==row['segment_sha256']
    for arm,state in selected['states'].items():
        for role,key in [('val',arm+'_validation'),('diagnostic',arm)]:
            ids=parts[role]
            logits=torch.cat([independent_forward(get_batch(cache,ids[i:i+c['batch']],'legacy'),state)
                              for i in range(0,len(ids),c['batch'])]).numpy()
            np.testing.assert_array_equal(logits,data[key])
        assert metrics(y[parts['diagnostic']],data[arm].argmax(1))==r['metrics'][arm]
    # Compare production SWAD selection with independent official implementation,
    # adapted only to permit CPU; numerical differential tests cover boundary cases.
    from test_swad_source import official_classes
    OA, OV=official_classes(); template=SignalNet(); v=OV(None,c['n_converge'],c['n_tolerance'],c['tolerance_ratio'])
    for row in r['history'][1:]:
        s=torch.load(folder/f"segment_{row['epoch']:02d}.pt",weights_only=False,map_location='cpu')['average']
        m=OA(template);m.load_state_dict(s['weights']);m.start_step=s['start_step'];m.end_step=s['end_step']
        v.update_and_evaluate(m,None,row['validation_ce'],None)
    official=v.get_final_model()
    for k,t in official.module.state_dict().items(): assert torch.equal(t,selected['states']['swad'][k])
    return {'held':h,'functional_logits_bitwise_equal':True,'official_valley_replay_bitwise_equal':True,
            'artifact_hashes_verified':len(r['files'])+len(r['history'])-1,'source_only':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['benchmark','train'])
    for name in ['data','prepared','out']: ap.add_argument('--'+name,type=Path,required=True)
    ap.add_argument('--cost',type=Path)
    ap.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'research/SWAD_SOURCE_PROTOCOL.json')
    a=ap.parse_args();c=json.loads(a.protocol.read_text());torch.set_num_threads(c['threads']);torch.use_deterministic_algorithms(True)
    y=np.load(a.data/'cache/y_clutterremoved.npy');users=np.load(a.data/'cache/subj_clutterremoved.npy')
    cache=np.load(a.prepared/'signals.npy',mmap_mode='r')
    identity={'script':digest(__file__),'protocol':digest(a.protocol),
              'helpers':{n:digest(Path(__file__).with_name(n)) for n in ['swad_core.py','circular_signal.py','audit_restored.py','full_view_heads.py','audit_circular_signal.py','test_swad_source.py']},
              'official_sources':{n:digest(Path(__file__).resolve().parents[1]/'research/sources/swad_official'/n) for n in ['swad.py','swa_utils.py','config.yaml','LICENSE']},
              'prepared_sha256':digest(a.prepared/'signals.npy'),'torch':torch.__version__,'numpy':np.__version__}
    assert identity['prepared_sha256']=='1f689bba7f10bb37faf07a557c52653843e81b5907f04d17fa92dbf4dbae7c5f'
    assert digest(a.data/'cache/y_clutterremoved.npy')=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd'
    assert digest(a.data/'cache/subj_clutterremoved.npy')=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480'
    if a.mode=='benchmark':
        if a.out.exists(): raise ValueError('Use a new cost filename')
        ids=source_order(parts_for(y,users,3)['train'],c['seed'],3,1)[:c['batch']];torch.manual_seed(c['seed']);net=SignalNet()
        opt=torch.optim.AdamW(net.parameters(),lr=c['lr'],weight_decay=c['weight_decay']);seg=AveragedModel(net);tail=AveragedModel(net)
        times=[];losses=[]
        for step in range(5):
            t=time.perf_counter();loss=F.cross_entropy(net(get_batch(cache,ids,'legacy')),torch.from_numpy(y[ids]).long())
            opt.zero_grad(set_to_none=True);loss.backward();assert all(torch.isfinite(p.grad).all() for p in net.parameters())
            assert sum(p.grad.abs().sum().item() for p in net.encoder.parameters())>0
            opt.step();seg.update_parameters(net,step=step);tail.update_parameters(net,step=step)
            times.append(time.perf_counter()-t);losses.append(loss.item())
        estimate=float(np.median(times[1:]))*sum(math.ceil(len(parts_for(y,users,h)['train'])/c['batch']) for h in c['users'])*c['epochs']*1.75/3600
        from audit_circular_signal import independent_forward
        seg.eval();x=get_batch(cache,ids,'legacy')
        with torch.no_grad(): assert torch.equal(seg(x),independent_forward(x,seg.module.state_dict()))
        dump(a.out,{'state':'cost_probe_only','identity':identity,'source_ids':ids.tolist(),'source_users':users[ids].tolist(),
                    'step_seconds':times,'losses':losses,'estimated_hours_with_75pct_headroom':estimate,
                    'gate_passed':estimate<=c['max_estimated_hours'],'quality_evidence':False,
                    'encoder_gradients_finite_nonzero':True,'averaged_real_model_independent_forward_bitwise_equal':True})
        return
    if a.cost is None: raise ValueError('Require completed matching cost probe')
    cost=json.loads(a.cost.read_text());assert cost['identity']==identity and cost['gate_passed']
    a.out.mkdir(parents=True,exist_ok=True);lock=(a.out/'queue.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.out/'identity.json').exists(): assert json.loads((a.out/'identity.json').read_text())==identity
    else: dump(a.out/'identity.json',identity);dump(a.out/'protocol.json',c)
    try:
        results=[];audits=[]
        for held in c['users']:
            dump(a.out/'queue_status.json',{'state':'training','pid':os.getpid(),'held':held})
            folder=a.out/f'held{held}'
            results.append(fit(folder,c,dict(identity,held=held),cache,y,users,held))
        dump(a.out/'queue_status.json',{'state':'auditing','pid':os.getpid()})
        for held in c['users']: audits.append(audit_run(a.out/f'held{held}',cache,y,users,c))
        summary={arm:np.array([r['metrics'][arm]['macro_accuracy'] for r in results]) for arm in ['erm','tail','swad']}
        d={arm:summary['swad']-summary[arm] for arm in ['erm','tail']};g=c['gate']
        passed=d['erm'].mean()>=g['swad_minus_erm'] and d['tail'].mean()>=g['swad_minus_tail'] and all((v>0).sum()>=g['wins_each'] for v in d.values()) and summary['swad'].min()>=summary['erm'].min()+g['worst_vs_erm_margin']
        dump(a.out/'AUDIT.json',{'state':'passed','identity':identity,'runs':audits})
        dump(a.out/'SUMMARY.json',{'state':'source_pilot_complete','identity':identity,'summary':{k:{'per_user':v.tolist(),'mean':float(v.mean()),'worst':float(v.min())} for k,v in summary.items()},
            'paired':{k:{'mean_pp':float(v.mean()*100),'wins':int((v>0).sum())} for k,v in d.items()},'gate_passed':bool(passed),'novelty_claim':False,'outer_eight_fold_result':False})
        dump(a.out/'queue_status.json',{'state':'complete','pid':os.getpid(),'trajectories':6,'selected_models':18})
    except BaseException as e:
        dump(a.out/'queue_status.json',{'state':'failed','pid':os.getpid(),'error':str(e),'type':type(e).__name__});raise


if __name__=='__main__': main()
