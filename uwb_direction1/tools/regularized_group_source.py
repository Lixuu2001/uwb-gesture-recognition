"""Matched regularization/group-DRO source pilot. No held labels enter training/selection."""
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
from regularized_groups import GroupRisk, balanced_order

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
    torch.manual_seed(c['seed'])
    net = SignalNet()
    risk = GroupRisk(eta=c['group_eta'],robust=arm=='dro_reg')
    opt = torch.optim.AdamW(net.parameters(), lr=c['lr'], weight_decay=c['weight_decays'][arm])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c['epochs'], eta_min=c['min_lr'])
    target = torch.from_numpy(y).long()
    if (out/'progress.pt').exists():
        s = torch.load(out/'progress.pt', weights_only=False, map_location='cpu')
        assert s['identity'] == identity
        net.load_state_dict(s['model']); opt.load_state_dict(s['optimizer'])
        scheduler.load_state_dict(s['scheduler']); risk.load(s['risk'])
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
        order,group_users = balanced_order(parts['train'],users,c['seed'],held,epoch,c['per_group']); losses = 0.; group_log=[]; q_log=[]
        lr = opt.param_groups[0]['lr']
        for ids in order:
            group=torch.from_numpy(np.searchsorted(group_users,users[ids])).long()
            individual=F.cross_entropy(net(get_batch(cache,ids,'legacy')),target[ids],reduction='none')
            loss,means,q=risk.loss(individual,group);group_log.append(means.numpy());q_log.append(q.numpy())
            if not torch.isfinite(loss): raise ValueError('Nonfinite source loss')
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); step += 1
            segment.update_parameters(net, step=step)
            if epoch >= c['tail_start_epoch']: tail.update_parameters(net, step=step)
            losses += loss.item()*len(ids)
        scheduler.step()
        ce = F.cross_entropy(predict(net,cache,parts['val'],'legacy',c['batch']),target[parts['val']]).item()
        if not np.isfinite(ce): raise ValueError('Nonfinite validation loss')
        if ce < best_ce: best_ce, best_epoch, best = ce, epoch, copy.deepcopy(net.state_dict())
        segment_path = out/f'segment_{epoch:02d}.pt'
        atomic_save(segment_path, {'average':pack_average(segment),'end_loss':ce,'epoch':epoch,'identity':identity})
        row = {'epoch':epoch,'validation_ce':ce,'source_ce':losses/order.size,'lr':lr,
               'source_ids_sha256':array_hash(order),'step':step,'seconds':time.perf_counter()-started,
               'segment_sha256':digest(segment_path)}
        np.savez_compressed(out/f'updates_{epoch:02d}.npz',ids=order,group_users=group_users,group_loss=np.stack(group_log),q=np.stack(q_log))
        row['updates_sha256']=digest(out/f'updates_{epoch:02d}.npz');row['logq']=risk.logq.tolist()
        history.append(row)
        atomic_save(out/'progress.pt', {'identity':identity,'model':net.state_dict(),'optimizer':opt.state_dict(),
            'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),'risk':risk.state(),
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
    names = ['split.json','progress.pt','selected.pt','predictions.npz','selection.json']
    result = {'identity':identity,'held_source_user':held,'arm':arm,'history':history,'selection':choice,'metrics':scores,
              'files':{name:digest(out/name) for name in names}}
    dump(out/'result.json',result); dump(out/'progress.json',{'state':'complete','held':held})
    return result


def audit_run(folder,cache,y,users,c):
    from audit_circular_signal import independent_forward
    r=json.loads((folder/'result.json').read_text());h=r['held_source_user'];arm=r['arm'];sp=json.loads((folder/'split.json').read_text());v=h+1 if h<8 else 3;good=(users>=3)&(users<=8)&~((users==6)&(y==3))
    expected=dict(train=np.flatnonzero(good&(users!=h)&(users!=v)),val=np.flatnonzero(good&(users==v)),diagnostic=np.flatnonzero(good&(users==h)))
    for key,ids in expected.items():np.testing.assert_array_equal(ids,sp[key])
    for n,sha in r['files'].items():assert digest(folder/n)==sha
    selected=torch.load(folder/'selected.pt',weights_only=False,map_location='cpu');progress=torch.load(folder/'progress.pt',weights_only=False,map_location='cpu');pred=np.load(folder/'predictions.npz')
    best=min(r['history'],key=lambda x:x['validation_ce']);assert best['epoch']==selected['selection']['erm_epoch']
    np.testing.assert_array_equal(pred['ids'],expected['diagnostic']);np.testing.assert_array_equal(pred['validation_ids'],expected['val']);np.testing.assert_array_equal(pred['y'],y[expected['diagnostic']])
    for kind,state in selected['states'].items():
        reference=progress['best'] if kind=='erm' else {k[7:]:t for k,t in progress['tail']['weights'].items() if k.startswith('module.')}
        for k,t in state.items():assert torch.equal(t,reference[k])
        for role,key in [('val',kind+'_validation'),('diagnostic',kind)]:
            ids=expected[role];ref=torch.cat([independent_forward(get_batch(cache,ids[i:i+c['batch']],'legacy'),state) for i in range(0,len(ids),c['batch'])]).numpy();np.testing.assert_array_equal(ref,pred[key])
        assert metrics(y[expected['diagnostic']],pred[kind].argmax(1))==r['metrics'][kind]
    # Independently rebuild balanced source batches and log-space q updates.
    logq=np.full(4,-np.log(4.));steps=0
    from scipy.special import logsumexp
    for row in r['history'][1:]:
        e=row['epoch'];saved=np.load(folder/f'updates_{e:02d}.npz');groups=np.unique(users[expected['train']]);pools=[expected['train'][users[expected['train']]==g] for g in groups];n=math.ceil(max(map(len,pools))/c['per_group']);cols=[]
        for g,pool in zip(groups,pools):
            perm=np.random.default_rng(c['seed']+10000*h+100000*e+int(g)).permutation(pool);cols.append(np.resize(perm,n*c['per_group']).reshape(n,c['per_group']))
        order=np.concatenate(cols,1);np.testing.assert_array_equal(order,saved['ids']);np.testing.assert_array_equal(groups,saved['group_users']);assert array_hash(order)==row['source_ids_sha256']
        for loss,q in zip(saved['group_loss'],saved['q']):
            if arm=='dro_reg':logq+=c['group_eta']*loss.astype('float64');logq-=logsumexp(logq)
            np.testing.assert_allclose(np.exp(logq),q,rtol=2e-7,atol=1e-8)
        np.testing.assert_allclose(logq,row['logq'],rtol=1e-12,atol=1e-12);steps+=len(order);assert steps==row['step']
        assert digest(folder/f'updates_{e:02d}.npz')==row['updates_sha256'];assert digest(folder/f'segment_{e:02d}.pt')==row['segment_sha256']
    np.testing.assert_allclose(logq,progress['risk']['logq'].numpy(),rtol=1e-12,atol=1e-12)
    return dict(held=h,arm=arm,artifact_hashes=len(r['files'])+2*(len(r['history'])-1),functional_logits_bitwise_equal=True,source_sampler_and_all_q_updates_rebuilt=True)


def summarize(a,c,identity,cache,y,users):
    scores={};audits=[]
    for arm in c['arms']:
        values={k:[] for k in ['erm','tail']}
        for h in c['users']:
            folder=a.out/arm/f'held{h}';audits.append(audit_run(folder,cache,y,users,c));r=json.loads((folder/'result.json').read_text())
            for k in values:values[k].append(r['metrics'][k]['macro_accuracy'])
        scores[arm]=values
    old=[]
    for h in c['users']:
        folder=a.baseline/f'held{h}';r=json.loads((folder/'result.json').read_text());assert digest(folder/'selected.pt')==r['files']['selected.pt'];old.append(r['metrics']['tail']['macro_accuracy'])
    primary=np.array(scores['dro_reg']['erm']);comparisons={}
    for name,ref in [('uniform_base',scores['uniform_base']['erm']),('uniform_reg',scores['uniform_reg']['erm']),('old_tail',old)]:
        ref=np.array(ref);delta=primary-ref;g=c['gate'];ok=delta.mean()>=g['mean_gain'] and (delta>0).sum()>=g['wins_each'] and delta.min()>=-g['max_user_decline']
        comparisons[name]=dict(mean_gain=float(delta.mean()),wins=int((delta>0).sum()),min_user_difference=float(delta.min()),passed=bool(ok))
    dump(a.out/'AUDIT.json',dict(passed=True,runs=audits));dump(a.out/'SUMMARY.json',dict(scores=scores,old_tail=old,comparisons=comparisons,passed=all(v['passed'] for v in comparisons.values()),novelty_claim=False,scope='Six source development trajectories; not outer8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['benchmark','train','audit']);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cost',type=Path);a=p.parse_args();c=json.loads(a.protocol.read_text());a.baseline=Path(c['baseline'])
    torch.set_num_threads(c['threads']);torch.use_deterministic_algorithms(True)
    paths=dict(cache=Path(c['prepared'])/'signals.npy',labels=Path(c['data'])/'cache/y_clutterremoved.npy',users=Path(c['data'])/'cache/subj_clutterremoved.npy');hashes={k:digest(v) for k,v in paths.items()};assert hashes==c['input_sha256']
    identity=dict(script=digest(__file__),protocol=digest(a.protocol),helpers={n:digest(Path(__file__).with_name(n)) for n in ['regularized_groups.py','circular_signal.py','swad_core.py','swad_source_pilot.py','full_view_heads.py','audit_restored.py','audit_circular_signal.py']},inputs=hashes,torch=torch.__version__,numpy=np.__version__)
    refs={str(h):digest(a.baseline/f'held{h}/result.json') for h in c['users']};identity['baseline_result_hashes']=refs
    cache=np.load(paths['cache'],mmap_mode='r');y=np.load(paths['labels']);users=np.load(paths['users'])
    if a.mode=='benchmark':
        if a.out.exists():raise ValueError('New cost filename required')
        order,gs=balanced_order(parts_for(y,users,3)['train'],users,c['seed'],3,1,c['per_group']);ids=order[0];x=get_batch(cache,ids,'legacy');target=torch.from_numpy(y[ids]).long();group=torch.from_numpy(np.searchsorted(gs,users[ids])).long();times={}
        for arm in c['arms']:
            torch.manual_seed(c['seed']);net=SignalNet();risk=GroupRisk(eta=c['group_eta'],robust=arm=='dro_reg');opt=torch.optim.AdamW(net.parameters(),lr=c['lr'],weight_decay=c['weight_decays'][arm]);avg=AveragedModel(net);timings=[]
            for step in range(5):
                t=time.perf_counter();loss,_,_=risk.loss(F.cross_entropy(net(x),target,reduction='none'),group);opt.zero_grad(set_to_none=True);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters());opt.step();avg.update_parameters(net,step=step);timings.append(time.perf_counter()-t)
            times[arm]=timings
        steps=sum(len(balanced_order(parts_for(y,users,h)['train'],users,c['seed'],h,1,c['per_group'])[0]) for h in c['users'])*c['epochs'];estimate=1.75*steps*sum(float(np.median(t[1:])) for t in times.values())/3600
        dump(a.out,dict(identity=identity,step_seconds=times,estimated_hours=estimate,passed=estimate<c['max_hours'],source_ids=ids.tolist(),quality_evidence=False));print(json.dumps(dict(hours=estimate,passed=estimate<c['max_hours'])));return
    if a.cost is None:raise ValueError('Matching cost required')
    cost=json.loads(a.cost.read_text());assert cost['identity']==identity and cost['passed']
    a.out.mkdir(exist_ok=True,parents=True);lock=(a.out/'queue.lock').open('a');fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (a.out/'identity.json').exists():assert json.loads((a.out/'identity.json').read_text())==identity
    else:dump(a.out/'identity.json',identity);dump(a.out/'protocol.json',c)
    try:
        if a.mode=='train':
            for h in c['users']:
                for arm in c['arms']:
                    dump(a.out/'queue_status.json',dict(state='training',held=h,arm=arm,pid=os.getpid()));fit(a.out/arm/f'held{h}',c,dict(identity,held=h,arm=arm),cache,y,users,h,arm)
        dump(a.out/'queue_status.json',dict(state='auditing',pid=os.getpid()));summarize(a,c,identity,cache,y,users);dump(a.out/'queue_status.json',dict(state='complete',pid=os.getpid()))
    except BaseException as e:
        dump(a.out/'queue_status.json',dict(state='failed',pid=os.getpid(),error=str(e),type=type(e).__name__));raise
if __name__=='__main__':main()
