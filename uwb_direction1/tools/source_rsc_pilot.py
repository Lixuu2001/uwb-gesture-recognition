"""Fixed RSC versus cardinality-matched random masking and existing strong source controls."""
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
from source_rsc import source_step

def make_model(arm, seed):
    torch.manual_seed(seed)
    return SignalNet()

def arm_config(c, arm):
    return dict(c)

def fit(out, c, identity, cache, y, users, held, arm, stop_after=None):
    c=arm_config(c,arm)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'result.json').exists():
        result = json.loads((out/'result.json').read_text())
        assert result['identity'] == identity
        for name, sha in result['files'].items():
            assert digest(out/name) == sha
        return result
    parts = parts_for(y, users, held)
    dump(out/'split.json', {k: v.tolist() for k,v in parts.items()})
    np.savez_compressed(out/'source_orders.npz', **{f'epoch_{e:02d}':source_order(parts['train'],c['seed'],held,e) for e in range(1,c['epochs']+1)})
    torch.manual_seed(c['seed'])
    net = make_model(arm,c['seed'])
    opt = torch.optim.AdamW(net.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, c['epochs'], eta_min=c['min_lr'])
    target = torch.from_numpy(y).long()
    if (out/'progress.pt').exists():
        s = torch.load(out/'progress.pt', weights_only=False, map_location='cpu')
        assert s['identity'] == identity
        net.load_state_dict(s['model']); opt.load_state_dict(s['optimizer'])
        scheduler.load_state_dict(s['scheduler'])
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
        order = source_order(parts['train'], c['seed'], held, epoch); losses = 0.; mask_rows=[]
        lr = opt.param_groups[0]['lr']
        for start in range(0,len(order),c['batch']):
            ids = order[start:start+c['batch']]
            loss,stats=source_step(net,opt,get_batch(cache,ids,'legacy'),target[ids],arm,c['feature_drop'],c['batch_drop'],c['mask_seed_base']+held*100000+step)
            mask_rows.append(dict(stats,step=step,source_ids_sha256=array_hash(ids),mask_seed=c['mask_seed_base']+held*100000+step,samples=len(ids)))
            step += 1
            segment.update_parameters(net, step=step)
            if epoch >= c['tail_start_epoch']: tail.update_parameters(net, step=step)
            losses += loss*len(ids)
        scheduler.step()
        ce = F.cross_entropy(predict(net,cache,parts['val'],'legacy',c['batch']),target[parts['val']]).item()
        if not np.isfinite(ce): raise ValueError('Nonfinite validation loss')
        if ce < best_ce: best_ce, best_epoch, best = ce, epoch, copy.deepcopy(net.state_dict())
        segment_path = out/f'segment_{epoch:02d}.pt'
        atomic_save(segment_path, {'average':pack_average(segment),'end_loss':ce,'epoch':epoch,'identity':identity})
        row = {'epoch':epoch,'validation_ce':ce,'source_ce':losses/len(order),'lr':lr,
               'source_ids_sha256':array_hash(order),'mask_batches':mask_rows,'step':step,'seconds':time.perf_counter()-started,
               'segment_sha256':digest(segment_path)}
        history.append(row)
        atomic_save(out/'progress.pt', {'identity':identity,'model':net.state_dict(),'optimizer':opt.state_dict(),
            'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state(),
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
    from audit_temporal_scale import functional_forward
    ids=source_order(parts_for(y,users,3)['train'],c['seed'],3,1)[:c['batch']]
    x=get_batch(cache,ids,'legacy');target=torch.from_numpy(y[ids]).long();times={}
    for arm in c['arms']:
        net=make_model(arm,c['seed']);net.eval()
        with torch.no_grad():assert torch.equal(net(x),functional_forward(x,net.state_dict(),'shared128'))
        opt=torch.optim.AdamW(net.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
        avg=AveragedModel(net);seg=AveragedModel(net);timings=[]
        for step in range(5):
            net.train();t=time.perf_counter()
            source_step(net,opt,x,target,arm,c['feature_drop'],c['batch_drop'],c['mask_seed_base']+3*100000+step)
            avg.update_parameters(net,step=step);seg.update_parameters(net,step=step)
            timings.append(time.perf_counter()-t)
        times[arm]=timings
    checkfolder=a.out.parent/'baseline_epoch_check'
    fit(checkfolder,c,dict(identity,held=3,arm='shared128'),cache,y,users,3,'shared128',stop_after=1)
    state=torch.load(checkfolder/'progress.pt',weights_only=False)
    old=json.loads((a.baseline/'held3/result.json').read_text())
    for i in [0,1]:
        for key in ['validation_ce']+(['source_ce','lr','source_ids_sha256','step'] if i else []):
            assert state['history'][i][key]==old['history'][i][key],key
    seg=torch.load(checkfolder/'segment_01.pt',weights_only=False)['average']
    ref=torch.load(a.baseline/'held3/segment_01.pt',weights_only=False)['average']
    for k,t in seg['weights'].items():assert torch.equal(t,ref['weights'][k])
    per_epoch=sum(math.ceil(len(parts_for(y,users,h)['train'])/c['batch']) for h in c['users'])
    estimate=sum(float(np.median(times[arm][1:]))*arm_config(c,arm)['epochs'] for arm in c['arms'])*per_epoch*1.75/3600
    dump(a.out,{'state':'cost_probe_only','identity':identity,'step_seconds':times,'source_ids':ids.tolist(),
        'parameters':265836,'functional_forward_bitwise_equal':True,
        'real_shared_baseline_epoch1_metrics_and_segment_weights_bitwise_equal':True,
        'estimated_hours_with_75pct_headroom':estimate,'gate_passed':estimate<=c['max_estimated_hours'],
        'quality_evidence':False})


def summarize(root,baseline,c,identity,cache,y,users):
    from audit_source_rsc import audit_run
    values={};audits=[]
    for arm in ['shared128','sam24']+c['arms']:
        rows={k:[] for k in ['erm','tail']}
        for h in c['users']:
            folder=(baseline/f'held{h}' if arm=='shared128' else Path(c['sam_root'])/'sam24'/f'held{h}' if arm=='sam24' else root/arm/f'held{h}')
            audits.append(audit_run(folder,cache,y,users,c))
            r=json.loads((folder/'result.json').read_text())
            for k in rows:rows[k].append(r['metrics'][k]['macro_accuracy'])
        values[arm]={k:{'per_user':v,'mean':float(np.mean(v)),'worst':min(v)} for k,v in rows.items()}
    pinned_sam=json.loads(Path(c['sam_summary']).read_text())['summary']['sam24']
    assert values['sam24']==pinned_sam
    strong=json.loads(Path(c['cnn_summary']).read_text())['scores']['cnn_readout']
    values['cnn_readout']={'tail':{'per_user':strong,'mean':float(np.mean(strong)),'worst':min(strong)}}
    comparisons={};g=c['gate'];passed=True;u=np.array(values['rsc24']['tail']['per_user'])
    for control,margin in g['mean_margin'].items():
        d=u-np.array(values[control]['tail']['per_user'])
        ok=d.mean()>=margin and int((d>0).sum())>=g['wins_each'] and d.min()>=g['minimum_paired_difference_each']
        passed=passed and bool(ok)
        comparisons[control]={'mean_pp':float(d.mean()*100),'wins':int((d>0).sum()),'minimum_paired_difference_pp':float(d.min()*100),'passed':bool(ok)}
    dump(root/'AUDIT.json',{'state':'passed','identity':identity,'runs':audits,'cnn_readout_provenance':'prior independent refit audit plus pinned summary SHA; not re-fitted in this audit'})
    dump(root/'SUMMARY.json',{'state':'source_pilot_complete','identity':identity,'summary':values,'paired_tail':comparisons,'gate_passed':passed,'novelty_claim':False,'outer_eight_fold_result':False})


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['benchmark','train','audit'])
    for n in ['data','prepared','baseline','out']:ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--cost',type=Path)
    ap.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parents[1]/'research/SOURCE_RSC_PROTOCOL.json')
    a=ap.parse_args();c=json.loads(a.protocol.read_text());torch.set_num_threads(c['threads']);torch.use_deterministic_algorithms(True)
    y=np.load(a.data/'cache/y_clutterremoved.npy');users=np.load(a.data/'cache/subj_clutterremoved.npy')
    cache=np.load(a.prepared/'signals.npy',mmap_mode='r')
    identity={'script':digest(__file__),'protocol':digest(a.protocol),'torch':torch.__version__,'numpy':np.__version__,
        'prepared_sha256':digest(a.prepared/'signals.npy'),
        'helpers':{n:digest(Path(__file__).with_name(n)) for n in ['source_rsc.py','audit_source_rsc.py','audit_source_sam.py','audit_temporal_scale.py','test_source_rsc.py','swad_source_pilot.py','swad_core.py','circular_signal.py','audit_restored.py','full_view_heads.py']}}
    assert identity['prepared_sha256']=='1f689bba7f10bb37faf07a557c52653843e81b5907f04d17fa92dbf4dbae7c5f'
    assert digest(a.data/'cache/y_clutterremoved.npy')=='d77b10f0fbda231246e71dfcf0b2f852863af48f80f159aed90f5f9b7c61a8bd'
    assert digest(a.data/'cache/subj_clutterremoved.npy')=='5ab32ab59e5b41c05d301b76949f159ddb1e9423a749deb760585a5f19cf9480'
    assert digest(c['sam_summary'])==c['sam_summary_sha256']
    assert digest(c['cnn_summary'])==c['cnn_summary_sha256']
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
        dump(a.out/'queue_status.json',{'state':'complete','pid':os.getpid(),'new_trajectories':12,'reused_trajectories':12})
    except BaseException as e:
        dump(a.out/'queue_status.json',{'state':'failed','pid':os.getpid(),'type':type(e).__name__,'error':str(e)});raise


if __name__=='__main__':main()
