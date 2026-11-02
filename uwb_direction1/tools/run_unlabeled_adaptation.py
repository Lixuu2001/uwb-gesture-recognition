"""Stream inference/adaptation only. Scoring labels are never loaded here."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from circular_signal import SignalNet,get_batch
from unlabeled_adaptation import Adapter
from audit_restored import digest,dump
from audit_circular_signal import independent_forward


def checkpoint(adapter):
    s=adapter.state();s['model']={n:s['model'][n] for n,_ in adapter.named};s['model_is_GN_delta_only']=True
    return s


def main(a):
    c=json.loads(a.protocol.read_text());torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    if a.out.exists():raise ValueError('Fresh output directory required')
    cachepath=Path(c['cache']);assert digest(cachepath)==c['cache_sha256'];cache=np.load(cachepath,mmap_mode='r')
    a.out.mkdir();dump(a.out/'protocol.json',c);started=time.perf_counter();records=[]
    for h in c['users']:
        root=Path(c['models'])/f'held{h}';p=root/'selected.pt';old=json.loads((root/'result.json').read_text());assert digest(p)==old['files']['selected.pt']
        state=torch.load(p,map_location='cpu',weights_only=False)['states']['tail'];net=SignalNet();net.load_state_dict(state);net.eval()
        split=json.loads((root/'split.json').read_text());ids=np.array(split['train'][:64] if a.cost else split['diagnostic'])
        if a.cost:
            times={}
            for arm in c['arms']:
                ad=Adapter(net,arm,**c['adapter']);t=time.perf_counter()
                for b in range(0,len(ids),c['batch']):ad.step(get_batch(cache,ids[b:b+c['batch']],'legacy'))
                times[arm]=time.perf_counter()-t
            estimate=2*7100/len(ids)*len(c['orders'])*sum(times.values())
            row=dict(seconds64=times,estimate_seconds=estimate,budget_seconds=c['budget_seconds'],passed=estimate<c['budget_seconds'],labels_loaded=False,source_train_ids=ids.tolist());dump(a.out/'COST.json',row);print(json.dumps(row),flush=True);return
        d=a.out/f'held{h}';d.mkdir();dump(d/'split.json',split);preds={};rows=[]
        for ordername in c['orders']:
            order=ids.copy() if ordername=='storage' else np.random.default_rng(c['order_seed']+h).permutation(ids)
            preds[ordername+'_ids']=order
            frozen=None
            for arm in c['arms']:
                ad=Adapter(net,arm,**c['adapter']);out=[];logs=[];snapshots=[];t=time.perf_counter()
                for b in range(0,len(order),c['batch']):
                    batchids=order[b:b+c['batch']];x=get_batch(cache,batchids,'legacy')
                    before={k:v.detach().clone() for k,v in ad.model.state_dict().items()} if b==0 else None
                    z,log=ad.step(x);out.append(z.numpy());log.update(start=b,ids=batchids.tolist());logs.append(log);snapshots.append(checkpoint(ad))
                    if b==0:
                        ref=independent_forward(x,before).numpy();np.testing.assert_allclose(z.numpy(),ref,rtol=0,atol=2e-5)
                arr=np.concatenate(out);preds[ordername+'_'+arm]=arr
                if arm=='frozen':frozen=arr
                else:np.testing.assert_array_equal(arr[:c['batch']],frozen[:c['batch']])
                for k,v in ad.model.state_dict().items():
                    if k not in dict(ad.named):torch.testing.assert_close(v,state[k],rtol=0,atol=0)
                record=dict(order=ordername,arm=arm,seconds=time.perf_counter()-t,updates=ad.updates,resets=ad.resets,batches=logs)
                rows.append(record);torch.save(snapshots,d/f'{ordername}_{arm}_states.pt')
        # Frozen ordering controls must yield equal per-sample logits within numerical tolerance.
        back=np.argsort(preds['shuffle_ids']);np.testing.assert_allclose(preds['storage_frozen'],preds['shuffle_frozen'][back],rtol=0,atol=2e-5)
        np.savez_compressed(d/'predictions.npz',**preds);dump(d/'updates.json',rows)
        record=dict(held=h,checkpoint_sha256=digest(p),checkpoint=str(p),ids=len(ids),files={p.name:digest(p) for p in d.iterdir()});dump(d/'receipt.json',record);records.append(record)
        print(json.dumps(dict(held=h,completed=True,labels_loaded=False)),flush=True)
    dump(a.out/'RUN.json',dict(seconds=time.perf_counter()-started,folds=records,labels_loaded=False,protocol_sha256=digest(a.protocol)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cost',action='store_true');main(p.parse_args())
