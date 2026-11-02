"""Recover frozen eightfold hidden features with independent functional replay."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump,expected_parts
from circular_signal import SignalNet,get_batch


def roles(y,u,h):
    p=expected_parts(y,u,h);return dict(train=p['train'],val=p['val'],diagnostic=p['test'])


@torch.no_grad()
def functional_hidden(x,s):
    n=len(x);x=x.reshape(n*3,4,90,63)
    for stage in range(3):
        b=f'encoder.{stage}.layers.'
        x=F.gelu(F.group_norm(F.conv2d(x,s[b+'0.weight'],stride=2,padding=1),4,s[b+'1.weight'],s[b+'1.bias']))
        x=F.gelu(F.group_norm(F.conv2d(x,s[b+'3.weight'],padding=1),4,s[b+'4.weight'],s[b+'4.bias']))
    x=F.adaptive_avg_pool2d(x,(3,4)).reshape(n,-1)
    return F.gelu(F.linear(x,s['head.1.weight'],s['head.1.bias']))


@torch.no_grad()
def extract(model,state,cache,ids):
    hh=[];ll=[];maxerr=0.
    for i in range(0,len(ids),64):
        x=get_batch(cache,ids[i:i+64],'legacy');z=model.head[:3](model.encoder(x.reshape(-1,4,90,63)).reshape(len(x),-1));f=functional_hidden(x,state)
        np.testing.assert_array_equal(z.numpy(),f.numpy());l=F.linear(f,state['head.4.weight'],state['head.4.bias']);np.testing.assert_array_equal(model.head[3:](z).numpy(),l.numpy())
        hh.append(z.numpy());ll.append(l.numpy())
    return np.concatenate(hh).astype(np.float64),np.concatenate(ll)


def main(a):
    cfg=json.loads(a.protocol.read_text());assert not a.out.exists();a.out.mkdir();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    for n,h in cfg['checkpoint_files'].items():assert digest(a.models/n)==h
    files={k:a.data/'cache'/n for k,n in [('raw','X_clutterremoved.npy'),('labels','y_clutterremoved.npy'),('users','subj_clutterremoved.npy')]};files['prepared']=a.prepared/'signals.npy'
    hashes={k:digest(p) for k,p in files.items()};assert hashes==cfg['input_sha256']
    y=np.load(files['labels']);u=np.load(files['users']);cache=np.load(files['prepared'],mmap_mode='r');ids=np.flatnonzero(~(((u==2)&np.isin(y,[10,11]))|((u==6)&(y==3))));assert len(ids)==9300
    lookup=np.full(len(y),-1);lookup[ids]=np.arange(len(ids));identity=dict(checkpoint_files=cfg['checkpoint_files'],input_sha256=hashes,protocol_sha256=digest(a.protocol),torch=torch.__version__,code_sha256={n:digest(Path(__file__).with_name(n)) for n in ['semantic_code_eightfold_prepare.py','circular_signal.py','audit_restored.py']});dump(a.out/'SOURCE_IDENTITY.json',identity)
    if not a.cost_only:
        assert json.loads(a.cost.read_text())['passed'];assert json.loads(a.cost.read_text())['protocol_sha256']==identity['protocol_sha256']
    start=time.perf_counter();audit=[]
    for h in ([1] if a.cost_only else range(1,9)):
        old=a.models/f'fold{h}/legacy';ck=torch.load(old/'best.pt',map_location='cpu',weights_only=False);result=json.loads((old/'result.json').read_text());best=min(result['history'],key=lambda z:z['validation_ce']);assert best['epoch']==ck['epoch']==result['selected_epoch'];assert ck['validation_ce']==best['validation_ce']
        net=SignalNet();net.load_state_dict(ck['model']);net.eval();parts=roles(y,u,h);saved=json.loads((old/'split.json').read_text())
        for k,v in parts.items():np.testing.assert_array_equal(v,saved['test' if k=='diagnostic' else k])
        if a.cost_only:
            t=time.perf_counter();extract(net,ck['model'],cache,parts['train'][:64]);elapsed=time.perf_counter()-t;estimate=elapsed*8*9300/64*2+1
            out=dict(source_train64_seconds=elapsed,estimate_seconds=estimate,budget=cfg['budget_seconds'],passed=estimate<cfg['budget_seconds'],protocol_sha256=identity['protocol_sha256'],no_validation_or_diagnostic_forward=True)
            dump(a.out/'COST.json',out);print(json.dumps(out));return
        hidden,logits=extract(net,ck['model'],cache,ids);errors={}
        with np.load(old/'predictions.npz') as z:
            for role,key,ikey in [('val','validation_logits','validation_ids'),('diagnostic','logits','ids')]:
                np.testing.assert_array_equal(z[ikey],parts[role]);new=logits[lookup[parts[role]]];errors[role]=float(abs(new-z[key]).max());np.testing.assert_allclose(new,z[key],atol=2e-5,rtol=2e-5);np.testing.assert_array_equal(new.argmax(1),z[key].argmax(1))
            pred=dict(ids=parts['diagnostic'],y=y[parts['diagnostic']],cnn=z['logits'])
        d=a.out/f'held{h}';d.mkdir();np.savez_compressed(d/'features.npz',ids=ids,hidden=hidden,cnn=logits);np.savez_compressed(d/'predictions.npz',**pred);dump(d/'split.json',{k:v.tolist() for k,v in parts.items()})
        audit.append(dict(held=h,selected_epoch=ck['epoch'],all9300_hidden_functional_replay_bitwise_equal=True,legacy_saved_logit_max_errors=errors,source_checkpoint_sha256=cfg['checkpoint_files'][f'fold{h}/legacy/best.pt']));print(json.dumps({'extracted':h}),flush=True)
    dump(a.out/'FEATURE_AUDIT.json',dict(state='passed',rows=audit,seconds=time.perf_counter()-start,old_logit_tolerance='atol2e-5 rtol2e-5: original3threads vs current1 and batch composition; argmax exact'))
    dump(a.out/'FILES.json',{str(p.relative_to(a.out)):digest(p) for p in sorted(a.out.rglob('*')) if p.is_file()})
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['protocol','models','data','prepared','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cost-only',action='store_true');p.add_argument('--cost',type=Path);a=p.parse_args()
    with threadpool_limits(limits=1):main(a)
