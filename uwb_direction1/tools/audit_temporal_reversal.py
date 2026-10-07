"""Independent conjugate-increment replay, functional CNN and ridge equations."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import hilbert
from threadpoolctl import threadpool_limits
from semantic_code_eightfold_prepare import functional_hidden
from energy_area import source_roles
from audit_restored import digest,dump


def independent_input(raw):
    # Start with the original analytic signal; derive reversed increments.
    z=hilbert(np.asarray(raw,dtype=np.float64),axis=-1);z=z-z.mean(axis=-2,keepdims=True)
    d=(z[...,1:,:]*z[...,:-1,:].conj())[...,::-1,:].conj()
    a=np.sqrt(abs(d))*np.angle(d);a=a.astype(np.float32)
    a=np.concatenate([a[...,:1,:],a],axis=-2)
    amp=np.abs(raw[:,:,::-1,:]).astype(np.float32)
    f=np.stack([amp,a],axis=2);f=(f-f.mean((-2,-1),keepdims=True))/(f.std((-2,-1),keepdims=True)+1e-6)
    f=f.astype(np.float16).astype(np.float32).reshape(len(raw),3,2,90,63,3).mean(-1).astype(np.float16).astype(np.float32)
    return torch.from_numpy(np.concatenate([f,np.zeros_like(f)],axis=2))


def main(root):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);t=time.perf_counter()
    c=json.loads((root/'PROTOCOL.json').read_text());identity=json.loads((root/'IDENTITY.json').read_text());manifest=json.loads((root/'FILES.json').read_text())
    assert digest(root/'PROTOCOL.json')==identity['protocol_sha256']
    for n,h in manifest.items():assert digest(root/n)==h
    for n,h in identity['code_sha256'].items():assert digest(Path(__file__).with_name(n))==h
    data=Path(c['data']);paths={'raw':data/'X_clutterremoved.npy','labels':data/'y_clutterremoved.npy','users':data/'subj_clutterremoved.npy','prepared':Path(c['prepared'])}
    for k,p in paths.items():assert digest(p)==c['input_sha256'][k]
    y=np.load(paths['labels']);users=np.load(paths['users']);raw=np.load(paths['raw'],mmap_mode='r');cache=np.load(paths['prepared'],mmap_mode='r')
    old=Path(c['readout_reference']);feat=Path(c['feature_reference']);assert digest(old/'FILES.json')==c['readout_manifest_sha256'] and digest(feat/'FILES.json')==c['feature_manifest_sha256']
    rm=json.loads((old/'FILES.json').read_text());fm=json.loads((feat/'FILES.json').read_text());q=np.array([1,0,3,2,7,6,5,4,9,8,-1,-1]);scores={k:[] for k in c['arms']};max_hidden=0.;max_score=0.;count=0;fits=0
    for h in c['users']:
        d=root/f'held{h}';parts=source_roles(y,users,h);saved=json.loads((d/'split.json').read_text())
        for role,ids in parts.items():np.testing.assert_array_equal(ids,saved[role])
        tr=parts['train'];va=parts['val'];te=parts['diagnostic'];eligible=np.array([y[i]<10 and q[y[i]] in y[tr[users[tr]==users[i]]] for i in tr]);rid=tr[eligible]
        f=np.load(feat/f'held{h}/features.npz');assert digest(feat/f'held{h}/features.npz')==fm[f'held{h}/features.npz']
        for n in ['models.npz','predictions.npz','selection.json','split.json']:assert digest(old/f'held{h}'/n)==rm[f'held{h}/{n}']
        lookup={int(v):i for i,v in enumerate(f['ids'])};ix={k:np.array([lookup[int(v)] for v in ids]) for k,ids in parts.items()};ff=np.load(d/'reverse_features.npz');np.testing.assert_array_equal(rid,ff['ids'])
        ck=Path(c['models'])/f'held{h}/segment_24.pt';assert digest(ck)==json.loads((d/'result.json').read_text())['checkpoint_sha256']
        rec=torch.load(ck,map_location='cpu',weights_only=False);state={k.removeprefix('module.'):v for k,v in rec['average']['weights'].items() if k.startswith('module.')};aa=[];bb=[]
        for st in range(0,len(rid),c['batch_size']):
            ids=rid[st:st+c['batch_size']];x=independent_input(np.asarray(raw[ids]));z=np.asarray(cache[ids],dtype=np.float32);naive=np.stack([z[:,:,0,::-1],z[:,:,2,::-1],np.zeros_like(z[:,:,0]),np.zeros_like(z[:,:,0])],axis=2)
            with torch.no_grad():aa.append(functional_hidden(x,state).numpy());bb.append(functional_hidden(torch.from_numpy(naive.copy()),state).numpy())
        a=np.concatenate(aa).astype(np.float64);b=np.concatenate(bb).astype(np.float64);count+=len(rid)
        for value,key in [(a,'reverse'),(b,'naive')]:
            max_hidden=max(max_hidden,float(abs(value-ff[key]).max()));np.testing.assert_allclose(value,ff[key],atol=2e-5,rtol=2e-5)
        m=np.load(d/'models.npz');pred=np.load(d/'predictions.npz');truth=np.load(d/'evaluation.npz');sel=json.loads((d/'selection.json').read_text())
        for role,ids in [('val',va),('diagnostic',te)]:np.testing.assert_array_equal(pred[role+'_ids'],ids);np.testing.assert_array_equal(truth[role+'_y'],y[ids])
        for arm in c['arms'][1:]:
            x=np.concatenate([f['hidden'][ix['train']],b if arm=='naive_mix' else a]);yy=np.r_[y[tr],y[rid] if arm=='unmapped_mix' else q[y[rid]]];uu=np.r_[users[tr],users[rid]]
            w=np.array([1/(len(np.unique(uu))*len(np.unique(yy[uu==u]))*np.sum((uu==u)&(yy==cl))) for u,cl in zip(uu,yy)])
            mean=np.average(x,axis=0,weights=w);scale=np.sqrt(np.average((x-mean)**2,axis=0,weights=w));scale[scale<=1e-12]=1.;z=(x-mean)/scale;bias=w@np.eye(12)[yy]
            candidates=[];pp=[]
            for j,lam in enumerate(c['lambdas']):
                # Explicit augmented intercept equations, unlike production centered solve.
                design=np.column_stack([np.ones(len(z)),z]);pen=np.diag(np.r_[0.,np.repeat(lam,128)])
                coef=np.linalg.solve(design.T@(w[:,None]*design)+pen,design.T@(w[:,None]*np.eye(12)[yy]));fits+=1
                out={role:np.column_stack([np.ones(len(ix[role])),(f['hidden'][ix[role]]-mean)/scale])@coef for role in ['val','diagnostic']};pp.append(out);v=out['val'];predicted=v.argmax(1)
                ma=float(np.mean([np.mean(predicted[y[va]==k]==k) for k in np.unique(y[va])]))
                candidates.append(dict(lam=lam,macro=ma,mse=float(np.mean((v-np.eye(12)[y[va]])**2))))
                prod=pred[arm+'_val_'+str(lam)];max_score=max(max_score,float(abs(v-prod).max()));np.testing.assert_allclose(v,prod,atol=2e-5,rtol=2e-5);np.testing.assert_array_equal(v.argmax(1),prod.argmax(1))
                # Stored parameters must reproduce saved predictions independently too.
                np.testing.assert_allclose(mean,m[arm+'_mean'],atol=2e-5,rtol=2e-5);np.testing.assert_allclose(scale,m[arm+'_scale'],atol=2e-5,rtol=2e-5)
                np.testing.assert_allclose(bias,m[arm+'_bias'],atol=1e-12,rtol=1e-12)
                stored=((f['hidden'][ix['val']]-m[arm+'_mean'])/m[arm+'_scale'])@m[arm+'_coefs'][j]+m[arm+'_bias'];np.testing.assert_allclose(stored,prod,atol=1e-12,rtol=1e-12)
            chosen=min(range(3),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']));assert chosen==sel[arm]['selected']
            for j,can in enumerate(candidates):
                assert can['macro']==sel[arm]['candidates'][j]['macro'];assert abs(can['mse']-sel[arm]['candidates'][j]['mse'])<1e-7
            for role in ['val','diagnostic']:
                v=pp[chosen][role];prod=pred[role+'_'+arm];max_score=max(max_score,float(abs(v-prod).max()));np.testing.assert_allclose(v,prod,atol=2e-5,rtol=2e-5);np.testing.assert_array_equal(v.argmax(1),prod.argmax(1))
                if role=='diagnostic':scores[arm].append(float(np.mean([np.mean(v.argmax(1)[y[te]==k]==k) for k in np.unique(y[te])])) )
        om=np.load(old/f'held{h}/models.npz');os=json.loads((old/f'held{h}/selection.json').read_text())['onehot']['selected'];ot=np.load(old/f'held{h}/predictions.npz')
        base=((f['hidden'][ix['diagnostic']]-om['mean'])/om['scale'])@om['coefs'][os]+om['bias'];np.testing.assert_allclose(base,ot['onehot']+.5,atol=1e-12,rtol=1e-12);np.testing.assert_array_equal(base.argmax(1),pred['diagnostic_ridge'].argmax(1));scores['ridge'].append(float(np.mean([np.mean(base.argmax(1)[y[te]==k]==k) for k in np.unique(y[te])])))
        print(json.dumps(dict(held=h,checked_training_reversals=len(rid),max_hidden=max_hidden,max_score=max_score)),flush=True)
    s=json.loads((root/'SUMMARY.json').read_text());passed=[]
    for k,v in scores.items():np.testing.assert_allclose(v,s['scores'][k],atol=1e-15,rtol=0);assert abs(np.mean(v)-s['means'][k])<1e-15
    for k,g in c['gate']['minimum_mean_gain'].items():
        delta=np.array(scores[c['primary']])-scores[k];yes=bool(delta.mean()>=g and (delta>1e-12).sum()>=c['gate']['minimum_wins'] and delta.min()>=-c['gate']['maximum_paired_decline']);assert yes==s['comparisons'][k]['passed'];passed.append(yes)
    assert all(passed)==s['gate_passed']
    out=dict(state='passed',artifact_hashes=len(manifest),independent_reversed_training_clips=count,functional_forward_clips=2*count,independent_ridge_fits=fits,max_hidden_error=max_hidden,max_score_error=max_score,all_argmax_equal=True,seconds=time.perf_counter()-t)
    dump(root/'AUDIT.json',out);print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    with threadpool_limits(limits=1):main(p.parse_args().root)
