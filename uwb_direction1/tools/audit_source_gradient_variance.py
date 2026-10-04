"""Independent centered NumPy variance/null replay and source-role audit."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax,logsumexp
from threadpoolctl import threadpool_limits
from audit_source_sam import functional_forward
from audit_restored import digest,dump


def centered(g,d):
    v=np.stack([np.square(g[d==x]-g[d==x].mean(0)).mean(0) for x in np.unique(d)])
    return v,float(np.square(v-v.mean(0)).mean())


def main(a):
    start=time.perf_counter();torch.set_num_threads(1);c=json.loads((a.runs/'protocol.json').read_text());s=json.loads((a.runs/'SUMMARY.json').read_text());rows=json.loads((a.runs/'ROWS.json').read_text())
    assert s['state']=='complete' and len(rows)==18
    for name,h in s['files'].items():assert digest(a.runs/name)==h
    cache=np.load(c['inputs']['cache'],mmap_mode='r');y=np.load(c['inputs']['labels']);u=np.load(c['inputs']['users']);qualified=[];audits=[]
    for row in rows:
        h=row['held'];e=row['epoch'];v=h+1 if h<8 else 3;p=Path(c['models'])/f'held{h}'/f'segment_{e:02d}.pt';assert digest(p)==c['model_sha256'][f'{h}/{e}']
        cp=torch.load(p,weights_only=False,map_location='cpu');state={k.removeprefix('module.'):vv for k,vv in cp['average']['weights'].items() if k.startswith('module.')}
        with np.load(a.runs/f'held{h}_epoch{e:02d}.npz') as saved:
            ids=saved['ids'];halves=saved['half'];labels=saved['labels'];domains=saved['users'];z=saved['features'];logits=saved['logits']
            assert len(np.unique(ids))==len(ids) and not set(u[ids])&{h,v,1,2}
            np.testing.assert_array_equal(labels,y[ids]);np.testing.assert_array_equal(domains,u[ids])
            eligible=(u>=3)&(u<=8)&(u!=h)&(u!=v)&~((u==6)&(y==3));sourceusers=np.unique(u[eligible]);classes=sorted(set.intersection(*[set(y[eligible&(u==d)]) for d in sourceusers]))
            expected=[];half=[]
            for d in sourceusers:
                for k in classes:
                    pool=np.flatnonzero(eligible&(u==d)&(y==k));choice=np.random.default_rng(c['seed']+10000*h+100*int(d)+int(k)).permutation(pool)[:48]
                    expected.extend(choice);half.extend([0]*24+[1]*24)
            np.testing.assert_array_equal(ids,expected);np.testing.assert_array_equal(halves,half)
            for off in [0,((len(ids)-1)//c['batch'])*c['batch']]:
                raw=np.asarray(cache[ids[off:off+c['batch']]],dtype=np.float32);x=torch.from_numpy(np.stack([raw[:,:,0],raw[:,:,2],np.zeros_like(raw[:,:,0]),np.zeros_like(raw[:,:,0])],axis=2))
                with torch.no_grad():pred=functional_forward(x,state,'shared128').numpy()
                np.testing.assert_array_equal(pred,logits[off:off+c['batch']])
            w=state['head.4.weight'].numpy().astype(np.float64);b=state['head.4.bias'].numpy().astype(np.float64);zd=z.astype(np.float64)
            residual=softmax(zd@w.T+b,axis=1);residual[np.arange(len(ids)),labels]-=1
            g=np.concatenate([np.einsum('nc,nd->ncd',residual,zd).reshape(len(ids),-1),residual],axis=1)
            sections={};matrices={}
            for partnum,(part,mask) in enumerate([('full',np.ones(len(ids),dtype=bool)),('half0',halves==0),('half1',halves==1)]):
                gg=g[mask];dd=domains[mask];yy=labels[mask];vv,value=centered(gg,dd);matrices[part]=vv
                np.testing.assert_allclose(vv,saved['variance_'+part],rtol=1e-7,atol=1e-24)
                rng=np.random.default_rng(c['seed']+10000*h+100*e+partnum);null=[]
                for _ in range(c['permutations']):
                    shuffled=dd.copy()
                    for k in np.unique(yy):
                        pos=np.flatnonzero(yy==k);shuffled[pos]=rng.permutation(dd[pos])
                    null.append(centered(gg,shuffled)[1])
                np.testing.assert_allclose(null,saved['null_'+part],rtol=1e-7,atol=1e-26)
                n=np.array(null);fraction=float((1+(n>=value).sum())/100);ratio=float(value/max(np.median(n),1e-300));old=row['sections'][part]
                np.testing.assert_allclose([value,np.median(n),np.quantile(n,.95),ratio],[old['statistic'],old['null_median'],old['null_q95'],old['ratio_to_null_median']],rtol=1e-7,atol=1e-26)
                assert fraction==old['permutation_exceedance_fraction'];sections[part]=(value,fraction,ratio,float(np.quantile(n,.95)))
            aa=(matrices['half0']-matrices['half0'].mean(0)).ravel();bb=(matrices['half1']-matrices['half1'].mean(0)).ravel();cos=float(aa@bb/max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-300));assert abs(cos-row['half_centered_variance_cosine'])<1e-7
            gate=c['gate'];passed=(sections['full'][1]<=gate['max_exceedance_fraction'] and sections['full'][2]>=gate['minimum_full_ratio'] and cos>=gate['minimum_half_cosine'] and all(sections[q][0]>sections[q][3] for q in ['half0','half1']))
            assert passed==row['qualifies']
            ce=float(np.mean(logsumexp(logits.astype(np.float64),axis=1)-logits[np.arange(len(ids)),labels]));assert abs(ce-row['source_ce'])<1e-12
            audits.append({'held':h,'epoch':e,'source_ids_verified':True,'functional_forward_first_last_batches_bitwise_equal':True,'all_297_null_statistics_independently_recomputed':True,'variance_and_gate_recomputed':True})
        print(json.dumps({'held':h,'epoch':e,'audited':True}),flush=True)
    qualifies=[h for h in c['source_users'] if all(next(r for r in rows if r['held']==h and r['epoch']==e)['qualifies'] for e in c['gate']['epochs'])]
    assert qualifies==s['qualifying_trajectories'] and (len(qualifies)>=c['gate']['min_trajectories'])==s['mechanism_gate_passed']
    dump(a.out,{'state':'passed','seconds':time.perf_counter()-start,'models':18,'null_statistics_recomputed':18*3*99,'runs':audits,'mechanism_gate_passed':s['mechanism_gate_passed'],'classification_gain_measured':False})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 with threadpool_limits(limits=1):main(a)
