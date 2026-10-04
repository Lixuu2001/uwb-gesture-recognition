"""Frozen source-role-only gradient variance diagnostic; no fitting or held scores."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from circular_signal import SignalNet,get_batch
from linear_gradient_variance import gradient_rows
from audit_restored import digest,dump


def sample_source(y,u,h,n,seed):
    v=h+1 if h<8 else 3
    role=np.flatnonzero((u>=3)&(u<=8)&(u!=h)&(u!=v)&~((u==6)&(y==3)))
    domains=np.unique(u[role]);classes=sorted(set.intersection(*[set(y[role[u[role]==d]]) for d in domains]))
    ids=[];half=[]
    for d in domains:
        for k in classes:
            pool=role[(u[role]==d)&(y[role]==k)]
            if len(pool)<n:raise ValueError('Insufficient class/user samples for fixed diagnostic')
            chosen=np.random.default_rng(seed+10000*h+100*int(d)+int(k)).permutation(pool)[:n]
            ids.extend(chosen.tolist());half.extend([0]*(n//2)+[1]*(n//2))
    return role,np.array(ids,dtype=np.int64),np.array(half,dtype=np.int64),[int(k) for k in classes]


def variance_matrix(g,d):
    groups=np.unique(d);m=(groups[:,None]==d[None,:]).astype(np.float64);m/=m.sum(1,keepdims=True)
    first=m@g;second=m@(g*g);v=second-first*first
    assert v.min()>=-max(float(second.max()),1e-300)*1e-12
    return np.maximum(v,0)


def score(v):return float(np.square(v-v.mean(0)).mean())


def permute_domains(y,d,rng):
    out=d.copy()
    for k in np.unique(y):
        ids=np.flatnonzero(y==k);out[ids]=rng.permutation(d[ids])
    return out


def diagnose(g,y,d,seed,permutations):
    observed=variance_matrix(g,d)
    direct=np.stack([g[d==u].var(0) for u in np.unique(d)])
    np.testing.assert_allclose(observed,direct,rtol=1e-8,atol=1e-22)
    stat=score(observed);rng=np.random.default_rng(seed);null=[]
    for i in range(permutations):
        shuffled=permute_domains(y,d,rng)
        v=variance_matrix(g,shuffled)
        if i==0:np.testing.assert_allclose(v,np.stack([g[shuffled==u].var(0) for u in np.unique(d)]),rtol=1e-8,atol=1e-22)
        null.append(score(v))
    null=np.array(null)
    return {'statistic':stat,'null_median':float(np.median(null)),'null_q95':float(np.quantile(null,.95)),
            'permutation_exceedance_fraction':float((1+(null>=stat).sum())/(1+len(null))),
            'ratio_to_null_median':float(stat/max(np.median(null),1e-300)),
            'mean_gradient_variance':float(observed.mean()),'mean_squared_gradient':float(np.square(g).mean())},observed,null


@torch.no_grad()
def features(net,cache,ids,batch):
    zs=[];logits=[]
    for start in range(0,len(ids),batch):
        x=get_batch(cache,ids[start:start+batch],'legacy');n=len(x)
        encoded=net.encoder(x.reshape(n*3,4,90,63)).reshape(n,3,-1)
        z=net.head[:4](encoded);out=net.head[4](z)
        if start==0:torch.testing.assert_close(out,net(x),rtol=0,atol=0)
        zs.append(z);logits.append(out)
    return torch.cat(zs),torch.cat(logits)


def main(a):
    c=json.loads(a.protocol.read_text());torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    root=Path(__file__).resolve().parent
    for f,sha in c['code_sha256'].items():assert digest(root/f)==sha,f
    if a.out.exists():raise ValueError('Fresh output required')
    paths={k:Path(v) for k,v in c['inputs'].items()}
    for k,path in paths.items():assert digest(path)==c['input_sha256'][k]
    cache=np.load(paths['cache'],mmap_mode='r');y=np.load(paths['labels']);u=np.load(paths['users'])
    a.out.mkdir();dump(a.out/'protocol.json',c);start=time.perf_counter();rows=[];model_hashes=[]
    for h in c['source_users']:
        role,ids,half,classes=sample_source(y,u,h,c['per_class_user'],c['seed'])
        folder=Path(c['models'])/f'held{h}'
        np.testing.assert_array_equal(role,json.loads((folder/'split.json').read_text())['train'])
        assert not set(u[ids])&{h,h+1 if h<8 else 3,1,2}
        for epoch in c['epochs']:
            path=folder/f'segment_{epoch:02d}.pt';sha=digest(path);assert sha==c['model_sha256'][f'{h}/{epoch}']
            cp=torch.load(path,weights_only=False,map_location='cpu');assert cp['epoch']==epoch and cp['identity']['held']==h
            state={k.removeprefix('module.'):v for k,v in cp['average']['weights'].items() if k.startswith('module.')}
            net=SignalNet();net.load_state_dict(state);net.eval()
            if a.cost:
                t=time.perf_counter();features(net,cache,ids[:64],64);infer=time.perf_counter()-t
                rng=np.random.default_rng(c['seed']);g=rng.normal(size=(2304,1548));yy=np.tile(np.repeat(np.arange(12),48),4);dd=np.repeat(np.arange(4),576)
                t=time.perf_counter();diagnose(g,yy,dd,c['seed'],9);stat=time.perf_counter()-t
                estimate=2*18*(infer*2304/64+3*stat*c['permutations']/9)
                cost={'source_inference_seconds64':infer,'synthetic_full_stat_seconds9':stat,'estimate_seconds_with_100pct_headroom':estimate,'budget_seconds':c['budget_seconds'],'passed':estimate<=c['budget_seconds'],'source_train_ids':ids[:64].tolist(),'uwb_gradient_distribution_evaluated':False}
                dump(a.out/'COST.json',cost);print(json.dumps(cost));return
            z,logits=features(net,cache,ids,c['batch'])
            weight=net.head[4].weight.detach().double();bias=net.head[4].bias.detach().double()
            g=gradient_rows(z.double(),weight,bias,torch.from_numpy(y[ids]).long()).detach().numpy()
            # Independent autograd on actual source examples, including parameter rows.
            w=weight.clone().requires_grad_();b=bias.clone().requires_grad_()
            for j in [0,len(ids)-1]:
                loss=F.cross_entropy(F.linear(z[j:j+1].double(),w,b),torch.from_numpy(y[ids[j:j+1]]).long())
                gw,gb=torch.autograd.grad(loss,(w,b));ref=torch.cat((gw.flatten(),gb)).detach().numpy()
                np.testing.assert_allclose(g[j],ref,rtol=1e-8,atol=1e-12)
            sections={};vs={};nulls={}
            for part,mask in [('full',np.ones(len(ids),dtype=bool)),('half0',half==0),('half1',half==1)]:
                stat,v,null=diagnose(g[mask],y[ids][mask],u[ids][mask],c['seed']+10000*h+100*epoch+len(sections),c['permutations'])
                sections[part]=stat;vs[part]=v;nulls[part]=null
            x=(vs['half0']-vs['half0'].mean(0)).ravel();zz=(vs['half1']-vs['half1'].mean(0)).ravel();cos=float(x@zz/max(np.linalg.norm(x)*np.linalg.norm(zz),1e-300))
            gate=c['gate'];qualifies=(sections['full']['permutation_exceedance_fraction']<=gate['max_exceedance_fraction'] and sections['full']['ratio_to_null_median']>=gate['minimum_full_ratio'] and cos>=gate['minimum_half_cosine'] and all(sections[k]['statistic']>sections[k]['null_q95'] for k in ['half0','half1']))
            ce=float(F.cross_entropy(logits.double(),torch.from_numpy(y[ids]).long()))
            row={'held':h,'epoch':epoch,'n':len(ids),'source_users':np.unique(u[ids]).tolist(),'common_classes':classes,'sections':sections,'half_centered_variance_cosine':cos,'qualifies':bool(qualifies),'source_ce':ce,'reference_1000_penalty_over_ce':1000*sections['full']['statistic']/max(ce,1e-300)}
            rows.append(row);model_hashes.append({'held':h,'epoch':epoch,'sha256':sha})
            np.savez_compressed(a.out/f'held{h}_epoch{epoch:02d}.npz',ids=ids,half=half,labels=y[ids],users=u[ids],features=z.numpy(),logits=logits.numpy(),**{'variance_'+k:v for k,v in vs.items()},**{'null_'+k:v for k,v in nulls.items()})
            print(json.dumps({'held':h,'epoch':epoch,'complete':True}),flush=True)
    dump(a.out/'ROWS.json',rows);dump(a.out/'MODELS.json',model_hashes)
    qualifying=[h for h in c['source_users'] if all(next(r for r in rows if r['held']==h and r['epoch']==e)['qualifies'] for e in c['gate']['epochs'])]
    summary={'state':'complete','seconds':time.perf_counter()-start,'qualifying_trajectories':qualifying,'mechanism_gate_passed':len(qualifying)>=c['gate']['min_trajectories'],'models':len(rows),'source_rows':sum(r['n'] for r in rows),'new_training':False,'held_signal_accessed':False,'classification_gain_measured':False,'files':{p.name:digest(p) for p in a.out.iterdir() if p.is_file()}}
    dump(a.out/'SUMMARY.json',summary);print(json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--cost',action='store_true');a=p.parse_args()
    with threadpool_limits(limits=1):main(a)
