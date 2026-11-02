"""Independent direct code-target sklearn fit and explicit-distance audit."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
from audit_restored import digest,dump

def macro(y,p):return float(np.mean([np.mean(p[y==k]==k) for k in np.unique(y)]))

def main(a):
    cfg=json.loads((a.run/'PROTOCOL.json').read_text());manifest=json.loads((a.run/'FILES.json').read_text())
    for n,h in manifest.items():assert digest(a.run/n)==h
    ref=json.loads((a.reference/'FILES.json').read_text());assert json.loads((a.reference/'SOURCE_IDENTITY.json').read_text())['checkpoint_files']==cfg['checkpoint_files']
    y=np.load(a.data/'cache/y_clutterremoved.npy');u=np.load(a.data/'cache/subj_clutterremoved.npy')
    for name,file in [('labels','y'),('users','subj')]:assert digest(a.data/f'cache/{file}_clutterremoved.npy')==cfg['input_sha256'][name]
    ids=np.flatnonzero(~(((u==2)&np.isin(y,[10,11]))|((u==6)&(y==3))));lookup=np.full(len(y),-1);lookup[ids]=np.arange(len(ids))
    code=dict(np.load(a.run/'codes.npz'));base=np.zeros((12,8))
    for i,ls in enumerate([[0],[1],[2],[3],[0,2],[0,3],[1,2],[1,3],[4],[5],[6],[7]]):base[i,ls]=1/np.sqrt(len(ls))
    np.testing.assert_array_equal(code['onehot'],np.eye(12));np.testing.assert_allclose(code['semantic'],np.c_[np.eye(12),base]/np.sqrt(2),atol=1e-15)
    perm=np.random.default_rng(cfg['seed']).permutation(12);np.testing.assert_allclose(code['permuted'],np.c_[np.eye(12),base[perm]]/np.sqrt(2),atol=1e-15)
    np.testing.assert_allclose(code['orthogonal']@code['orthogonal'].T,np.eye(12),atol=1e-14)
    scores={k:[] for k in cfg['arms']+['cnn']};max_error=0.;fits=0
    for h in cfg['source_users']:
        d=a.run/f'held{h}';old=a.reference/f'held{h}'
        for n in ['features.npz','split.json','predictions.npz']:assert digest(old/n)==ref[f'held{h}/{n}']
        parts=json.loads((d/'split.json').read_text());oldsplit=json.loads((old/'split.json').read_text());assert parts==oldsplit
        val=h%8+1
        for k,m in [('train',(u[ids]!=h)&(u[ids]!=val)),('val',u[ids]==val),('diagnostic',u[ids]==h)]:np.testing.assert_array_equal(parts[k],ids[m])
        tr,va,te=[np.array(parts[k]) for k in ['train','val','diagnostic']];xfile=np.load(old/'features.npz');np.testing.assert_array_equal(xfile['ids'],ids);x=xfile['hidden']
        w=np.empty(len(tr));yt=y[tr];ut=u[tr]
        for user in sorted(set(ut)):
            classes=sorted(set(yt[ut==user]))
            for cl in classes:
                mask=(ut==user)&(yt==cl);w[mask]=1/(len(set(ut))*len(classes)*mask.sum())
        train=x[lookup[tr]];mu=np.average(train,axis=0,weights=w);scale=np.sqrt(np.average((train-mu)**2,axis=0,weights=w));scale[scale<=1e-12]=1
        train=(train-mu)/scale;v=(x[lookup[va]]-mu)/scale;test=(x[lookup[te]]-mu)/scale
        saved=np.load(d/'predictions.npz');sel=json.loads((d/'selection.json').read_text())
        for k,z in [('ids',te),('y',y[te]),('validation_ids',va),('validation_y',y[va])]:np.testing.assert_array_equal(saved[k],z)
        regs={};vals={}
        for arm,c in code.items():
            regs[arm]=[];vals[arm]=[]
            for j,lam in enumerate(cfg['lambdas']):
                reg=Ridge(alpha=lam*len(tr),fit_intercept=True,solver='cholesky').fit(train,c[yt],sample_weight=w*len(tr));fits+=1
                vp=reg.predict(v);logit=vp@c.T-.5*np.square(c).sum(1);err=float(abs(logit-saved[f'{arm}_validation_{j}']).max());max_error=max(max_error,err)
                np.testing.assert_allclose(logit,saved[f'{arm}_validation_{j}'],atol=1e-9,rtol=1e-9)
                dist=np.square(vp[:,None,:]-c[None,:,:]).sum(-1);np.testing.assert_array_equal(logit.argmax(1),dist.argmin(1));regs[arm].append(reg);vals[arm].append(vp)
        checkpoint=np.load(d/'models.npz');np.testing.assert_allclose(checkpoint['mean'],mu,atol=1e-12);np.testing.assert_allclose(checkpoint['scale'],scale,atol=1e-12)
        for j,reg in enumerate(regs['onehot']):
            np.testing.assert_allclose(checkpoint['coefs'][j],reg.coef_.T,atol=1e-10,rtol=1e-10);np.testing.assert_allclose(checkpoint['bias'],reg.intercept_,atol=1e-10,rtol=1e-10)
        for arm,c in code.items():
            candidates=[]
            for j,lam in enumerate(cfg['lambdas']):
                pp=vals[arm][j];cls=np.square(pp[:,None,:]-c[None,:,:]).sum(-1).argmin(1)
                candidates.append(dict(macro=macro(y[va],cls),mse=float(np.square(vals['onehot'][j]-np.eye(12)[y[va]]).mean()),lam=lam))
            j=min(range(3),key=lambda j:(-candidates[j]['macro'],candidates[j]['mse'],-candidates[j]['lam']));assert sel[arm]['selected']==j
            for k in range(3):
                for key in candidates[k]:np.testing.assert_allclose(candidates[k][key],sel[arm]['candidates'][k][key],atol=1e-10,rtol=1e-10)
            out=regs[arm][j].predict(test);logit=out@c.T-.5*np.square(c).sum(1);max_error=max(max_error,float(abs(logit-saved[arm]).max()))
            np.testing.assert_allclose(logit,saved[arm],atol=1e-9,rtol=1e-9);pred=np.square(out[:,None,:]-c[None,:,:]).sum(-1).argmin(1);np.testing.assert_array_equal(pred,saved[arm].argmax(1));scores[arm].append(macro(y[te],pred))
        with np.load(old/'predictions.npz') as z:
            np.testing.assert_array_equal(saved['cnn'],z['cnn'])
        scores['cnn'].append(macro(y[te],saved['cnn'].argmax(1)));np.testing.assert_array_equal(saved['orthogonal'].argmax(1),saved['onehot'].argmax(1))
    summary=json.loads((a.run/'SUMMARY.json').read_text())
    for k in scores:np.testing.assert_allclose(scores[k],summary['scores'][k],atol=1e-14);np.testing.assert_allclose(np.mean(scores[k]),summary['means'][k],atol=1e-14)
    decisions=[]
    for k,threshold in cfg['gate']['minimum_mean_gain'].items():
        delta=np.array(scores['semantic'])-scores[k];passed=bool(delta.mean()>=threshold and (delta>1e-12).sum()>=cfg['gate']['minimum_wins'] and delta.min()>=-cfg['gate']['maximum_paired_decline']);assert passed==summary['comparisons'][k]['passed'];decisions.append(passed)
    assert all(decisions)==summary['gate_passed']
    out=dict(state='passed',direct_code_target_sklearn_fits=fits,sha_verified_artifacts=len(manifest),maximum_prediction_absolute_error=max_error,all_saved_predictions_and_selection_verified=True,orthogonal_and_onehot_equivalent=True,gate_recomputed=all(decisions),summary_sha256=digest(a.run/'SUMMARY.json'))
    dump(a.run/'AUDIT.json',out);print(json.dumps(out))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['run','reference','data']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=1):main(a)
