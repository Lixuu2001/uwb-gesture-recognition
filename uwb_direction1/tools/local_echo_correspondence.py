"""One-dimensional envelope correspondence screen; not measured hand velocity.

Independent NumPy patch matching, no learned parameters or external flow code.
No time reversal is treated as a class-preserving augmentation.
"""
import numpy as np
from scipy.signal import hilbert


def envelope(raw):
    return np.abs(hilbert(np.asarray(raw,dtype=np.float64),axis=-1))


def scores(a,b,centers,radius,max_shift):
    """[pairs,range], [pairs,centers] -> centered cosine cost volume."""
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    centers=np.asarray(centers,dtype=np.int64)
    if a.ndim!=2 or b.shape!=a.shape or centers.shape[0]!=len(a):raise ValueError('Bad shapes')
    offsets=np.arange(-radius,radius+1);shifts=np.arange(-max_shift,max_shift+1)
    if np.any(centers-radius-max_shift<0) or np.any(centers+radius+max_shift>=a.shape[1]):raise ValueError('Out of range')
    source=a[np.arange(len(a))[:,None,None],centers[:,:,None]+offsets]
    target=b[np.arange(len(a))[:,None,None,None],centers[:,:,None,None]+shifts[None,None,:,None]+offsets]
    source=source-source.mean(-1,keepdims=True);target=target-target.mean(-1,keepdims=True)
    an=np.linalg.norm(source,axis=-1);bn=np.linalg.norm(target,axis=-1)
    den=an[:,:,None]*bn
    cost=np.divide(np.sum(source[:,:,None,:]*target,axis=-1),den,out=np.full(den.shape,-2.),where=den>1e-24)
    return np.clip(cost,-2,1),an,bn


def match(a,b,centers,c):
    if np.ndim(centers)==1:centers=np.broadcast_to(centers,(len(a),len(centers)))
    volume,an,bn=scores(a,b,centers,c['patch_radius'],c['max_shift'])
    k=volume.argmax(-1);shift=k-c['max_shift'];peak=np.take_along_axis(volume,k[:,:,None],-1)[...,0]
    gap=peak-np.partition(volume,-2,axis=-1)[...,-2]
    reverse,_,_=scores(b,a,centers+shift,c['patch_radius'],c['max_shift'])
    rk=reverse.argmax(-1);back=rk-c['max_shift'];cycle=np.abs(shift+back)
    rpeak=np.take_along_axis(reverse,rk[:,:,None],-1)[...,0]
    rgap=rpeak-np.partition(reverse,-2,axis=-1)[...,-2]
    reference=np.std(a,axis=-1)[:,None]*np.sqrt(2*c['patch_radius']+1)
    eligible=(reference>1e-12)&(an>=c['texture_fraction']*reference)
    # Reject search endpoints: their optimum could lie beyond the fixed search.
    good=eligible&(peak>=c['min_correlation'])&(gap>=c['min_gap'])&(rpeak>=c['min_correlation'])&(rgap>=c['min_gap'])&(cycle<=c['max_cycle_error'])&(np.abs(shift)<c['max_shift'])&(np.abs(back)<c['max_shift'])
    return dict(cost=volume,reverse_cost=reverse,shift=shift,back=back,peak=peak,gap=gap,cycle=cycle,eligible=eligible,reliable=good,moving=good&(shift!=0))


def select_ids(y,u,c):
    result=[]
    for subject in c['source_users']:
        good=(u==subject)&~((u==6)&(y==3))
        for label in sorted(np.unique(y[good]).tolist()):
            pool=np.flatnonzero(good&(y==label))
            rng=np.random.default_rng(np.random.SeedSequence([c['seed'],int(subject),int(label)]))
            result.extend(sorted(rng.choice(pool,min(c['per_user_class'],len(pool)),replace=False).tolist()))
    return np.array(result,dtype=np.int64)


def summarize(ids,users,arrays,c):
    rows=[]
    for user in c['source_users']:
        idx=users[ids]==user
        for view in range(3):
            row={'source_user':user,'view':view,'records':int(idx.sum())}
            for arm in ['adjacent','distant']:
                e=arrays[arm+'_eligible'][idx,view];g=arrays[arm+'_reliable'][idx,view];m=arrays[arm+'_moving'][idx,view]
                # Per-clip fractions first; every clip equal weight within user/view.
                denominator=e.sum(axis=(1,2));valid=denominator>0
                row[arm+'_eligible_records']=int(valid.sum())
                for name,mask in [('reliable',g),('moving',m)]:
                    fraction=np.divide(mask.sum(axis=(1,2)),denominator,out=np.zeros(len(e),dtype=float),where=denominator>0)
                    row[arm+'_'+name+'_fraction']=float(fraction.mean())
            row['reliable_gap']=row['adjacent_reliable_fraction']-row['distant_reliable_fraction']
            row['passed']=row['adjacent_reliable_fraction']>=c['gate']['min_reliable'] and row['reliable_gap']>=c['gate']['min_gap_over_distant'] and row['adjacent_moving_fraction']>=c['gate']['min_moving']
            rows.append(row)
    wins={str(v):sum(r['passed'] for r in rows if r['view']==v) for v in range(3)}
    return {'rows':rows,'passing_users_per_view':wins,'gate_passed':sum(n>=c['gate']['users_per_view'] for n in wins.values())>=c['gate']['views_required'],'classification_evaluated':False,'novelty_claim':False}
