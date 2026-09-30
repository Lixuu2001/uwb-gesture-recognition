"""Source-only diagnostic of absolute versus relative range-bin centroids.

No target signal or target performance is read, no network training/selection.
Coordinates are bin indices, not calibrated distances or 3D trajectories.
"""
import argparse
from pathlib import Path
import numpy as np
from audit_restored import expected_parts,dump,digest


def residual(y,x):
    coef=np.linalg.lstsq(x,y,rcond=None)[0]
    return ((y-x@coef)**2).sum(axis=0)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Use a new diagnostic directory')
    labels=np.load(args.data/'cache/y_clutterremoved.npy')
    users=np.load(args.data/'cache/subj_clutterremoved.npy')
    ids=expected_parts(labels,users,1)['train']
    assert not np.isin(users[ids],[1,2]).any()
    raw=np.load(args.data/'cache/X_clutterremoved.npy',mmap_mode='r')
    bins=np.arange(189,dtype=np.float64)
    centers=[];spreads=[];temporal=[]
    for start in range(0,len(ids),32):
        a=np.abs(np.asarray(raw[ids[start:start+32]],dtype=np.float64))
        den=a.sum(axis=(-2,-1))+1e-12
        mean=(a*bins).sum(axis=(-2,-1))/den
        var=(a*(bins[None,None,None,:]-mean[:,:,None,None])**2).sum(axis=(-2,-1))/den
        frame=(a*bins).sum(-1)/(a.sum(-1)+1e-12)
        centers.append(mean);spreads.append(np.sqrt(var));temporal.append(frame.std(-1))
    center=np.concatenate(centers);spread=np.concatenate(spreads);tsd=np.concatenate(temporal)
    source_y=labels[ids];source_u=users[ids]
    user_design=np.stack([source_u==u for u in np.unique(source_u)],axis=1).astype(float)
    class_design=np.eye(12)[source_y]
    total=((center-center.mean(0))**2).sum(0)
    user_sse=residual(center,user_design);class_sse=residual(center,class_design)
    both_sse=residual(center,np.concatenate([user_design,class_design],axis=1))
    report={'scope':'fold1 six source users only; descriptive diagnostic, no classifier evidence',
        'source_users':np.unique(source_u).tolist(),'source_samples':len(ids),'coordinates':'fast-time/range-bin indices, not calibrated distance',
        'mean_centroid_bins_by_user':{str(u):center[source_u==u].mean(0).tolist() for u in np.unique(source_u)},
        'centroid_std_bins_all_source':center.std(0).tolist(),
        'within_clip_temporal_centroid_std_bins_mean':tsd.mean(0).tolist(),
        'within_clip_range_spread_bins_mean':spread.mean(0).tolist(),
        'r2_class_only':(1-class_sse/total).tolist(),'r2_user_only':(1-user_sse/total).tolist(),
        'additional_total_variance_explained_by_user_after_class':((class_sse-both_sse)/total).tolist(),
        'additional_total_variance_explained_by_class_after_user':((user_sse-both_sse)/total).tolist(),
        'interpretation_limits':['in-sample additive regression, not causal style identification',
             'class and user interactions omitted; constant re-centering may also remove class information',
             'centroids mix multipath/scatterers; not ground-truth hand position',
             'no accuracy gain, safe augmentation, or novelty established'],
        'script_sha256':digest(__file__)}
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out/'source_centroids.npz',ids=ids,users=source_u,labels=source_y,centroid=center,
                        range_spread=spread,temporal_centroid_std=tsd)
    report['source_features_sha256']=digest(args.out/'source_centroids.npz')
    dump(args.out/'DIAGNOSTIC.json',report)
    print(__import__('json').dumps(report))


if __name__=='__main__':main()
