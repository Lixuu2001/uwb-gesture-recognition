"""Source-level duplicate/label and view-count audit. No performance-based filtering."""
import numpy as np
import scipy.io as sio
from pipeline import BASE,ROOT,digest,dump,quality_mask

def main():
    import hashlib
    root=ROOT/'12652592';out={'mat_duplicates':[],'per_view_duplicates':{},'mat_view_counts':[]}
    for typ in ['ClutterRemoved','RawData']:
        p=next(root.rglob('HV_02_G11_'+typ+'.mat'));q=next(root.rglob('HV_02_G12_'+typ+'.mat'))
        a=sio.loadmat(p);b=sio.loadmat(q)
        for view in ['Left','Top','Right']:
            aa=next(v for k,v in a.items() if 'Radar'+view in k);bb=next(v for k,v in b.items() if 'Radar'+view in k)
            out['mat_duplicates'].append({'type':typ,'view':view,'equal_float64':np.array_equal(aa,bb),
                'max_abs_diff':float(np.max(np.abs(aa-bb))),'G11':str(p),'G12':str(q),'hash11':digest(p),'hash12':digest(q)})
        p=next(root.rglob('HV_06_G4_'+typ+'.mat'))
        out['mat_view_counts'].append({'path':str(p),'sha256':digest(p),'arrays':sio.whosmat(p)})
    y=np.load(BASE/'prepared/y.npy');s=np.load(BASE/'prepared/subj.npy')
    for dtype in ['clutterremoved','raw']:
        x=np.load(ROOT/f'cache/X_{dtype}.npy',mmap_mode='r')
        for v in range(3):
            seen={};pairs=[]
            for i in range(len(x)):
                h=hashlib.blake2b(np.asarray(x[i,v]).tobytes(),digest_size=16).hexdigest()
                if h in seen:
                    j=seen[h];pairs.append({'i':i,'j':j,'subject_i':int(s[i]),'subject_j':int(s[j]),'class_i':int(y[i]),'class_j':int(y[j])})
                seen[h]=i
            out['per_view_duplicates'][f'{dtype}_{v}']=pairs
    good=quality_mask(y,s);out['excluded_indices']=np.where(~good)[0].tolist()
    out['retained_count']=int(good.sum());out['per_subject']={int(p):int((good&(s==p)).sum()) for p in np.unique(s)}
    out['policy']='Exclude both ambiguous HV2 G11/G12 classes and all HV6 G4 across views. No relabeling. Shared mask for every experiment.'
    dump(BASE/'results/quality_audit.json',out)
    print('QUALITY',out['retained_count'],{k:len(v) for k,v in out['per_view_duplicates'].items()},flush=True)

if __name__=='__main__':main()
