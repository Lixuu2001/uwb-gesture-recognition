import re,json,time
import numpy as np
import pandas as pd
from pipeline import BASE,ROOT,dump,digest
from uwb_osr.data import _index_files

def audit():
    result={'log_metrics':{},'csv_checks':[],'cautions':[]}
    for name in ['f2_loso_c0_nofusion','f2_loso_c0_xmv_k1nb4','sv_loso_cr_c0','sv_loso_c0']:
        p=ROOT/'logs'/f'{name}.log';text=p.read_text(encoding='utf-8',errors='replace')
        result['log_metrics'][name]={'sha256':digest(p),'closed_acc':[float(v) for v in re.findall(r'closed_acc=([\d.]+)',text)],
            'subj_oscr':[(int(a),float(b)) for a,b in re.findall(r'LOSO subj (\d+): OSCR=([\d.]+)',text)],
            'header':next((l for l in text.splitlines() if l.startswith('== MV-OSR')),None)}
    for dtype in ['ClutterRemoved','Raw']:
        idx=_index_files(str(ROOT/'12652592'),dtype)
        xx=np.load(ROOT/f'cache/X_{dtype.lower()}.npy',mmap_mode='r')
        yy=np.load(ROOT/f'cache/y_{dtype.lower()}.npy');ss=np.load(ROOT/f'cache/subj_{dtype.lower()}.npy')
        for subject in range(1,9):
            for gesture in range(1,13):
                targets=np.where((ss==subject)&(yy==gesture-1))[0]
                for v,view in enumerate(['Left','Top','Right']):
                    f=idx[(subject,gesture,view)]
                    values=pd.read_csv(f,header=None).to_numpy(dtype='float32')
                    if values.shape[0]%90:raise ValueError(f'Noninteger gesture count: {f}')
                    values=values.reshape(-1,90,189)
                    n=min(len(values),len(targets))
                    equal=np.array_equal(values[:n],xx[targets[:n],v],equal_nan=True)
                    record={'dtype':dtype,'subject':subject,'gesture':gesture,'view':view,
                            'source':f,'sha256':digest(f),'source_n':len(values),'cache_n':len(targets),'equal_prefix':equal}
                    result['csv_checks'].append(record)
                    if not equal: print('MISMATCH',record,flush=True)
            print('AUDIT',dtype,subject,flush=True)
        dump(BASE/'results/source_audit.json',result)
    result['cautions']=[
        'Raw log headers do not preserve every config field and only aggregate closed_acc; current source cannot certify historical code state.',
        'ClosedSetHead is MLP; inspector evaluation closed_acc uses logits argmax, not inspector distance ensemble.',
        '49.2 physical scalar pooling accuracies are not numeric centroid values. 0.00% classification is not proof of an identically zero physical signal.',
        '49.2 explicitly denies feature-level constant blindness; mean feature 2-way 82.25%.',
        '86.3 logistic regression is supervised, not zero-training. Missing 39-D feature code prevents exact reproduction.',
        'abs(raw) does not in general erase all motion direction; moving envelope retains range-time displacement.',
        'Hilbert phase increment is wrapped; phase coherence, aliasing and scattering require qualification. Static cancellation is not domain-shift immunity.',
        '2->1 reconstruction null result justifies low priority, not a mathematical impossibility proof for 1->2.',
        'Historical OSCR ablations do not prove all fusion architectures fail in 12-class closed-set LOSO.'
    ]
    dump(BASE/'results/source_audit.json',result)
    print('SOURCE_AUDIT_COMPLETE',flush=True)

if __name__=='__main__':audit()
