import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from threadpoolctl import threadpool_limits
from pipeline import BASE,RESULTS,split,dump,digest

def main():
    y=np.load(BASE/'prepared/y.npy');s=np.load(BASE/'prepared/subj.npy')
    motion=np.load(BASE/'prepared/stats.npy');amp=np.load(BASE/'prepared/amp_stats.npy')
    out=[]
    groups={'all12':np.arange(12),'seed8':np.array([0,1,2,3,8,9,10,11]),'swipe8':np.arange(8)}
    for group,classes in groups.items():
        subset=np.where(np.isin(y,classes))[0];yy=y[subset];ss=s[subset]
        for views in [[0],[1],[2],[0,1,2]]:
            for feature in ['motion','amplitude','combined']:
                parts=[]
                if feature!='amplitude':parts.append(motion[subset][:,views].reshape(len(subset),-1))
                if feature!='motion':parts.append(amp[subset][:,views].reshape(len(subset),-1))
                x=np.concatenate(parts,1)
                for held in range(1,9):
                    tr,va,te=split(yy,ss,held,2026)
                    best=None
                    for c in [.01,.1,1.]:
                        model=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=400,tol=1e-4))
                        model.fit(x[tr],yy[tr]);v=accuracy_score(yy[va],model.predict(x[va]))
                        if best is None or v>best[0]:best=(v,c,model)
                    pred=best[2].predict(x[te])
                    row={'group':group,'views':views,'feature':feature,'dim':x.shape[-1],'held_subject':held,
                         'C':best[1],'val_acc':best[0],'accuracy':accuracy_score(yy[te],pred),
                         'macro_f1':f1_score(yy[te],pred,average='macro'),'test_ids':subset[te].tolist(),'pred':pred.tolist()}
                    out.append(row)
                print('PHYSICAL',group,views,feature,'acc',np.mean([r['accuracy'] for r in out[-8:]]),flush=True)
                dump(RESULTS/'physical.json',{'feature_definition':'13 per view: mean,std,q10,q50,q90,positive_fraction,6 temporal means,abs_mean. Not recovered historical code.','source_hash':digest(__file__),'results':out})
    print('PHYSICAL_COMPLETE',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=2):main()
