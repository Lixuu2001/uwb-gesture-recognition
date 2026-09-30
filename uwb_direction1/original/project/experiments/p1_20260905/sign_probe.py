"""Direction-only probe: orthogonal swipes calibrate polarity, diagonals test it."""
import numpy as np
from pipeline import BASE,RESULTS,dump,quality_mask

def main():
    c=np.load(BASE/'prepared/motion.npy');q=np.load(BASE/'prepared/quality.npy')
    y=np.load(BASE/'prepared/y.npy');s=np.load(BASE/'prepared/subj.npy');good=quality_mask(y,s)
    rows=[]
    for agg in ['mean','quality']:
        f=c.mean(-1) if agg=='mean' else (c*q).sum(-1)/(q.sum(-1)+1e-12)
        sources={'Left':f[:,0],'Top':f[:,1],'Right':f[:,2],'Right-minus-Left':f[:,2]-f[:,0]}
        for name,x in sources.items():
            for axis,positive,negative,testsign in [('horizontal',0,1,{4:1,5:1,6:-1,7:-1}),('vertical',2,3,{4:1,5:-1,6:1,7:-1})]:
                for held in range(1,9):
                    tr=good&(s!=held)
                    a=x[tr&(y==positive)].mean();b=x[tr&(y==negative)].mean()
                    polarity=1 if a>=b else -1
                    te=np.where(good&(s==held)&np.isin(y,[4,5,6,7]))[0]
                    truth=np.array([testsign[int(g)] for g in y[te]])
                    for threshold_kind,threshold in [('zero',0.),('train_midpoint',(a+b)/2)]:
                        pred=np.where((x[te]-threshold)*polarity>=0,1,-1)
                        rows.append({'aggregation':agg,'source':name,'axis':axis,'held_subject':held,
                                     'threshold_kind':threshold_kind,'train_polarity':polarity,'threshold':float(threshold),
                                     'accuracy':float((pred==truth).mean())})
    dump(RESULTS/'direction_probe.json',{'task':'Two-way direction on four unseen diagonal classes; supervised training-subject orthogonal polarity calibration; NOT 12-class recognition.','rows':rows})
    print('DIRECTION_PROBE_COMPLETE')

if __name__=='__main__':main()
