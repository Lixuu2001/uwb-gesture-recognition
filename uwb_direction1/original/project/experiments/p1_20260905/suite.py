"""Serial GPU queue. Each arm writes per-fold resume markers only after success."""
import subprocess,sys,time,json,os
import numpy as np
from pipeline import BASE,RESULTS,dump

def run(tag,*extra,seed=2026):
    cmd=[sys.executable,'-u',str(BASE/'train.py'),'--tag',tag,'--seed',str(seed),*map(str,extra)]
    print('DISPATCH',tag,seed,flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS/f'{tag}_seed{seed}.log','a',encoding='utf-8') as log:
        r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:raise RuntimeError(f'{tag} failed, see log')
    aggregate();print('FINISHED',tag,seed,flush=True)

def read(tag,seed=2026):
    paths=[RESULTS/f'{tag}_s{seed}_h{i}.json' for i in range(1,9)]
    return [json.loads(p.read_text(encoding='utf-8')) for p in paths] if all(p.exists() for p in paths) else []

def gate(candidate,baseline,seed=2026):
    c=read(candidate,seed);b=read(baseline,seed)
    if not c or not b:return False
    d=np.array([x['accuracy']-y['accuracy'] for x,y in zip(c,b)])
    return bool(d.mean()>=.02 and (d>0).sum()>=6)

def aggregate():
    groups={}
    for p in RESULTS.glob('*_s*_h*.json'):
        r=json.loads(p.read_text(encoding='utf-8'))
        if 'accuracy' not in r:continue
        key=r['tag']+'_seed'+str(r['seed']);groups.setdefault(key,[]).append(r)
    out=[]
    for key,rows in sorted(groups.items()):
        out.append({'key':key,'folds':len(rows),'mean_accuracy':float(np.mean([r['accuracy'] for r in rows])),
                    'std_subject':float(np.std([r['accuracy'] for r in rows],ddof=1)) if len(rows)>1 else None,
                    'macro_f1':float(np.mean([r['macro_f1'] for r in rows])),
                    'parameters':rows[0]['parameters'],'held_subjects':[r['held_subject'] for r in rows]})
    dump(RESULTS/'neural_summary.json',out)

def main():
    if os.name=='nt':
        import ctypes;ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    run('sv_L_off','--mode','off','--views',0)
    aggregate()
    for v,name in enumerate(['L','T','R']):
        if v:run(f'sv_{name}_off','--mode','off','--views',v)
        for mode in ['only','concat']:
            run(f'sv_{name}_{mode}','--mode',mode,'--views',v)
    run('sv_L_signed','--mode','signed','--views',0)
    for readout in ['mean','concat']:
        run('teacher_'+readout,'--mode','off','--views',0,1,2,'--readout',readout,'--save-teacher')
    teacher='teacher_mean,teacher_concat'
    for v,name in enumerate(['L','T','R']):
        run(f'sv_{name}_kd','--mode','off','--views',v,'--teacher',teacher)
    # Fixed R&D rules from PLAN, no retuning on test accuracy.
    decisions={}
    for v,name in enumerate(['L','T','R']):
        base=f'sv_{name}_off'
        for mode in ['only','concat','kd']:
            tag=f'sv_{name}_{mode}';decisions[tag]=gate(tag,base)
            if decisions[tag]:
                run(base,'--mode','off','--views',v,seed=2027)
                if mode=='kd':
                    for ro in ['mean','concat']:
                        run('teacher_'+ro,'--mode','off','--views',0,1,2,'--readout',ro,'--save-teacher',seed=2027)
                    run(tag,'--mode','off','--views',v,'--teacher',teacher,seed=2027)
                else:run(tag,'--mode',mode,'--views',v,seed=2027)
                if mode=='concat':run(f'sv_{name}_duplicate','--mode','duplicate','--views',v)
        # Motion auxiliary candidate only if either input or KD pays on this view.
        if any(decisions[f'sv_{name}_{m}'] for m in ['only','concat','kd']):
            for aux in ['self','joint','quality']:
                run(f'sv_{name}_aux_{aux}','--mode','off','--views',v,'--aux',aux)
    dump(RESULTS/'gate_decisions.json',decisions)
    aggregate();dump(RESULTS/'SUITE_COMPLETE.json',{'complete':True,'time':time.time()})
    print('SUITE_COMPLETE',flush=True)

if __name__=='__main__':main()
