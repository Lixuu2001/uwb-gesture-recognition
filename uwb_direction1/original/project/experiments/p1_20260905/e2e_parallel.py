"""Two independent LOSO fold workers; unchanged training protocol."""
import json,sys,subprocess,time
from pathlib import Path
B=Path(__file__).resolve().parent
O=B/'e2e_reconstruction_results'

def done(h):
    for m in ['fixed_mask','capacity','joint']:
        p=O/f'{m}_L_s2026_h{h}'
        if not p.with_suffix('.json').exists() or not Path(str(p)+'_predictions.npz').exists():return False
        r=json.loads(p.with_suffix('.json').read_text())
        if r['fold']!=h or r['seed']!=2026:return False
    return True

def main():
    todo=[h for h in range(1,9) if not done(h)];running={}
    while todo or running:
        while todo and len(running)<2:
            h=todo.pop(0);f=(O/f'fold{h}.parallel.log').open('a',encoding='utf-8')
            p=subprocess.Popen([sys.executable,'-u',str(B/'e2e_reconstruction.py'),'--folds',str(h)],cwd=B,stdout=f,stderr=subprocess.STDOUT)
            running[h]=(p,f);print('START_FOLD',h,flush=True)
        for h,(p,f) in list(running.items()):
            rc=p.poll()
            if rc is None:continue
            f.close()
            if rc!=0 or not done(h):raise RuntimeError(f'fold {h} failed rc={rc}; see log')
            print('DONE_FOLD',h,flush=True);del running[h]
        time.sleep(1)
    print('E2E_ALL_COMPLETE',flush=True)

if __name__=='__main__':main()
