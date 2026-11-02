"""Finite eight-fold label-only control with paired audit and auto summary."""
import os
import subprocess
import sys
import time
from pipeline import BASE,RESULTS,dump,digest
from followup_analysis import load,check_pair
from label_uniform import TAG

OUT=BASE/'teacher_necessity_results'


def complete(h):
    r=load(TAG,2026,h)
    if r is None:return False
    m,a=r;b,p=load('sv_R_project_uniform',2026,h);check_pair(a,p)
    assert m['teacher'] is None and m['initial_head_hash']==b['initial_head_hash']
    assert m['config']['projection_matrix_sha']==b['config']['projection_matrix_sha']
    assert m['config']['wrapper_sha']==digest(BASE/'label_uniform.py')
    assert (RESULTS/f'{TAG}_s2026_h{h}.pt').exists()
    return True


def main():
    OUT.mkdir(exist_ok=True)
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    finished={h for h in range(1,9) if complete(h)};todo=[h for h in range(1,9) if h not in finished];active={}
    try:
        while todo or active:
            while todo and len(active)<2:
                h=todo.pop(0);log=(OUT/f'fold{h}.log').open('a',encoding='utf-8')
                p=subprocess.Popen([sys.executable,'-u',str(BASE/'label_uniform.py'),'--folds',str(h)],cwd=BASE,stdout=log,stderr=subprocess.STDOUT)
                active[h]=p,log;print('START_LABEL_CONTROL',h,p.pid,flush=True)
            changed=False
            for h,(p,log) in list(active.items()):
                rc=p.poll()
                if rc is None:continue
                log.close();del active[h]
                if rc!=0 or not complete(h):raise RuntimeError(f'fold {h} failed rc={rc}')
                finished.add(h);changed=True;print('DONE_LABEL_CONTROL',h,flush=True)
            if changed:subprocess.run([sys.executable,str(BASE/'teacher_necessity.py')],cwd=BASE,check=True)
            dump(OUT/'queue_state.json',{'status':'complete' if len(finished)==8 else 'running','queue_pid':os.getpid(),
                 'updated':time.strftime('%Y-%m-%d %H:%M:%S %z'),'completed':sorted(finished),'expected_runs':8,
                 'active':[{'fold':h,'pid':p.pid} for h,(p,log) in active.items()]})
            if active:time.sleep(3)
        print('LABEL_CONTROL_ALL_COMPLETE',flush=True)
    except BaseException as exc:
        for p,log in active.values():
            if p.poll() is None:p.terminate()
            p.wait();log.close()
        dump(OUT/'queue_state.json',{'status':'failed','error':repr(exc),'completed':sorted(finished)})
        raise


if __name__=='__main__':main()
