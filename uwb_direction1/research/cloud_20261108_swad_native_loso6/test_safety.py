"""Prove the outer stop path also stops separately sessioned workers, using toy processes."""
from pathlib import Path
import sys,subprocess,tempfile,json
import phase_controller as outer
def main():
    with tempfile.TemporaryDirectory() as tmp:
        fold=Path(tmp);d=fold/'run/execution/train';d.mkdir(parents=True)
        children=[]
        try:
            for name in ['IDENTITY.json','FIRST_AUDIT_IDENTITY.json']:
                p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True)
                children.append(p);(d/name).write_text(json.dumps(outer.identity(p.pid)))
            controller=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True);children.append(controller)
            outer.stop_owned(fold,controller)
            for p in children:assert p.wait(timeout=5)!=0
        finally:
            for p in children:
                if p.poll() is None:p.terminate();p.wait()
    print(json.dumps(dict(state='pass',toy_controller_worker_first_auditor_separate_sessions_all_stopped=True,no_UWB_arrays_or_training=True)))
if __name__=='__main__':main()
