from pathlib import Path
import sys,os,json,fcntl
from common import B,ready,write
from independent_supervisor import bounded_run
def main():
    c=ready()
    with (B/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (B/'run01').exists() and not (B/'supervisor01').exists()
        write(B/'STATE.json',dict(state='producer_active',pid=os.getpid(),diagnostic_scored=False))
        for script,directory,phase in [('run_pilot.py','supervisor01','producer'),('audit_complete.py','audit_supervisor01','full_numeric_audit')]:
            write(B/'STATE.json',dict(state=phase+'_active',pid=os.getpid(),diagnostic_scored=False))
            r=bounded_run([sys.executable,'-u',str(B/script)],B/directory,7200,28780,28800)
            if r['exit_code']!=0 or r['wall_timeout'] or r['resource_stop_reason']:
                write(B/'STATE.json',dict(state='execution_incomplete_'+phase,pid=None,termination=r,scientific_negative=False,diagnostic_scored=False));return
        write(B/'STATE.json',dict(state='complete_audited_unscored_requires_private_backup_then_once_finalize',pid=None,diagnostic_scored=False))
if __name__=='__main__':main()
