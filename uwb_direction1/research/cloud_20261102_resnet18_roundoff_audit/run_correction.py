from pathlib import Path
import sys,os,fcntl
from completion_common import O,ready,write
sys.path.insert(0,'/workspace/uwb-resnet18-imagenet-pilot-20261102')
from independent_supervisor import bounded_run
def main():
 ready()
 with (O/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (O/'supervisor01').exists() and not (O/'COMPLETE_AUDIT.json').exists();write(O/'STATE.json',dict(state='corrected_full_numeric_audit_active',pid=os.getpid(),diagnostic_scored=False,original_failed_audit_preserved=True))
  r=bounded_run([sys.executable,'-u',str(O/'corrected_full_audit.py')],O/'supervisor01',3600,14380,14400)
  if r['exit_code']!=0 or r['wall_timeout'] or r['resource_stop_reason'] or r['cgroup_oom_kill_delta']!=0:write(O/'STATE.json',dict(state='execution_incomplete_corrected_numeric_audit',pid=None,termination=r,diagnostic_scored=False));return
  write(O/'STATE.json',dict(state='complete_corrected_audit_unscored_requires_private_backup_then_once_finalize',pid=None,original_failed_audit_preserved=True,diagnostic_scored=False))
if __name__=='__main__':main()
