import os,sys,fcntl,time,resource
from common import B,ready,write,read,sha
from independent_supervisor import bounded_run
def main():
 c=ready()
 insurance=read(B/'INITIAL_PRIVATE_INSURANCE.json')
 assert insurance['state']=='owner_only_raw_allmember_initial_insurance_verified'
 assert insurance['frozen_sha256']==sha(B/'FROZEN_HASHES.json')
 assert insurance['diagnostic_labels_scored'] is False and insurance['shared'] is False
 from stream_archive import verify
 p=B/'initial_insurance';m=read(p/'MANIFEST.json');v=verify(m,read(p/'READBACK_INPUT.json'))
 assert m['scope']=='initial_seed_only_ShuffleNetV2_x1_SGDM40_B15_recovery_20261103'
 assert m['archive_sha256']==insurance['archive_sha256']==v['archive_sha256']
 start=time.monotonic();cpu0=resource.getrusage(resource.RUSAGE_CHILDREN)
 with (B/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (B/'run01').exists() and not (B/'supervisor01').exists()
  for script,directory,phase in [('run_pilot.py','supervisor01','producer'),('audit_complete.py','audit_supervisor01','full_numeric_audit')]:
   now=resource.getrusage(resource.RUSAGE_CHILDREN);leftwall=c['total_wall_seconds']-(time.monotonic()-start);leftcpu=c['total_CPU_seconds']-(now.ru_utime+now.ru_stime-cpu0.ru_utime-cpu0.ru_stime);assert leftwall>0 and leftcpu>20
   limits=c['producer_limits' if phase=='producer' else 'audit_limits'];wall=min(limits['wall'],leftwall);hard=int(min(limits['hard_cpu'],leftcpu));soft=min(limits['soft_cpu'],hard-20)
   write(B/'STATE.json',dict(state=phase+'_active',pid=os.getpid(),diagnostic_scored=False));r=bounded_run([sys.executable,'-u',str(B/script)],B/directory,wall,soft,hard)
   if r['exit_code']!=0 or r['wall_timeout'] or r['resource_stop_reason'] or r['cgroup_oom_kill_delta']:
    write(B/'STATE.json',dict(state='execution_incomplete_'+phase,pid=None,termination=r,scientific_negative=False,diagnostic_scored=False));return
  write(B/'STATE.json',dict(state='complete_audited_unscored_requires_private_streaming_backup_then_once_finalize',pid=None,diagnostic_scored=False))
if __name__=='__main__':main()
