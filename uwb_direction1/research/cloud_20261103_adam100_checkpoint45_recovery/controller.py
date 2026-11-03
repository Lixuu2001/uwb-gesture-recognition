import os,sys,fcntl,time,resource,subprocess,datetime
from trial_common import B,ready,write,read,sha
from independent_supervisor import bounded_run
from backup import privacy
def main():
 c=ready()
 old=__import__('pathlib').Path(c['operational_recovery']['original_root'])
 insurance=read(old/'INITIAL_PRIVATE_INSURANCE.json')
 assert insurance['state']=='owner_only_raw_allmember_initial_insurance_verified'
 assert insurance['frozen_sha256']==c['operational_recovery']['original_frozen_sha256']
 assert insurance['diagnostic_labels_scored'] is False and insurance['shared'] is False
 from stream_archive import verify
 p=old/'initial_insurance';m=read(p/'MANIFEST.json');rows=read(p/'READBACK_INPUT.json');privacy(rows);v=verify(m,rows)
 assert m['complete_modelset'] is False
 assert m['scope']=='initial_seed_only_ShuffleNetV2_x1_Adam100_B20_20261103'
 assert m['archive_sha256']==insurance['archive_sha256']==v['archive_sha256']
 input_receipt=read(B/'CHECKPOINT45_PRIVATE_INSURANCE.json')
 assert input_receipt['state']=='checkpoint45_input_raw_allmember_verified'
 assert input_receipt['frozen_sha256']==sha(B/'FROZEN_HASHES.json')
 inp=B/'recovery_input_insurance';im=read(inp/'MANIFEST.json');ir=read(inp/'READBACK_INPUT.json');privacy(ir);iv=verify(im,ir)
 assert im['complete_modelset'] is False and iv['archive_sha256']==input_receipt['archive_sha256']
 members={x['path']:x for x in read(inp/'MEMBERS.json')['members']}
 assert members['original/run01/final_full.pt']['sha256']==read(B/'RECOVERY_INPUTS.json')['checkpoint_sha256']
 original_launch=read(old/'CONTROLLER_LAUNCH.json')['utc']
 prior_elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(original_launch)).total_seconds()
 assert prior_elapsed>=0
 prior_cpu_charge=5*prior_elapsed
 start=time.monotonic()-prior_elapsed;cpu0=resource.getrusage(resource.RUSAGE_CHILDREN)
 with (B/'controller.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (B/'run01').exists() and not (B/'supervisor01').exists()
  for script,directory,phase in [('produce.py','supervisor01','producer'),('audit_full.py','audit_supervisor01','full_numeric_audit')]:
   now=resource.getrusage(resource.RUSAGE_CHILDREN);leftwall=c['total_wall_seconds']-(time.monotonic()-start);leftcpu=c['total_CPU_seconds']-prior_cpu_charge-(now.ru_utime+now.ru_stime-cpu0.ru_utime-cpu0.ru_stime);assert leftwall>0 and leftcpu>20
   limits=c['producer_limits' if phase=='producer' else 'audit_limits'];wall=min(limits['wall'],leftwall);hard=int(min(limits['hard_cpu'],leftcpu));soft=min(limits['soft_cpu'],hard-20)
   write(B/'STATE.json',dict(state=phase+'_active',pid=os.getpid(),diagnostic_scored=False))
   try:r=bounded_run([sys.executable,'-u',str(B/script)],B/directory,wall,soft,hard,c['RSS_stop_KiB'])
   except (OSError,subprocess.SubprocessError):
    r=read(B/directory/'TERMINATION.json')
    write(B/'STATE.json',dict(state='execution_incomplete_'+phase,pid=None,termination=r,scientific_negative=False,diagnostic_scored=False));return
   if r['exit_code']!=0 or r['wall_timeout'] or r['resource_stop_reason'] or r['cgroup_oom_kill_delta']:
    write(B/'STATE.json',dict(state='execution_incomplete_'+phase,pid=None,termination=r,scientific_negative=False,diagnostic_scored=False));return
  write(B/'STATE.json',dict(state='complete_audited_unscored_requires_private_streaming_backup_then_once_finalize',pid=None,diagnostic_scored=False))
if __name__=='__main__':main()
