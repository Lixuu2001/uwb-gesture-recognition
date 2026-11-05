"""Monitor the surviving ORIGINAL producer; never restart it or alter frozen science."""
from pathlib import Path
import sys,json,os,time,subprocess,resource,signal,fcntl,shutil
B=Path('/workspace/uwb-mixstyle-gap64-20261105');M=Path(__file__).resolve().parent
sys.path.insert(0,str(B))
from common import read,write,sha,ready
HZ=os.sysconf('SC_CLK_TCK');BOOT=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
def sample(pid):
 try:
  s=Path(f'/proc/{pid}/stat').read_text();a=s[s.rfind(')')+2:].split()
  try:r=Path(f'/proc/{pid}/status').read_text()
  except FileNotFoundError:r=''
  rss=next((int(l.split()[1]) for l in r.splitlines() if l.startswith('VmRSS:')),0)
  return dict(pid=pid,startticks=int(a[19]),boot_id=BOOT,state=a[0],cpu=(int(a[11])+int(a[12]))/HZ,RSS_KiB=rss,wait_status=int(a[49]))
 except FileNotFoundError:return None
def same(q,old):return q and all(q[k]==old[k] for k in ['pid','startticks','boot_id'])
def main():
 c=ready(True);lock=(M/'recovery_controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (M/'RECOVERY_CONTROLLER_IDENTITY.json').exists()
 old=read(B/'CONTROLLER_IDENTITY.json');dead=sample(old['pid']);assert same(dead,old) and dead['state']=='Z' and os.waitstatus_to_exitcode(dead['wait_status'])==1
 producer=read(B/'run/execution/train/IDENTITY.json');assert same(sample(producer['pid']),producer) and sample(producer['pid'])['state']!='Z'
 assert read(B/'STATE.json')['error']=="IndexError('list index out of range')"
 assert (B/'run/train/FIRST_RELEASE.json').exists() and not (B/'run/replay').exists()
 for n in ['STATE.json','RESOURCE_LEDGER.json','controller.log','run/train/FIRST_RELEASE.json']:
  target=M/'original_monitor_failure'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(B/n,target)
 unknown_first=read(B/'run/execution/train/FIRST_AUDIT_IDENTITY.json');assert sample(unknown_first['pid']) is None
 # The old first-audit termination was not persisted. Charge its whole600second bound,
 # and perform one independent first-update verification without replacing the producer.
 base_cpu=dead['cpu']+600.;wall_origin=old['startticks']/HZ;old_ledger=read(B/'RESOURCE_LEDGER.json')
 write(M/'RECOVERY_CONTROLLER_IDENTITY.json',sample(os.getpid()))
 repair=dict(state='reattached_to_original_live_producer_no_model_restart',original_controller=dead,producer=producer,original_first_audit=unknown_first,original_first_audit_exit_not_persisted=True,unknown_first_audit_conservative_CPU_charge_seconds=600.,original_wall_origin_monotonic=wall_origin,original_ledger=old_ledger,frozen_sha256=sha(B/'FROZEN_HASHES.json'),scientific18_unchanged=True,source_configuration_or_seed_changed=False,producer_restarted=False,budget_reset=False,recovery_source_sha256=sha(__file__))
 write(M/'MONITOR_RECOVERY.json',repair);write(B/'MONITOR_RECOVERY.json',repair)
 original_final_cpu=None
 def cost(live):
  ru=resource.getrusage(resource.RUSAGE_CHILDREN);own=resource.getrusage(resource.RUSAGE_SELF);cpu=base_cpu+ru.ru_utime+ru.ru_stime+own.ru_utime+own.ru_stime;rss=0
  q=sample(producer['pid'])
  if same(q,producer):cpu+=q['cpu'];rss+=q['RSS_KiB'] if q['state']!='Z' else 0
  elif original_final_cpu is not None:cpu+=original_final_cpu
  else:raise AssertionError('Original producer identity vanished before terminal accounting')
  for p in live:
   z=sample(p.pid)
   if z and p.returncode is None:cpu+=z['cpu'];rss+=z['RSS_KiB'] if z['state']!='Z' else 0
  return dict(wall=time.monotonic()-wall_origin,cpu=max(cpu,old_ledger['cpu']),RSS_KiB=rss)
 def guard(stage,live):
  z=cost(live);write(B/'RESOURCE_LEDGER.json',dict(**z,stage=stage,recovery_monitor=True,unknown_original_first_audit_CPU_charge_seconds=600.,budget_reset=False))
  if z['wall']>c['real_wall_cap'] or z['cpu']>c['real_CPU_cap'] or z['RSS_KiB']>c['real_RSS_KiB']:
   q=sample(producer['pid'])
   if same(q,producer) and q['state']!='Z':os.killpg(producer['pid'],signal.SIGTERM)
   for p in live:
    if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
   raise AssertionError('Original cumulative fixed resource cap')
  return z
 def execute(stage,first=False):
  d=B/'run/execution'/('recovery_first' if first else stage);d.mkdir(exist_ok=False)
  p=subprocess.Popen([sys.executable,str(B/'worker.py'),stage],stdin=subprocess.DEVNULL,stdout=(d/'stdout.log').open('x'),stderr=subprocess.STDOUT,start_new_session=True)
  write(d/'IDENTITY.json',sample(p.pid))
  while p.poll() is None:guard('train' if first else stage,[p]);time.sleep(.5)
  p.wait();z=guard('train' if first else stage,[]);term=dict(exit_code=p.returncode,resource_stop_reason=None,cumulative=z,diagnostic_labels_read=False,users7_8_labels_used=False,budget_reset=False)
  write(d/'TERMINATION.json',term);assert p.returncode==0,term
  return term
 try:
  write(B/'STATE.json',dict(state='original_MixStyle_producer_running_under_recovered_monitor',diagnostic_labels_read=False,producer_restarted=False,budget_reset=False))
  first=execute('first',True)
  write(B/'run/execution/train/FIRST_AUDIT_TERMINATION.json',dict(exit_code=0,native_first_release_exists=True,original_exit_not_persisted=True,independent_reverification_actual_exit_code=first['exit_code'],independent_reverification_identity=read(B/'run/execution/recovery_first/IDENTITY.json'),unknown_original_first_audit_CPU_charge_seconds=600.))
  while True:
   q=sample(producer['pid']);assert same(q,producer),'Original producer identity changed, no restart'
   z=guard('train',[])
   if q['state']=='Z':
    code=os.waitstatus_to_exitcode(q['wait_status']);assert code==0,dict(actual_wait_status=q['wait_status'],exit_code=code)
    original_final_cpu=q['cpu'];assert read(B/'run/train/COMPLETE.json')['epochs']==64
    write(B/'run/execution/train/TERMINATION.json',dict(exit_code=code,resource_stop_reason=None,cumulative=z,diagnostic_labels_read=False,users7_8_labels_used=False,exit_observed_via_same_identity_proc_stat_wait_status=True,actual_wait_status=q['wait_status'],original_producer_CPU_seconds=q['cpu'],producer_restarted=False,budget_reset=False));break
   time.sleep(1)
  for stage in ['replay','output']:execute(stage)
  z=guard('output',[]);write(B/'STATE.json',dict(state='pilot_audited_frozen_unscored_requires_complete_private_backup_before_once_gate',completed=['train','replay','output'],diagnostic_labels_read=False,users7_8_labels_used=False,wall=z['wall'],cpu=z['cpu'],monitor_recovered=True,producer_restarted=False,budget_reset=False))
  write(M/'RECOVERY_CONTROLLER_TERMINATION.json',dict(exit_code=0,state='original_producer_and_one_fresh_replay_complete_unscored',resource=z))
 except BaseException as exc:
  write(M/'RECOVERY_CONTROLLER_FAILURE.json',dict(error=repr(exc),wall=time.monotonic()-wall_origin,scientific_negative=False,no_automatic_restart=True));raise
 finally:lock.close()
if __name__=='__main__':main()
