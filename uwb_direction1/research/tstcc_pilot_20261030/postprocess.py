"""Wait for existing run, audit once and build local private archive; no training."""
import fcntl,hashlib,json,subprocess,sys,time,traceback
from pathlib import Path
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def training_ready(base):
 f=base/'supervisor01/TERMINATION.json'
 if not f.exists():return False
 term=read(f)
 if term['exit_code']!=0 or term['wall_timeout']:raise RuntimeError('training_failed; no audit/retry')
 state=read(base/'run01/STATE.json')
 if state['state']!='complete' or state['pid'] is not None:raise RuntimeError('training termination/state mismatch; no retry')
 return True
def main():
 lock=open(B/'postprocess.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(state='waiting_existing_training',pid=__import__('os').getpid(),training_restarted=False,Drive_upload_or_Git_push_capability=False)
 def record(**kw):state.update(kw);state['observed_unix']=time.time();(B/'POSTPROCESS_STATE.json').write_text(json.dumps(state,indent=2))
 def verify():
  for n,h in read(B/'POSTPROCESS_HASHES.json').items():assert sha(B/n)==h,n
  for n,h in read(B/'FROZEN_HASHES.json').items():assert sha(B/n)==h,n
 try:
  verify();record();deadline=read(B/'LAUNCH_RECEIPT.json')['launched_unix']+43200
  while not training_ready(B):
   if time.time()>deadline:raise TimeoutError('existing training termination missing beyond43200s; no retry')
   time.sleep(5)
  verify();assert not (B/'audit_supervisor01').exists() and not (B/'COMPLETE_AUDIT.json').exists();record(state='completed_run_audit_running')
  with (B/'AUDIT_LAUNCH.log').open('xb') as log:
   result=subprocess.run([sys.executable,'-u',str(B/'launch.py'),'--audit'],cwd=B,stdout=log,stderr=subprocess.STDOUT,timeout=7300)
  if result.returncode!=0:raise RuntimeError('completion audit failed; preserve logs and repair manually, no retry')
  assert read(B/'COMPLETE_AUDIT.json')['state']=='pass';verify();record(state='local_archive_building')
  with (B/'BACKUP_BUILD.log').open('xb') as log:
   result=subprocess.run([sys.executable,'-u',str(B/'backup.py'),'complete','--out',str(B/'complete_backup_20261030')],cwd=B,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  if result.returncode!=0:raise RuntimeError('local backup build failed; no retry')
  manifest=read(B/'complete_backup_20261030/MANIFEST.json');record(state='audited_local_archive_ready_requires_Drive_upload_readback_and_Git_publication',pid=None,archive_sha256=manifest['archive_sha256'],archive_bytes=manifest['archive_bytes'],archive_chunks=len(manifest['chunks']),private_remote_complete=False)
 except BaseException as e:record(state='postprocess_incomplete',pid=None,error=repr(e),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:lock.close()
if __name__=='__main__':assert sys.argv[1:]==['--run'];main()
