"""One manual bounded audit repair, then complete archive; never train/retry."""
import fcntl,hashlib,importlib.util,json,os,subprocess,sys,time,traceback
from pathlib import Path
B=Path(__file__).resolve().parent
STUDY=Path('/workspace/uwb-hinception-pilot-20261031')
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def preflight(base,repair):
 base=Path(base);repair=Path(repair)
 if (base/'COMPLETE_AUDIT.json').exists() or (base/'audit_attempt01').exists() or (base/'complete_backup_20261031').exists():raise RuntimeError('Existing completion/archived attempt: inspect, never repeat automatically')
 original=read(repair/'ORIGINAL_FAILURE.json')
 for name,digest in original['files_sha256'].items():
  if sha(base/name)!=digest:raise RuntimeError('Changed original failure/science evidence')
 term=read(base/'audit_supervisor01/TERMINATION.json')
 if term['exit_code']!=1 or term['wall_timeout'] or Path('/proc',str(term['pid'])).exists():raise RuntimeError('Original failed auditor not safely finished')
 for name,digest in read(base/'FROZEN_HASHES.json').items():
  if sha(base/name)!=digest:raise RuntimeError('Changed frozen science/dependency')
 for name,digest in read(base/'run01/FILES.json').items():
  if sha(base/'run01'/name)!=digest:raise RuntimeError('Changed completed training artifact')
 if not read(repair/'RNG_DIAGNOSIS.json')['restored_inspection_rng_then_selected_weights_equals_saved_final_without_real_forward']:raise RuntimeError('Checkpoint diagnosis not exact')
 if read(repair/'REGRESSION_CHECK.json')['state']!='pass_synthetic_native_reload_observational_regression':raise RuntimeError('Synthetic regression not passing')
 manifest=read(repair/'REPAIR_HASHES.json')
 for name,digest in manifest.items():
  if sha(repair/name)!=digest:raise RuntimeError('Changed repair code/evidence')
 marker=read(base/'AUDIT_REPAIR_REGISTRATION.json')
 if marker.get('state')!='remote_manual_audit_repair_exact_readback_verified' or marker.get('protocol_sha256')!=sha(base/'PROTOCOL.json') or marker.get('repair_manifest_sha256')!=sha(repair/'REPAIR_HASHES.json'):raise RuntimeError('Missing exact remote repair registration')
 return True
def main():
 lock=open(B/'repair.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(state='manual_repair_preflight',pid=os.getpid(),training_restarted=False,automatic_retry=False,Drive_or_Git_capability=False)
 def record(**kw):state.update(kw);state['observed_unix']=time.time();(B/'STATE.json').write_text(json.dumps(state,indent=2)+'\n')
 try:
  preflight(STUDY,B);record()
  # Original immutable failure is retained under its attempt number. The
  # canonical supervisor directory is then the successful/failing new attempt.
  (STUDY/'audit_supervisor01').rename(STUDY/'audit_attempt01')
  assert sha(STUDY/'audit_attempt01/process.log')==read(B/'ORIGINAL_FAILURE.json')['files_sha256']['audit_supervisor01/process.log']
  helper=Path('/workspace/uwb-multirocket-20261029/launch.py');assert sha(helper)=='b8f8038cc8242d20158b1e4870283fc61f884a039931fa08cf9940ea6b3d2eab'
  spec=importlib.util.spec_from_file_location('manual_audit_repair_supervisor',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  record(state='repaired_full100epoch_audit_running')
  result=module.bounded_run([sys.executable,'-u',str(B/'audit_complete_repaired.py'),'--run'],STUDY/'audit_supervisor01',14400,14390,14400)
  if result['exit_code']!=0 or result['wall_timeout']:raise RuntimeError('Manual repaired audit failed; preserve, no automatic retry')
  audit=read(STUDY/'COMPLETE_AUDIT.json');assert audit['state']=='pass' and audit['training_updates_replayed_exact']==7400 and audit['epochs_replayed']==100 and audit['auditor_sha256']==sha(B/'audit_complete_repaired.py')
  for name,digest in audit['artifact_hashes'].items():assert sha(name)==digest,name
  record(state='repaired_audit_pass_local_complete_archive_building')
  with (B/'BACKUP_BUILD.log').open('xb') as log:
   result=subprocess.run([sys.executable,'-u',str(B/'build_complete_repaired.py'),'complete','--out',str(STUDY/'complete_backup_20261031')],stdout=log,stderr=subprocess.STDOUT,timeout=600)
  if result.returncode!=0:raise RuntimeError('Repaired complete archive build failed; preserve, no retry')
  manifest=read(STUDY/'complete_backup_20261031/MANIFEST.json');record(state='repaired_audit_complete_local_archive_ready_requires_private_readback_and_publication',pid=None,archive_sha256=manifest['archive_sha256'],archive_bytes=manifest['archive_bytes'],archive_chunks=len(manifest['chunks']),private_remote_complete=False)
 except BaseException as exc:record(state='manual_audit_repair_incomplete',pid=None,error=repr(exc),traceback=traceback.format_exc(),scientific_negative=False);raise
 finally:lock.close()
if __name__=='__main__':assert sys.argv[1:]==['--run'];main()
