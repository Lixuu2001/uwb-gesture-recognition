"""Hash/schema-only audit crosscheck and real queue observation; no arrays/fit."""
from pathlib import Path
import datetime,hashlib,json,sys
B=Path('/workspace/uwb-hinception-sixfold-20261031')
OLD=Path('/workspace/uwb-hinception-sixfold-trained23-20261031/snapshot01')
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
def verify_audits(write=True):
 sys.path.insert(0,str(B));from readiness import ready
 cfg=read(B/'PROTOCOL.json');ready(cfg);receipt=read(OLD/'DRIVE_BACKUP_RECEIPT.json');proof=read(OLD/'READBACK_VERIFIED.json');manifest=read(OLD/'MANIFEST.json')
 assert sha(OLD/'DRIVE_BACKUP_RECEIPT.json')=='ec0b55b62db4ab30783c51ebb57c79fbb6699736cafac8e1f6f19e5ab803a44d'
 assert receipt['state']=='private_readback_verified' and receipt['scope']=='held2_held3_training_complete_full_audits_pending' and proof['state']=='remote_archive_and_all_members_verified'
 assert receipt['archive_sha256']==manifest['archive_sha256']==proof['archive_sha256']=='ad129c8fbf376cb34648ae354fa387959081b819bc3dee04c35db998c9525883'
 assert sha(OLD/manifest['archive_name'])==manifest['archive_sha256'] and sha(OLD/'payload/MEMBERS.json')==manifest['members_manifest_sha256'];assert len(receipt['chunks'])==14 and all(r['download_verified'] for r in receipt['chunks'])
 merged={};audits={};rows={}
 for held in [2,3]:
  fold=B/'run01'/('held'+str(held));a=read(fold/'COMPLETE_AUDIT.json');s=read(fold/'STATE.json');j=read(B/'jobs'/('held'+str(held))/'STATE.json');n=cfg['counts'][str(held)][0];steps=100*((n+63)//64)
  assert a['state']=='pass' and a['held']==held and a['epochs_replayed']==100 and a['training_updates_replayed_exact']==steps and a['training_appearances_verified']==100*n and a['diagnostic_labels_read'] is False and a['users7_8_labels_used'] is False and a['independent_validation'] is False and a['novel_contribution_validated'] is False
  for flag in ['all_initial_first_best_last_final_states_exact','native_selected_last_weights_optimizer_exact','all6900RMS_and_author_znorm_independent_float64_reference_verified','evaluation_probabilities_and_IDs_exact']:assert a[flag] is True
  assert s['state']=='complete' and s['pid'] is None and s['all_predictions_frozen'] and s['diagnostic_labels_read'] is False and j['state']=='complete_audited_unscored_fold' and j['pid'] is None
  for kind in ['train','audit']:
   t=read(B/'jobs'/('held'+str(held))/kind/'TERMINATION.json');assert t['exit_code']==0 and t['wall_timeout'] is False
   if kind=='train':assert t==a['training_termination']
  assert a['auditor_sha256']==sha(B/'audit_fold.py') and a['inspection_helper_sha256']==sha(B/'inspection_rng.py')
  for p,d in a['artifact_hashes'].items():
   if p in merged:assert merged[p]==d
   merged[p]=d
  assert a['evaluation_rows']==sum(cfg['counts'][str(held)][1:]);audits[held]=a;rows[str(held)]=dict(state='pass',epochs=100,updates=steps,training_appearances=100*n,selected_epoch=a['selected_epoch_verified'],evaluation_rows=a['evaluation_rows'],audit_sha256=sha(fold/'COMPLETE_AUDIT.json'),artifact_hashes=len(a['artifact_hashes']),diagnostic_labels_read=False)
 for p,d in merged.items():assert sha(p)==d,p
 # Tie every immutable archived producer file to the later independent replay.
 linked=[]
 for r in read(OLD/'payload/MEMBERS.json')['members']:
  parts=Path(r['path']).parts
  if len(parts)>=3 and parts[0]=='run01' and parts[1] in ['held2','held3']:
   held=int(parts[1][4:]);current=B/r['path'];assert sha(current)==r['sha256'] and sha(OLD/'payload'/r['path'])==r['sha256']
   if current.name!='FILES.json':assert audits[held]['artifact_hashes'][str(current)]==r['sha256']
   linked.append(dict(path=r['path'],sha256=r['sha256']))
 assert len(linked)==38 and not (B/'run01/ALL_NEW_PREDICTIONS_FROZEN.json').exists() and not (B/'run01/SUMMARY.json').exists()
 result=dict(state='passed_full_audits23_and_unchanged_private_producer_archive_crosslink',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audited_folds=[2,3],rows=rows,distinct_audit_artifact_hashes_reverified=len(merged),producer_files_linked_to_old_private_archive=linked,producer23_archive_sha256=manifest['archive_sha256'],producer23_receipt_sha256=sha(OLD/'DRIVE_BACKUP_RECEIPT.json'),all1465_frozen_prerequisites_reverified=True,diagnostic_scoring_performed=False,users7_8_labels_used=False,all_five_new_folds_complete=False,independent_validation=False,novel_contribution_validated=False)
 if write:(HERE/'AUDIT_CROSSCHECK.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
def process(pid,expected):
 p=Path('/proc')/str(pid);args=[x.decode() for x in (p/'cmdline').read_bytes().split(b'\0') if x];status={}
 for line in (p/'status').read_text().splitlines():
  if ':' in line:
   k,v=line.split(':',1)
   if k in ['Name','State','PPid','VmRSS','Threads']:status[k]=v.strip()
 assert status['State'].split()[0] not in ['Z','X'] and expected in args;return dict(pid=pid,command=args,status=status)
def observe():
 q=read(B/'STATE.json');assert q['state']=='running_unscored_source_confirmation' and sorted(q['completed'])==[2,3] and not q['failed'] and q['pending']==[6] and set(q['active'])=={'4','5'};procs=[process(q['pid'],str(B/'controller.py'))];folds={}
 for held in ['4','5']:
  job=read(B/'jobs'/('held'+held)/'STATE.json');worker=read(B/'run01'/('held'+held)/'STATE.json');assert job['state']=='training' and worker['state']=='training' and worker['total_steps']>=1 and worker['diagnostic_labels_read'] is False;procs.extend([process(job['pid'],str(B/'job.py')),process(worker['pid'],str(B/'run_fold.py'))]);assert int(procs[-1]['status']['PPid'])==job['pid'] and procs[-1]['command'][-2:]==['--held',held];folds[held]=dict(job=job,worker=worker)
 result=dict(state='verified_full_audits23_complete_original_training45_active',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue=q,folds=folds,processes=procs,completed_producer_folds=[2,3],fully_audited_new_folds=[2,3],no_new_diagnostic_labels_scored=True,users7_8_labels_used=False,source_sixfold_result_available=False,complete_private_modelset_verified=False)
 (B/'LIVE_OBSERVATION.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
 a=verify_audits();o=observe();print(json.dumps(dict(state=a['state'],distinct_hashes=a['distinct_audit_artifact_hashes_reverified'],private_producer_files_linked=len(a['producer_files_linked_to_old_private_archive']),active_training=list(o['folds']),audit_folds=a['audited_folds'],diagnostic_scoring=False)))
