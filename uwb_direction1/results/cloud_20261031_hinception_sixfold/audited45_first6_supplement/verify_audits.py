"""Pure metadata/hash binding of full audits4/5 to preserved producer bytes."""
from pathlib import Path
import datetime,importlib.util,json,sys
TOOLS=Path('/workspace/uwb-hinception-producer-preservation-20261101')
sys.path.insert(0,str(TOOLS))
from producer_snapshot import B,read,sha,require,core
from observe_current import observe
HERE=Path(__file__).resolve().parent
OLD={4:Path('/workspace/uwb-hinception-producer4-20261101'),5:Path('/workspace/uwb-hinception-producer5-20261101')}
RECEIPTS={4:'dc4dd2c410ee82a94d392c779bdaedd9b2f946433e456e3a86bee17721153388',5:'023774868443d98d7b56318c1ea8f4d6c74365bc3e4e7545c817a6041ff65b44'}
def verify(write=True):
 require(sha(TOOLS/'producer_snapshot.py')=='044f6b6cd8348c8626cc84f7adab38c7dde4937c27ef4e909b361e1deccf9aa4','Changed producer helper');require(sha(TOOLS/'observe_current.py')=='d03bacc2ee0426d6550892180ea3e1100f59c871e01796867679615777050b5d','Changed observer');sys.path.insert(0,str(B));from readiness import ready
 cfg=read(B/'PROTOCOL.json');ready(cfg);merged={};audits={};rows={};references={};linked=[]
 for held in [4,5]:
  fold=B/'run01'/('held'+str(held));a=read(fold/'COMPLETE_AUDIT.json');s=read(fold/'STATE.json');j=read(B/'jobs'/('held'+str(held))/'STATE.json');n=cfg['counts'][str(held)][0];steps=100*((n+63)//64)
  require(a['state']=='pass' and a['held']==held and a['epochs_replayed']==100 and a['training_updates_replayed_exact']==steps and a['training_appearances_verified']==100*n and a['diagnostic_labels_read'] is False and a['users7_8_labels_used'] is False,'Wrong/incomplete audit')
  for flag in ['all_initial_first_best_last_final_states_exact','native_selected_last_weights_optimizer_exact','all6900RMS_and_author_znorm_independent_float64_reference_verified','evaluation_probabilities_and_IDs_exact']:require(a[flag] is True,'Missing full numeric check')
  require(a['independent_validation'] is False and a['novel_contribution_validated'] is False and s['state']=='complete' and s['pid'] is None and s['all_predictions_frozen'] and s['diagnostic_labels_read'] is False and j['state']=='complete_audited_unscored_fold' and j['pid'] is None,'Wrong completion/disclosure')
  for kind in ['train','audit']:
   t=read(B/'jobs'/('held'+str(held))/kind/'TERMINATION.json');require(t['exit_code']==0 and t['wall_timeout'] is False,'Unsuccessful termination')
   if kind=='train':require(t==a['training_termination'],'Changed embedded training TERM')
  require(a['auditor_sha256']==sha(B/'audit_fold.py') and a['inspection_helper_sha256']==sha(B/'inspection_rng.py'),'Changed auditor binding');require(a['evaluation_rows']==sum(cfg['counts'][str(held)][1:]),'Wrong prediction rows')
  for p,d in a['artifact_hashes'].items():
   if p in merged:require(merged[p]==d,'Conflicting shared artifact')
   merged[p]=d
  audits[held]=a;rows[str(held)]=dict(state='pass',epochs=100,updates=steps,training_appearances=100*n,selected_epoch=a['selected_epoch_verified'],evaluation_rows=a['evaluation_rows'],audit_sha256=sha(fold/'COMPLETE_AUDIT.json'),artifact_hashes=len(a['artifact_hashes']),diagnostic_labels_read=False)
  old=OLD[held];receipt=read(old/'DRIVE_BACKUP_RECEIPT.json');proof=read(old/'READBACK_VERIFIED.json');m=read(old/'MANIFEST.json');require(sha(old/'DRIVE_BACKUP_RECEIPT.json')==RECEIPTS[held] and receipt['state']=='private_readback_verified' and proof['state']=='remote_archive_and_all_members_verified','Missing verified producer recovery')
  require(receipt['held']==m['held']==held and receipt['archive_sha256']==m['archive_sha256']==proof['archive_sha256'] and sha(old/m['archive_name'])==m['archive_sha256'],'Changed preserved model archive');require(sha(old/'payload/MEMBERS.json')==m['members_manifest_sha256'] and len(receipt['chunks'])==7 and all(r['download_verified'] for r in receipt['chunks']),'Changed private proof')
  for r in read(old/'payload/MEMBERS.json')['members']:
   prefix='run01/held'+str(held)+'/'
   if r['path'].startswith(prefix):
    current=B/r['path'];require(sha(current)==r['sha256'] and sha(old/'payload'/r['path'])==r['sha256'],'Preserved producer changed')
    if current.name!='FILES.json':require(a['artifact_hashes'][str(current)]==r['sha256'],'Not bound to replay evidence')
    linked.append(dict(held=held,path=r['path'],sha256=r['sha256']))
  references[str(held)]=dict(archive_sha256=m['archive_sha256'],receipt_sha256=RECEIPTS[held],local_directory=str(old),chunks=7)
 for p,d in merged.items():require(sha(p)==d,'Changed audit artifact: '+p)
 require(len(linked)==38,'Missing preserved producer files');require(not (B/'run01/SUMMARY.json').exists() and not (B/'run01/ALL_NEW_PREDICTIONS_FROZEN.json').exists(),'Global scoring boundary changed')
 result=dict(state='full_audits45_verified_and_linked_to_unchanged_private_models',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_audited_folds=[4,5],rows=rows,distinct_artifact_hashes_reverified=len(merged),linked_producer_files=linked,model_recovery_references=references,all1465_frozen_prerequisites_reverified=True,diagnostic_scoring_performed=False,users7_8_labels_used=False,all_five_new_folds_complete=False,complete_source6_modelset=False,independent_validation=False,novel_contribution_validated=False)
 if write:(HERE/'AUDIT_CROSSCHECK.json').write_text(json.dumps(result,indent=2)+'\n')
 return result
if __name__=='__main__':
 a=verify();o=observe();print(json.dumps(dict(state=a['state'],hashes=a['distinct_artifact_hashes_reverified'],linked_files=len(a['linked_producer_files']),active_phases={h:r['phase'] for h,r in o['folds'].items()},full_audited_new_folds=o['fully_audited_new_folds'],diagnostic_scoring=False)))
