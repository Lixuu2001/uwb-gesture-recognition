"""Post-completion private archive builder/checker; no model, array or label load."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,importlib.util,json,shutil,tarfile,time
B=Path('/workspace/uwb-hinception-sixfold-20261031')
HELPER=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for data in iter(lambda:f.read(4*1024*1024),b''):h.update(data)
 return h.hexdigest()
def require(value,message):
 if not value:raise RuntimeError(message)
def checked(p,d):require(sha(p)==d,'Changed artifact: '+str(p))
def relative_name(n):
 p=PurePosixPath(n)
 require(n and not p.is_absolute() and '..' not in p.parts and str(p)==n,'Unsafe member name')
 return p
def verify_members(archive,manifest_sha,expected_payload):
 with tarfile.open(archive,'r:gz') as tar:
  entries=tar.getmembers();names=[p.name for p in entries]
  require(len(names)==len(set(names)) and all(p.isfile() for p in entries),'Duplicate/non-file member')
  for name in names:relative_name(name)
  require('MEMBERS.json' in names,'No member manifest')
  data=tar.extractfile('MEMBERS.json').read();require(hashlib.sha256(data).hexdigest()==manifest_sha,'Changed member manifest');metadata=json.loads(data);members=metadata['members']
  require(len(members)==expected_payload,'Wrong payload count');expected={r['path'] for r in members}
  require(len(expected)==len(members) and set(names)==expected|{'MEMBERS.json'},'Missing/unexpected member')
  for row in members:
   p=relative_name(row['path']);f=tar.extractfile(str(p));h=hashlib.sha256();length=0
   for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block);length+=len(block)
   require(length==row['bytes'] and h.hexdigest()==row['sha256'],'Corrupt payload member: '+str(p))
 return dict(payload_members=len(members),archive_members=len(entries),metadata={k:v for k,v in metadata.items() if k!='members'})
def check_complete(base,readiness_check=None):
 # Refuse active/partial work before any dependency or model hashes/copies.
 state=read(base/'STATE.json')
 require(state['state']=='source6_audited_complete_requires_full_private_modelset_backup_and_publication' and state['pid'] is None and not state['active'] and not state['pending'] and not state['failed'] and sorted(state['completed'])==[2,3,4,5,6],'Source confirmation still active/incomplete')
 cfg=read(base/'PROTOCOL.json')
 if readiness_check is None:
  spec=importlib.util.spec_from_file_location('source6_private_readiness',base/'readiness.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);readiness_check=mod.ready
 readiness_check(cfg)
 require(cfg['new_held_users']==[2,3,4,5,6] and cfg['epochs']==100 and cfg['batch_size']==64 and cfg['seed']==20261029,'Changed fixed confirmation')
 global_freeze=read(base/'run01/ALL_NEW_PREDICTIONS_FROZEN.json');require(set(global_freeze)=={'2','3','4','5','6'},'Incomplete global freeze')
 summary=read(base/'run01/SUMMARY.json');require(summary['state']=='complete_exploratory_source_sixfold_confirmation' and summary['all_five_predictions_frozen_before_new_diagnostic_scoring'] and summary['users7_8_labels_used'] is False and summary['independent_validation'] is False and summary['novel_contribution_validated'] is False,'Wrong score boundary/disclosure')
 for held in [2,3,4,5,6]:
  fold=base/'run01'/('held'+str(held));job=base/'jobs'/('held'+str(held));n=cfg['counts'][str(held)][0];updates=100*((n+63)//64);a=read(fold/'COMPLETE_AUDIT.json');s=read(fold/'STATE.json');j=read(job/'STATE.json')
  require(s['state']=='complete' and s['pid'] is None and s['diagnostic_labels_read'] is False and s['all_predictions_frozen'] and s['total_steps']==updates,'Incomplete producer')
  require(j['state']=='complete_audited_unscored_fold' and j['pid'] is None,'Incomplete fold job')
  require(a['state']=='pass' and a['held']==held and a['epochs_replayed']==100 and a['training_updates_replayed_exact']==updates and a['training_appearances_verified']==100*n and a['diagnostic_labels_read'] is False and a['users7_8_labels_used'] is False,'Incomplete/wrong full audit')
  for flag in ['all_initial_first_best_last_final_states_exact','native_selected_last_weights_optimizer_exact','all6900RMS_and_author_znorm_independent_float64_reference_verified','evaluation_probabilities_and_IDs_exact']:require(a[flag] is True,'Missing complete numeric verification')
  for p,d in a['artifact_hashes'].items():checked(p,d)
  for name in ['initial.pt','first_update.pt','best.pt','last.pt','final_prediction_state.pt','best_model.keras','last_model.keras','IDENTITY.json','split.json','EVAL_BATCH_IDS.json','iterations.jsonl','epochs.jsonl','FILES.json','TRAINING_COMPLETE.json','TRAINING_RESULT.json']:require((fold/name).is_file(),'Missing complete fold file: '+name)
  f=read(fold/'ALL_PREDICTIONS_FROZEN.json');require(set(f)=={'validation','diagnostic'} and f==global_freeze[str(held)],'Wrong dual/global prediction freeze')
  for role,d in f.items():checked(fold/(role+'_predictions_unscored.npz'),d)
  for kind in ['train','audit']:
   term=read(job/kind/'TERMINATION.json');require(term['exit_code']==0 and term['wall_timeout'] is False,'Unsuccessful termination')
 up=Path(cfg['upstream_workspace'])/'complete_backup_20261031';m=read(up/'MANIFEST.json');receipt=read(up/'DRIVE_BACKUP_RECEIPT.json');proof=read(up/'READBACK_VERIFIED.json')
 require(receipt['state']=='private_readback_verified' and receipt['standalone_complete_modelset_recoverable'] and receipt['all_completed_models_included'] and proof['state']=='remote_archive_and_all_members_verified','Reused held1 recovery incomplete')
 require(receipt['archive_sha256']==m['archive_sha256']==proof['archive_sha256'],'Changed reused held1 archive binding');archive=up/m['archive_name'];checked(archive,m['archive_sha256']);verify_members(archive,m['members_manifest_sha256'],m['tracked_payload_files'])
 old_members=read(up/'payload/MEMBERS.json')['members'];checked(up/'payload/MEMBERS.json',m['members_manifest_sha256'])
 for row in old_members:checked(up/'payload'/str(relative_name(row['path'])),row['sha256'])
 return cfg,up,old_members
def pack(out,scope,members,prefix):
 payload=out/'payload';snapshot=dict(state='immutable_complete_HInception_source_sixfold_modelset',scope=scope,members=members,created_unix=time.time(),new_folds=[2,3,4,5,6],reused_held1_full_payload_included=True,all_completed_models_included=True,full_replay_audits_complete=True,raw_arrays_included=False,source_maps_included=False,runtime_binaries_included=False,full_environment_or_job_resume_snapshot=False,private_remote_readback_verified=False,limitations='All six complete model trajectories/checkpoints, prediction/ID/log/audit evidence and source/dependency text. Runtime binaries and original raw/maps restore separately from pinned private indexes. Native last/selected optimizer states and full state objects retained; no automatic whole-job resume. This local archive does not establish remote preservation until all chunks and every member are read back.')
 (payload/'MEMBERS.json').write_text(json.dumps(snapshot,indent=2)+'\n');archive=out/(prefix+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tar:
  for p in sorted(payload.rglob('*')):
   if p.is_file():tar.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as f:
  i=0
  while data:=f.read(7*1024*1024):
   p=out/(prefix+f'.part{i:03d}');p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=sha(p)));i+=1
 manifest={k:v for k,v in snapshot.items() if k!='members'};manifest.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),members_manifest_sha256=sha(payload/'MEMBERS.json'),tracked_payload_files=len(members),chunks=chunks);(out/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');verify_members(archive,manifest['members_manifest_sha256'],len(members));return manifest
def build(base,out):
 cfg,up,old_members=check_complete(base)
 # No destination is created if completion/readiness/audit/member guards fail.
 out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();members=[];copied=set()
 def copy(source,name):
  n=str(name);relative_name(n);require(n not in copied,'Duplicate destination');copied.add(n);p=Path(source);require(p.is_file() and not p.is_symlink(),'Nonregular source');h=sha(p);target=payload/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);require(sha(p)==sha(target)==h,'Copy changed');members.append(dict(path=n,bytes=target.stat().st_size,sha256=h))
 for p in sorted(base.iterdir()):
  if p.is_file() and p.suffix in ['.py','.md','.json','.log']:copy(p,Path('study')/p.name)
 for directory in ['run01','jobs']:
  for p in sorted((base/directory).rglob('*')):
   if p.is_file() and p.name!='worker.lock' and p.suffix!='.tmp':copy(p,Path(directory)/p.relative_to(base/directory))
 for row in old_members:copy(up/'payload'/row['path'],Path('reused_held1/complete_payload')/row['path'])
 copy(up/'payload/MEMBERS.json',Path('reused_held1/complete_payload/MEMBERS.json'))
 for name in ['MANIFEST.json','DRIVE_BACKUP_RECEIPT.json','READBACK_VERIFIED.json']:copy(up/name,Path('reused_held1/recovery')/name)
 for i,p in enumerate(cfg['dependencies_text']):copy(p,Path('dependencies/text')/(str(i)+'_'+Path(p).name))
 copy(cfg['source_manifest'],Path('dependencies/source_ids.npz'))
 for prefix in ['localcnn','gap']:
  for kind in ['summary','audit']:copy(cfg[prefix+'_'+kind],Path('comparators')/(prefix+'_'+kind+'.json'))
  for held,row in cfg['comparator_predictions'][prefix].items():checked(row['path'],row['sha256']);copy(row['path'],Path('comparators')/prefix/('held'+held+'_predictions_unscored.npz'))
 copy('/workspace/uwb-multirocket-20261029/launch.py',Path('dependencies/independent_supervisor.py'))
 for p in sorted(HELPER.iterdir()):
  if p.is_file() and p.suffix in ['.py','.md','.json','.log']:copy(p,Path('preservation')/p.name)
 # Recheck all completion bindings after copying, before sealing archive.
 check_complete(base);manifest=pack(out,'complete_six_models_and_five_new_full_replay_audits',members,'hinception_source6_complete_20261031');print(json.dumps(dict(state='complete_local_archive_not_yet_private_readback',bytes=manifest['archive_bytes'],chunks=len(manifest['chunks']),payload_members=len(members))))
def verify_readback(out):
 manifest=read(out/'MANIFEST.json');require(manifest['state']=='immutable_complete_HInception_source_sixfold_modelset' and manifest['scope']=='complete_six_models_and_five_new_full_replay_audits' and manifest['new_folds']==[2,3,4,5,6] and manifest['all_completed_models_included'] and manifest['reused_held1_full_payload_included'] and manifest['full_replay_audits_complete'],'Partial manifest cannot establish complete recovery');rows=read(out/'READBACK_INPUT.json');require(len(rows)==len(manifest['chunks']),'Wrong chunk count');archive=out/'remote_reassembled.tar.gz'
 with archive.open('wb') as f:
  for c,r in zip(manifest['chunks'],rows):
   p=Path(r['path']);require(r['name']==c['name'] and p.stat().st_size==c['bytes'],'Wrong readback name/size');checked(p,c['sha256'])
   with p.open('rb') as part:shutil.copyfileobj(part,f)
 require(archive.stat().st_size==manifest['archive_bytes'],'Wrong archive size');checked(archive,manifest['archive_sha256']);counts=verify_members(archive,manifest['members_manifest_sha256'],manifest['tracked_payload_files'])
 for key in ['state','scope','new_folds','all_completed_models_included','reused_held1_full_payload_included','full_replay_audits_complete']:require(counts['metadata'][key]==manifest[key],'Changed archive/external scope binding')
 proof=dict(state='remote_archive_and_all_members_verified',scope=manifest['scope'],chunks_verified=len(rows),tracked_payload_members_verified=counts['payload_members'],archive_members=counts['archive_members'],archive_sha256=manifest['archive_sha256'],complete_modelset_verified=True);(out/'READBACK_VERIFIED.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof));return proof
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['check','build','verify']);p.add_argument('--out',type=Path);a=p.parse_args()
 if a.action=='check':check_complete(B);print('all_complete_guards_passed')
 else:
  require(a.out is not None,'An empty --out is required');build(B,a.out) if a.action=='build' else verify_readback(a.out)
