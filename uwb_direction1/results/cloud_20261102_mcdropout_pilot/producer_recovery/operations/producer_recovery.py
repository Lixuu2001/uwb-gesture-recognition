"""Read-only preservation of a completed MC producer while numerical audit runs.

This does not invoke or replace the frozen complete-audit backup guard, and
cannot authorize scoring. Frozen scientific files are copied without edits.
"""
from pathlib import Path
import argparse,json,sys,shutil,tarfile,time
B=Path('/workspace/uwb-mcdropout-pilot-20261102');sys.path.insert(0,str(B))
from guards import ready,read,sha,require_inference_complete
from backup import core
SCOPE='completed_MC100_inference_producer_numeric_audit_pending_not_certified'
def build(out):
 c=ready();term=require_inference_complete();cc=core();run=B/'run01';closed=read(run/'FILES.json');assert term['exit_code']==0 and not term['wall_timeout']
 assert not (B/'run01/SUMMARY.json').exists();assert read(run/'INFERENCE_COMPLETE.json')['samples_per_role']==100
 out.mkdir(exist_ok=False);payload=out/'payload';payload.mkdir();members=[];seen=set()
 def copy(src,name):
  src=Path(src);name=str(name);cc.relative_name(name);assert name not in seen and src.is_file() and not src.is_symlink();seen.add(name);h=sha(src);dst=payload/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);assert sha(dst)==sha(src)==h;members.append(dict(path=name,bytes=dst.stat().st_size,sha256=h))
 for n in read(B/'FROZEN_HASHES.json'):
  if not Path(n).is_absolute():copy(B/n,Path('study')/n)
 for n in ['FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json']:copy(B/n,Path('study')/n)
 for p in sorted(run.rglob('*')):
  if p.is_file() and p.name!='worker.lock' and p.suffix!='.tmp':copy(p,Path('run01')/p.relative_to(run))
 for p in sorted((B/'supervisor01').iterdir()):
  if p.is_file():copy(p,Path('supervisor01')/p.name)
 for i,p in enumerate(c['dependencies_text']):copy(p,Path('dependencies/text')/(str(i)+'_'+Path(p).name))
 copy(c['source_manifest'],'dependencies/source_ids.npz');copy(c['selected_checkpoint'],'dependencies/immutable_selected_TimeMIL41.pt')
 src=Path('/workspace/uwb-mcdropout-source-probe-20261102');legacy=Path('/workspace/uwb-timemil-20261029')
 for p,h in read(src/'SOURCE_DEPENDENCIES.json')['files'].items():assert sha(p)==h;copy(p,Path('dependencies/private_legacy_source')/Path(p).relative_to(legacy))
 from backup import CORE
 copy(CORE,'preservation/complete_modelset.py')
 # The auditor and controller may be live: their mutable files are NOT copied.
 meta=dict(state='immutable_completed_MC100_producer_snapshot',scope=SCOPE,created_unix=time.time(),members=members,complete_modelset=False,numeric_audit_certified=False,full_MC_producer_outputs_included=True,all_initial_first_final_native_and_selected_weights_included=True,all100_MC_logits_and_RNG_per_role_included=True,exact_IDs_and_deterministic_controls_included=True,successful_producer_termination_included=True,full_numerical_audit_included=False,diagnostic_labels_scored=False,raw_arrays_included=False,source_maps_included=False,runtime_binaries_included=False,whole_job_resume=False,limitations='Completed producer recovery only. Live auditor/controller state excluded. Byte preservation of all produced outputs is not numerical certification and cannot authorize diagnostic scoring. Complete audited recovery still required after successful independent audit.')
 (payload/'MEMBERS.json').write_text(json.dumps(meta,indent=2)+'\n');prefix='mc100_completed_producer_20261102';archive=out/(prefix+'.tar.gz')
 with tarfile.open(archive,'w:gz') as t:
  for p in sorted(payload.rglob('*')):
   if p.is_file():t.add(p,arcname=str(p.relative_to(payload)),recursive=False)
 chunks=[]
 with archive.open('rb') as f:
  i=0
  while data:=f.read(7*1024*1024):
   p=out/(prefix+f'.part{i:03d}');p.write_bytes(data);chunks.append(dict(name=p.name,bytes=len(data),sha256=sha(p)));i+=1
 m={k:v for k,v in meta.items() if k!='members'};m.update(archive_name=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),members_manifest_sha256=sha(payload/'MEMBERS.json'),tracked_payload_files=len(members),chunks=chunks);(out/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n');cc.verify_members(archive,m['members_manifest_sha256'],len(members))
 for n,h in closed.items():assert sha(run/n)==h,n
 print(json.dumps(dict(state='local_completed_producer_archive_verified',scope=SCOPE,bytes=m['archive_bytes'],chunks=len(chunks),members=len(members)+1,numeric_audit_certified=False)))
def verify(out):
 # Reuse the unchanged frozen transport/member verifier, not its complete guard.
 from backup import verify as verified_transport
 verified_transport(out);m=read(out/'MANIFEST.json');assert m['scope']==SCOPE and not m['complete_modelset'] and not m['numeric_audit_certified']
def receipt(out):
 m=read(out/'MANIFEST.json');v=read(out/'READBACK_VERIFIED.json');rows=read(out/'READBACK_INPUT.json');assert m['scope']==v['scope']==SCOPE and not m['complete_modelset'] and not v['complete_modelset_verified'] and m['archive_sha256']==v['archive_sha256'];assert len(rows)==len(m['chunks'])==len(set(r['file_id'] for r in rows));chunks=[]
 for c,r in zip(m['chunks'],rows):
  assert all(c[k]==r[k] for k in ['name','bytes','sha256']) and r['raw_fetch'] and not r['include_base64'] and r['download_verified'];md=r['metadata'];assert md['shared'] is False and md['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf'] and len(md['permissions'])==1 and md['permissions'][0]['role']=='owner' and md['size_bytes']==c['bytes'];chunks.append({k:x for k,x in r.items() if k!='path'})
 r=dict(state='private_readback_verified',scope=SCOPE,folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',archive_name=m['archive_name'],archive_bytes=m['archive_bytes'],archive_sha256=m['archive_sha256'],members_manifest_sha256=m['members_manifest_sha256'],chunks=chunks,chunks_verified=len(rows),archive_members=v['archive_members'],full_MC_producer_outputs_byte_preserved=True,all100_MC_logits_and_RNG_per_role_included=True,all_produced_states_and_selected_weights_included=True,complete_modelset_verified=False,numeric_audit_certified=False,diagnostic_labels_scored=False,raw_arrays_included=False,source_maps_included=False,runtime_binaries_included=False,whole_job_resume=False)
 (out/'DRIVE_BACKUP_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n');index=dict(schema_version=1,created_date_utc='2026-11-02',purpose='Completed MC100 producer recovery while independent numerical audit is pending; never use as complete-audit/scoring authorization.',**r,receipt_sha256=sha(out/'DRIVE_BACKUP_RECEIPT.json'),max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),folder_must_remain_owner_only=True,raw_maps_runtime_restore_separately=True,empty_search_is_not_duplicate_absence_proof=True);(out/'CLOUD_DOWNLOAD_INDEX.json').write_text(json.dumps(index,indent=2)+'\n');print(json.dumps(dict(scope=SCOPE,bytes=m['archive_bytes'],chunks=len(rows),members=v['archive_members'],archive_sha256=m['archive_sha256'],receipt_sha256=sha(out/'DRIVE_BACKUP_RECEIPT.json'),numeric_audit_certified=False)))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['build','verify','receipt']);p.add_argument('--out',type=Path,required=True);a=p.parse_args();globals()[a.action](a.out)
