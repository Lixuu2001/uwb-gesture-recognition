"""Operational receipt only after frozen raw-chunk/member verifier has passed."""
from pathlib import Path
import argparse,json,hashlib
def main(b):
 read=lambda p:json.loads(p.read_bytes());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();m=read(b/'MANIFEST.json');v=read(b/'READBACK_VERIFIED.json');rows=read(b/'READBACK_INPUT.json')
 assert v['state']=='remote_archive_and_all_members_verified' and v['archive_sha256']==m['archive_sha256'] and v['scope']==m['scope'] and v['complete_modelset_verified']==m['complete_modelset'];assert len(rows)==len(m['chunks'])==len(set(r['file_id'] for r in rows))
 chunks=[]
 for c,r in zip(m['chunks'],rows):
  assert all(c[k]==r[k] for k in ['name','bytes','sha256']) and r['download_verified'] and r['raw_fetch'] and not r['include_base64'];meta=r['metadata'];assert meta['shared'] is False and meta['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf'] and len(meta['permissions'])==1 and meta['permissions'][0]['role']=='owner' and meta['size_bytes']==c['bytes'];chunks.append({k:z for k,z in r.items() if k!='path'})
 r=dict(state='private_readback_verified',scope=m['scope'],folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',archive_name=m['archive_name'],archive_bytes=m['archive_bytes'],archive_sha256=m['archive_sha256'],members_manifest_sha256=m['members_manifest_sha256'],chunks=chunks,chunks_verified=len(rows),archive_members=v['archive_members'],complete_modelset_verified=m['complete_modelset'],numeric_audit_certified=m['complete_modelset'],all_produced_initial_first_best_last_final_and_native_models_included=m['complete_modelset'],diagnostic_labels_scored=False,raw_arrays_included=False,source_maps_included=False,runtime_binaries_included=False,whole_job_resume=False)
 (b/'DRIVE_BACKUP_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n')
 index=dict(schema_version=1,created_date_utc='2026-11-02',purpose='Audited complete DLinear default max10/patience3 modelset recovery.' if m['complete_modelset'] else 'Immutable DLinear initial/first recovery only; not latest/complete.',**r,receipt_sha256=sha(b/'DRIVE_BACKUP_RECEIPT.json'),max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),folder_must_remain_owner_only=True,latest_epoch_privately_preserved=m['complete_modelset'],raw_maps_runtime_restore_separately=True,empty_search_is_not_duplicate_absence_proof=True)
 (b/'CLOUD_DOWNLOAD_INDEX.json').write_text(json.dumps(index,indent=2)+'\n');print(json.dumps(dict(scope=m['scope'],bytes=m['archive_bytes'],chunks=len(rows),members=v['archive_members'],archive_sha256=m['archive_sha256'],receipt_sha256=sha(b/'DRIVE_BACKUP_RECEIPT.json'),full_modelset=m['complete_modelset'])))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);main(p.parse_args().directory)
