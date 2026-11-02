"""Remove duplicate operational copies only after private/member verification.

Retains verified remote_reassembled archive, manifest, all receipts, raw download
files and MEMBERS. Never touches original producer, scientific code or data.
"""
from pathlib import Path
import json,os
from producer_recovery import validate,read,sha,core,SCOPE,HERE
def write(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
def main():
 b=HERE/'snapshot01';assert not (b/'TRANSPORT_CLEANUP.json').exists()
 _,original,_,_=validate();m=read(b/'MANIFEST.json');v=read(b/'READBACK_VERIFIED.json');r=read(b/'DRIVE_BACKUP_RECEIPT.json')
 assert r['state']=='private_readback_verified' and m['scope']==v['scope']==r['scope']==SCOPE
 assert v['producer_training_bytes_recoverable'] and not v['numeric_audit_certified'] and not r['complete_modelset_verified']
 keep=b/'remote_reassembled.tar.gz';assert keep.stat().st_size==m['archive_bytes'] and sha(keep)==m['archive_sha256']==r['archive_sha256']
 checked=core().verify_members(keep,m['members_manifest_sha256'],m['tracked_payload_files']);assert checked['archive_members']==v['archive_members']
 members=b/'payload/MEMBERS.json';assert sha(members)==m['members_manifest_sha256']
 planned=[]
 def candidate(p,size,digest,kind):
  p=Path(p);assert p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(b.resolve()) and p!=keep and os.stat(p).st_nlink==1
  assert p.stat().st_size==size and sha(p)==digest
  planned.append(dict(path=str(p.relative_to(b)),bytes=size,sha256=digest,kind=kind))
 candidate(b/m['archive_name'],m['archive_bytes'],m['archive_sha256'],'duplicate_archive')
 for c in m['chunks']:candidate(b/c['name'],c['bytes'],c['sha256'],'uploaded_chunk_copy')
 for x in read(members)['members']:
  core().relative_name(x['path']);candidate(b/'payload'/x['path'],x['bytes'],x['sha256'],'duplicate_packaging_payload')
 assert len({x['path'] for x in planned})==len(planned)
 plan=dict(state='all_duplicate_transport_bytes_checked_before_pruning',retained_archive=str(keep),retained_archive_sha256=m['archive_sha256'],retained_members_manifest_sha256=m['members_manifest_sha256'],producer_scope=SCOPE,removed_files=planned,logical_bytes_to_reclaim=sum(x['bytes'] for x in planned),original_producer_FILES_unchanged=True)
 write(b/'TRANSPORT_CLEANUP_PLAN.json',plan)
 for x in planned:(b/x['path']).unlink()
 assert validate()[1]==original and keep.is_file() and sha(keep)==m['archive_sha256'] and sha(members)==m['members_manifest_sha256']
 proof=dict(state='verified_duplicate_transport_copies_pruned_originals_preserved',plan_sha256=sha(b/'TRANSPORT_CLEANUP_PLAN.json'),removed_file_count=len(planned),logical_bytes_reclaimed=plan['logical_bytes_to_reclaim'],retained_verified_archive=keep.name,retained_archive_sha256=m['archive_sha256'],all_original_producer_models_logs_IDs_predictions_and1401frozen_bytes_unchanged=True,private_index_and_receipts_unchanged=True,raw_download_files_retained=True,models_removed_from_original_producer=False,raw_arrays_removed=False,numeric_audit_certified=False,complete_modelset_verified=False)
 write(b/'TRANSPORT_CLEANUP.json',proof);print(json.dumps(proof))
if __name__=='__main__':main()
