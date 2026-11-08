"""Remove only redundant verified gzip download-cache bytes; preserve original86 and receipts."""
from pathlib import Path
import json,time,os,sys
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import ready,read,write,sha

def main():
 start=time.monotonic();cpu=time.process_time();ready()
 assert read(P/'STATE.json')['state']=='all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once'
 assert not(P/'SUMMARY.json').exists() and not(P/'RAW_TRANSFER_CACHE_RECLAIM_VERIFIED.json').exists()
 cache=Path('/workspace/uwb-transfer-20261103');raw=Path('/workspace/uwb-restored-20261103')
 manifest=read(cache/'TRANSFER_MANIFEST.json');assert manifest['state']=='complete' and len(manifest['files'])==86
 persistent=read(P/'INITIAL_COMPOSITE_RECOVERY_INDEX.json')['raw_arrays_original86_persistent']
 chunk_paths={c['path'] for f in manifest['files'] for c in f['chunks']}
 entries={r['path']:r for r in persistent['entries'] if r['path'] in chunk_paths};assert len(entries)==len(chunk_paths)==163
 receipts={str(Path(r['path']).relative_to(cache)):r for f in (cache/'receipts').glob('*.json') for r in [read(f)]};assert len(receipts)==163
 verified_raw=[];chunks=[];before=os.statvfs(cache).f_bavail*os.statvfs(cache).f_frsize
 for row in manifest['files']:
  path=raw/row['path'];assert path.resolve().is_relative_to(raw.resolve()) and path.is_file() and not path.is_symlink()
  assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
  verified_raw.append(dict(path=str(path),bytes=row['bytes'],sha256=row['sha256']))
  for c in row['chunks']:
   f=cache/c['path'];r=receipts[c['path']];e=entries[c['path']];m=r['metadata']
   assert f.resolve().is_relative_to((cache/'chunks').resolve()) and f.is_file() and not f.is_symlink()
   assert f.stat().st_size==r['bytes']==e['bytes']==c['bytes'] and sha(f)==r['sha256']==e['sha256']==c['sha256']
   assert r['file_id']==e['id']==m['id'] and not m['shared'] and m['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
   assert m['permissions']==[dict(type='user',role='owner')] and r['raw_fetch'] and r['include_base64'] is False and r['immediately_materialized_SHA_verified']
   chunks.append(dict(path=str(f),bytes=c['bytes'],sha256=c['sha256'],persistent_file_id=r['file_id']))
 assert len(chunks)==163 and len({r['path'] for r in chunks})==163 and sum(r['bytes'] for r in verified_raw)==2141981100
 plan=dict(state='all_original86_and_redundant163_cached_chunks_rechecked_before_cache_only_eviction',raw_files=verified_raw,cache_chunks=chunks,cache_bytes=sum(r['bytes'] for r in chunks),persistent_index_sha256=sha(P/'INITIAL_COMPOSITE_RECOVERY_INDEX.json'),transfer_manifest_sha256=sha(cache/'TRANSFER_MANIFEST.json'),local_manifest_receipts_and_Drive_objects_retained=True,remote_metadata_and_raw_receipts_reused_not_new_fetch=True,current_private_folder_observation_file='held6_completed_backup/DESTINATION_PRIVACY_VERIFIED.json',science_models_predictions_original_raw_files_untouched=True,free_bytes_before=before)
 write(P/'RAW_TRANSFER_CACHE_RECLAIM_PLAN.json',plan)
 for c in chunks:
  f=Path(c['path']);assert f.is_file() and not f.is_symlink() and f.stat().st_size==c['bytes'];f.unlink()
 for r in verified_raw:assert Path(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256']
 ready();after=os.statvfs(cache).f_bavail*os.statvfs(cache).f_frsize
 result=dict(state='redundant_raw_download_gzip_cache_only_evicted_after_full_original_and_receipt_SHA_checks',raw_files_retained_and_rechecked=86,raw_bytes_retained=2141981100,cache_parts_removed=163,cache_logical_bytes_reclaimed=plan['cache_bytes'],free_bytes_before=before,free_bytes_after=after,actual_filesystem_available_increase=after-before,all_163_persistent_private_Drive_IDs_and_raw_readback_receipts_retained=True,manifest_and_receipts_retained=True,models_predictions_and_method_sources_untouched=True,held_labels_or_raw_array_values_decoded=False,source_frozen_unchanged=True,no_new_fit_or_download=True,plan_sha256=sha(P/'RAW_TRANSFER_CACHE_RECLAIM_PLAN.json'),operator_sha256=sha(__file__),wall=time.monotonic()-start,CPU=time.process_time()-cpu)
 write(P/'RAW_TRANSFER_CACHE_RECLAIM_VERIFIED.json',result);print(json.dumps(result))
if __name__=='__main__':main()
