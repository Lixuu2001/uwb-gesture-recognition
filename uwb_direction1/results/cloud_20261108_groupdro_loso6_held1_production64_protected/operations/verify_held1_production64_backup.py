"""Check this partial production-only archive, never release a sixfold score."""
from pathlib import Path
import sys,json,time
P=Path('/workspace/uwb-groupdro-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import read,write
from stream_archive import verify
start=time.monotonic();cpu=time.process_time();d=P/'held1_production64_backup';m=read(d/'MANIFEST.json')
assert m['held1_production64_complete'] and not m['complete_modelset'] and not m['complete_all_six'] and not m['held1_replay_included']
a=read(d/'TRANSFER_ATTEMPT_RESULTS.json');assert a['complete'] and a['verified_parts']==len(m['chunks'])==8 and a['failed_parts']==0
rows=[read(d/f'READBACK_{i:03}.json') for i in range(8)]
for r in rows:
 q=r['metadata'];assert not q['shared'] and q['permissions']==[dict(type='user',role='owner')] and q['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
 assert r['raw_fetch'] and r['download_verified'] and not r['include_base64']
v=verify(m,rows);assert not v['complete_modelset_verified']
write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
receipt=dict(state='held1_production64_only_native_q_all_records_outputs_owneronly_raw_allmember_verified_not_complete6_or_whole_job_resume',archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],scope=m['scope'],held1_production64_complete=True,held1_replay_included=False,complete_all_six=False,whole_job_resume=False,held_truth_scored=False,initial_recovery_chain_preserved=True,chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows])
write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',raw_fetch_options=dict(download_raw_file=True,include_base64=False),max_concurrency=3,**receipt))
write(P/'HELD1_PRODUCTION64_COMPOSITE_RECOVERY_INDEX.json',dict(initial_raw_baseline_all6_initial_chain=read(P/'INITIAL_COMPOSITE_RECOVERY_INDEX.json'),held1_production64_partial=read(d/'CLOUD_DOWNLOAD_INDEX.json'),scope='retain all original raw and GAP baseline/all6 initial chains plus first completed production fold; replay still running; not final cohort or whole job resume',no_retraining_to_recover_completed_production=True,do_not_overwrite_or_duplicate_active_phase=True,preserve_original_elapsed_budget=True))
write(P/'HELD1_PRODUCTION64_PRIVATE_RECEIPT.json',dict(**receipt,raw_allmember_verification=v,transfer_failed_parts=0,automatic_retries=0,verification_wall=time.monotonic()-start,verification_CPU=time.process_time()-cpu))
print(json.dumps(dict(chunks_verified=v['chunks_verified'],payload=v['tracked_payload_members_verified'],archive_members=v['archive_members'],bytes=m['archive_bytes'],complete_modelset=v['complete_modelset_verified'])))
