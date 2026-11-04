from pathlib import Path
import sys,json
from snapshot import ROOT,read,write
B=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));from common import privacy
from stream_archive import verify
b=B/'progress03_backup';m=read(b/'MANIFEST.json');rows=[read(b/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
privacy(rows);v=verify(m,rows);write(b/'READBACK_INPUT.json',rows);write(b/'READBACK_VERIFIED.json',v)
r=dict(state='private_intermediate_native_checkpoint_overlay_all_raw_chunks_and_payloads_verified',scope=m['scope'],archive_bytes=m['archive_bytes'],archive_sha256=m['archive_sha256'],members_manifest_sha256=m['members_manifest_sha256'],tracked_payload_files=m['tracked_payload_files'],final_source6_modelset_complete=False,final_scoring_authority=False,diagnostic_labels_read=False,users7_8_labels_used=False,initial_archive_required=True,initial_archive_sha256=read(B/'capture01/INITIAL_PRIVATE_INSURANCE.json')['archive_sha256'],prior_insurance_retained=True,prior_insurance_commit='fd3e5b65ed35c1332e43e3ba3c51db014b7ae1e8',captured_members={a:read(B/'capture01'/f'{a}_SNAPSHOT_VERIFIED.json') for a in ['timemil','shuffle']},chunks=[{k:v for k,v in row.items() if k!='path'} for row in rows])
write(b/'DRIVE_BACKUP_RECEIPT.json',r);write(b/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**r));print(json.dumps(v))
