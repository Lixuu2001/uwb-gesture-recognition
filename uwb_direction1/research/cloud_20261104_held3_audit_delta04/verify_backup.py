from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');sys.path.insert(0,str(ROOT))
from common import read,write,privacy
from stream_archive import verify
b=B/'delta04_backup';m=read(b/'MANIFEST.json');rows=[read(b/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))];privacy(rows)
v=verify(m,rows);write(b/'READBACK_INPUT.json',rows);write(b/'READBACK_VERIFIED.json',v)
r=dict(state='held3_complete_producers_own_replay_checkpoint_delta_raw_allmember_verified',scope=m['scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],payload_files=m['tracked_payload_files'],members_manifest_sha256=m['members_manifest_sha256'],delta_dependency=True,chain_depth=1,base_archive_sha256=m['base_archive_sha256'],base_index_repository_path=read(B/'delta_payload/DELTA.json')['base_index_repository_path'],initial_source_data_runtime_dependencies_required=True,final_scoring_authority=False,diagnostic_labels_read=False,new_accuracy_result=False,full_training_audit_complete=False,chunks=[{k:v for k,v in row.items() if k!='path'} for row in rows])
write(b/'DRIVE_BACKUP_RECEIPT.json',r);write(b/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**r));print(json.dumps(v))
