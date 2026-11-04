from pathlib import Path
import sys,json
from delta import read,write
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');sys.path.insert(0,str(ROOT));from common import privacy
from stream_archive import verify
b=B/'delta04_backup';m=read(b/'MANIFEST.json');rows=[read(b/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))];privacy(rows);v=verify(m,rows);write(b/'READBACK_INPUT.json',rows);write(b/'READBACK_VERIFIED.json',v)
r=dict(state='private_raw_allmember_verified_delta_and_native_reconstruction',scope=m['scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],tracked_payload_files=m['tracked_payload_files'],base_archive_sha256=m['base_archive_sha256'],standalone_recovery=False,final_scoring_authority=False,diagnostic_labels_read=False,new_model_fit=False,local_reconstruction=read(B/'LOCAL_RECONSTRUCTION.json'),native_roundtrip=read(B/'reconstructed01/AUGMENTED_VERIFIED.json'),chunks=[{k:v for k,v in row.items() if k!='path'} for row in rows])
write(b/'DRIVE_BACKUP_RECEIPT.json',r);write(b/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),base_index_relative_path='../delta_payload/BASE_ARCHIVE_INDEX.json',**r));print(json.dumps(v))
