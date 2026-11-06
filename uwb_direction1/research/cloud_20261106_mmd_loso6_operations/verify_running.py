"""Private raw all-member verification of an explicitly incomplete running checkpoint."""
from pathlib import Path
import sys,json
P=Path('/workspace/uwb-mmd-loso6-20261106');sys.path.insert(0,str(P/'shared'))
from common import read,write
from stream_archive import verify
b=P/'running_backup';m=read(b/'MANIFEST.json');assert not m['complete_modelset'] and m['partial_training_checkpoint']
rows=[read(b/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
for r in rows:
 q=r['metadata'];assert q['shared'] is False and q['permissions']==[dict(type='user',role='owner')] and q['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
 assert r['raw_fetch'] and r['download_verified'] and r['include_base64'] is False
v=verify(m,rows);assert v['complete_modelset_verified'] is False
write(b/'READBACK_INPUT.json',rows);write(b/'READBACK_VERIFIED.json',v)
write(b/'CLOUD_DOWNLOAD_INDEX.json',dict(state='partial_held2_committed_epoch1_native_training_checkpoint_private_raw_allmember_verified',folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',raw_fetch_options=dict(download_raw_file=True,include_base64=False),max_concurrency=3,archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],tracked_payload_files=m['tracked_payload_files'],complete_training_modelset=False,checkpoint_epoch=m['checkpoint_epoch'],checkpoint_updates=m['checkpoint_updates'],standalone_auto_resume_permitted=False,needs_exact_initial_and_full_source_private_index=True,chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows]))
print(json.dumps(v))
