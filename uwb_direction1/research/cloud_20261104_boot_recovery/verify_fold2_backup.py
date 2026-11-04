from pathlib import Path
import sys,json
ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');B=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));from common import read,write,privacy,sha
from stream_archive import verify
from checks import verify_audits
from controller import valid_fold
assert valid_fold(2);verify_audits(2)
d=B/'held2_complete_backup';m=read(d/'MANIFEST.json');rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))];privacy(rows);v=verify(m,rows);write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
assert v['complete_modelset_verified'] and not m['delta_dependency'] and not m['final_scoring_authority']
entries={x['path']:x for x in read(d/'MEMBERS.json')['members']}
for arm in ['timemil','shuffle']:
 for name in ['selected.pt','final_full.pt','ARM_COMPLETE.json','inference_audit/CERTIFICATE.json','inference_audit/TERMINATION.json']:
  key=f'folds/held2/{arm}/{name}';assert entries[key]['sha256']==sha(ROOT/key)
for name in ['shuffle/replay/COMPLETE_AUDIT.json','shuffle/replay/TERMINATION.json','fusion/FOLD_FROZEN.json','fusion/diagnostic_unscored.npz','FUSION_TERMINATION.json']:
 key='folds/held2/'+name;assert entries[key]['sha256']==sha(ROOT/key)
r=dict(state='complete_held2_models_all_protocol_audits_and_three_arms_raw_privately_verified',held=2,scope=m['scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],payload_files=m['tracked_payload_files'],members_manifest_sha256=m['members_manifest_sha256'],delta_dependency=False,initial_data_runtime_source_dependencies_required=True,all5newfolds_complete=False,final_scoring_authority=False,diagnostic_labels_read=False,users7_8_labels_used=False,chunks=[{k:v for k,v in row.items() if k!='path'} for row in rows])
write(d/'DRIVE_BACKUP_RECEIPT.json',r);write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**r));print(json.dumps(v))
