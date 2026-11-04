from pathlib import Path
import sys,json,shutil
from delta import read,write,sha,create,apply
B=Path(__file__).resolve().parent;BASE=Path('/workspace/uwb-source6-progress-insurance03-20261104');ROOT=Path('/workspace/uwb-softvote-source6-real-20261103')
m=read(BASE/'progress03_backup/MANIFEST.json');v=read(BASE/'progress03_backup/READBACK_VERIFIED.json');assert m['archive_sha256']==v['archive_sha256']=='ef0b343b2904657b18207c3bde8922554cbd73a46c0fe9bcad0da6fe7255039e'
assert sha(BASE/'progress03_backup/MEMBERS.json')==m['members_manifest_sha256']
entry={x['path']:x for x in read(BASE/'progress03_backup/MEMBERS.json')['members']}['overlay/PAYLOAD_HASHES.json'];assert sha(BASE/'capture01/PAYLOAD_HASHES.json')==entry['sha256']
plan=create(BASE/'capture01',B/'capture01',B/'delta_payload',m['archive_sha256'])
shutil.copyfile(BASE/'progress03_backup/CLOUD_DOWNLOAD_INDEX.json',B/'delta_payload/BASE_ARCHIVE_INDEX.json')
result=apply(BASE/'capture01',B/'delta_payload',B/'reconstructed01');write(B/'LOCAL_RECONSTRUCTION.json',result)
print(json.dumps(dict(changed_files=len(plan['changed']),current_files=len(read(B/'capture01/PAYLOAD_HASHES.json')),removed_prior_paths=len(plan['omitted_prior_paths']),changed_bytes=sum((B/'delta_payload/changed'/p).stat().st_size for p in plan['changed']))))
