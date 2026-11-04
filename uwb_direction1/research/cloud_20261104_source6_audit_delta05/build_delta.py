from pathlib import Path
import shutil,json
from delta import read,write,sha,create,apply
B=Path(__file__).resolve().parent;BASE=Path('/workspace/uwb-source6-delta-insurance04-20261104')
m=read(BASE/'delta04_backup/MANIFEST.json');v=read(BASE/'delta04_backup/READBACK_VERIFIED.json');assert m['archive_sha256']==v['archive_sha256']=='413946b9a0fc2894cc904e998dcfc6bba65f4c0c43d7a799742ba39a50d989a7'
assert sha(BASE/'delta04_backup/MEMBERS.json')==m['members_manifest_sha256']
entry={x['path']:x for x in read(BASE/'delta04_backup/MEMBERS.json')['members']}['delta_payload/CURRENT_PAYLOAD_HASHES.json'];assert sha(BASE/'reconstructed01/PAYLOAD_HASHES.json')==entry['sha256']
p=create(BASE/'reconstructed01',B/'capture01',B/'delta_payload',m['archive_sha256']);p.update(chain_depth=2,ancestor_archive_sha256='ef0b343b2904657b18207c3bde8922554cbd73a46c0fe9bcad0da6fe7255039e',base_reconstruction_required=True)
write(B/'delta_payload/DELTA.json',p)
shutil.copyfile(BASE/'delta04_backup/CLOUD_DOWNLOAD_INDEX.json',B/'delta_payload/BASE_DELTA04_INDEX.json');shutil.copyfile(BASE/'delta_payload/BASE_ARCHIVE_INDEX.json',B/'delta_payload/ANCESTOR_INSURANCE03_INDEX.json')
r=apply(BASE/'reconstructed01',B/'delta_payload',B/'reconstructed01');write(B/'LOCAL_RECONSTRUCTION.json',r)
assert sha(B/'capture01/PAYLOAD_HASHES.json')==sha(B/'reconstructed01/PAYLOAD_HASHES.json')
print(json.dumps(dict(changed_files=len(p['changed']),current_files=r['files'],changed_bytes=sum((B/'delta_payload/changed'/n).stat().st_size for n in p['changed']),chain_depth=2)))
