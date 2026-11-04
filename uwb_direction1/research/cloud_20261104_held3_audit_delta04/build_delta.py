from pathlib import Path
import sys,shutil,json
from delta import read,write,sha,create,apply
B=Path(__file__).resolve().parent;BASE=Path('/workspace/uwb-source6-held3-producer-insurance03-20261104');ROOT=Path('/workspace/uwb-softvote-source6-real-20261103')
sys.path.insert(0,str(ROOT));from common import privacy
from stream_archive import verify
archive=BASE/'held3_producer_backup';m=read(archive/'MANIFEST.json');v=read(archive/'READBACK_VERIFIED.json');rows=read(archive/'READBACK_INPUT.json');privacy(rows)
assert m['archive_sha256']==v['archive_sha256']=='c77fe20741e4e2cdde385c163a30ae44708379f8bc4bd0dd80422c495f6c3906'
assert verify(m,rows)==v and sha(archive/'MEMBERS.json')==m['members_manifest_sha256']
entry={x['path']:x for x in read(archive/'MEMBERS.json')['members']}['overlay/PAYLOAD_HASHES.json']
assert sha(BASE/'capture01/PAYLOAD_HASHES.json')==entry['sha256']
for arm in ['timemil','shuffle']:
    q=read(B/'capture01'/f'{arm}_SNAPSHOT_VERIFIED.json');assert q['native_instantiation_restore_recapture_exact'] and q['complete_member_manifest_verified'] and q['selected_model_matches_native_committed_state']
assert read(B/'capture01/timemil_SNAPSHOT_VERIFIED.json')['retained_output_audit_verified']
sh=read(B/'capture01/shuffle_SNAPSHOT_VERIFIED.json');assert sh['own_checkpoint_native_restore_recapture_exact'] and sh['own_replay_prefix_of_producer_steps_verified'] and not sh['full_training_replayed']
p=create(BASE/'capture01',B/'capture01',B/'delta_payload',m['archive_sha256'])
p.update(chain_depth=1,base_overlay_archive_member_prefix='overlay/',base_index_repository_path='uwb_direction1/results/cloud_20261104_held3_shuffle100_insurance03/held3_producer_backup/CLOUD_DOWNLOAD_INDEX.json',base_reconstruction_required=True)
write(B/'delta_payload/DELTA.json',p)
shutil.copyfile(archive/'CLOUD_DOWNLOAD_INDEX.json',B/'delta_payload/BASE_ARCHIVE_INDEX.json')
result=apply(BASE/'capture01',B/'delta_payload',B/'reconstructed01');write(B/'LOCAL_RECONSTRUCTION.json',result)
assert sha(B/'capture01/PAYLOAD_HASHES.json')==sha(B/'reconstructed01/PAYLOAD_HASHES.json')
assert sha(B/'delta.py')==sha(Path('/workspace/uwb-source6-audit-delta05-20261104/delta.py'))
write(B/'BASE_VERIFIED.json',dict(state='existing_private_base_raw_allmember_and_overlay_table_verified',archive_sha256=m['archive_sha256'],base_index_repository_path=p['base_index_repository_path'],base_payload_table_sha256=p['base_payload_table_sha256'],current_payload_table_sha256=p['current_payload_table_sha256'],chain_depth=1,generic_delta_unchanged=True))
print(json.dumps(dict(state=result['state'],current_files=result['files'],changed_files=result['changed_files'],changed_bytes=sum((B/'delta_payload/changed'/n).stat().st_size for n in p['changed']),chain_depth=1)))
