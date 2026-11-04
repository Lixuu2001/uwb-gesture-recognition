from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');sys.path.insert(0,str(ROOT))
from common import read,write,sha
from stream_archive import build
p=read(B/'delta_payload/DELTA.json')
for arm in ['timemil','shuffle']:
    a=read(B/'capture01'/f'{arm}_SNAPSHOT_VERIFIED.json');r=read(B/'reconstructed01'/f'{arm}_SNAPSHOT_VERIFIED.json')
    for k in ['state_sha256','checkpoint_sha256','committed_epoch','committed_updates','own_replay_epoch','native_instantiation_restore_recapture_exact','retained_output_audit_verified','own_checkpoint_native_restore_recapture_exact','own_replay_prefix_of_producer_steps_verified']:
        assert a[k]==r[k],(arm,k)
    assert r['complete_member_manifest_verified'] and r['selected_model_matches_native_committed_state']
names=['snapshot.py','augment_capture.py','verify_native.py','delta.py','build_delta.py','build_backup.py','verify_backup.py','observe.py','prepare_publication.py','README.md','CAPTURE.log','CAPTURE_TIMEMIL.log','CAPTURE_SHUFFLE.log','RECONSTRUCTED_TIMEMIL.log','RECONSTRUCTED_SHUFFLE.log','LOCAL_RECONSTRUCTION.json','BASE_VERIFIED.json','LIVE_OBSERVATION.json']
write(B/'SOURCE_HASHES.json',{n:sha(B/n) for n in names})
files={'insurance/'+n:B/n for n in names+['SOURCE_HASHES.json']}
for root,prefix in [(B/'delta_payload','delta_payload'),(B/'capture01','capture_proofs'),(B/'reconstructed01','reconstructed_proofs')]:
    for f in root.rglob('*'):
        if f.is_file() and (prefix=='delta_payload' or f.name.endswith('_SNAPSHOT_VERIFIED.json')):files[prefix+'/'+str(f.relative_to(root))]=f
m=build(B/'delta04_backup','uwb_held3_complete_producers_own7_delta04_20261104',files,'held3_complete_producers_own_replay_checkpoint_delta_depth1_not_full_audit_or_scoring_authority',chunk_size=7340032,limit=268435456)
m.update(state='held3_complete_producers_own_audit_checkpoint_delta_streamed',complete_modelset=False,delta_dependency=True,chain_depth=1,base_archive_sha256=p['base_archive_sha256'],initial_source_data_runtime_dependencies_required=True,final_scoring_authority=False)
write(B/'delta04_backup/MANIFEST.json',m);print(json.dumps(m))
