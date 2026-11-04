from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103')
sys.path.insert(0,str(ROOT));from common import read,write,sha
from stream_archive import build
for n,h in read(B/'capture01/PAYLOAD_HASHES.json').items():assert sha(B/'capture01'/n)==h
for arm in ['timemil','shuffle']:
    row=read(B/'capture01'/f'{arm}_SNAPSHOT_VERIFIED.json')
    assert row['state']=='pass_captured_native_committed_model_optimizer_RNG_journal_and_log_prefix'
    assert row['native_instantiation_restore_recapture_exact']
sh=read(B/'capture01/shuffle_SNAPSHOT_VERIFIED.json')
assert sh['committed_epoch']==100 and sh['complete_member_manifest_verified'] and sh['selected_model_matches_native_committed_state']
assert sh['uncommitted_suffix_rows']==0 and not sh['full_training_replayed']
names=['snapshot.py','verify_native.py','capture_when_complete.py','observe.py','build_backup.py','verify_backup.py','prepare_publication.py','README.md','CAPTURE_TRIGGER.json','CAPTURE.log','CAPTURE_TIMEMIL.log','CAPTURE_SHUFFLE.log','LIVE_OBSERVATION.json','SHUFFLE_ARM_COMPLETE.json']
write(B/'SOURCE_HASHES.json',{n:sha(B/n) for n in names})
files={'insurance/'+n:B/n for n in names+['SOURCE_HASHES.json']}
for f in (B/'capture01').rglob('*'):
    if f.is_file():files['overlay/'+str(f.relative_to(B/'capture01'))]=f
m=build(B/'held3_producer_backup','uwb_held3_shuffle100_producer_insurance03_20261104',files,'held3_complete_shuffle100_and_running_timemil_native_insurance_not_full_audit_or_scoring_authority',chunk_size=7340032,limit=268435456)
m.update(state='held3_shuffle100_and_intermediate_timemil_recovery_archive_streamed',complete_modelset=False,delta_dependency=False,initial_source_data_runtime_dependencies_required=True,final_scoring_authority=False)
write(B/'held3_producer_backup/MANIFEST.json',m)
print(json.dumps(m))
