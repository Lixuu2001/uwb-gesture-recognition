from pathlib import Path
import sys,json
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');sys.path.insert(0,str(ROOT));from common import read,write,sha
from stream_archive import build
for n,h in read(B/'capture01/PAYLOAD_HASHES.json').items():assert sha(B/'capture01'/n)==h
for a in ['timemil','shuffle']:assert read(B/'capture01'/f'{a}_SNAPSHOT_VERIFIED.json')['state']=='pass_captured_native_committed_model_optimizer_RNG_journal_and_log_prefix'
names=['verify_state.py','snapshot.py','verify_native.py','build_backup.py','verify_backup.py','README.md','PRE_RECOVERY_OBSERVATION.json','PRE_RESUME_LOGS.json','RESUME_LAUNCH.json','PRE_CONTROLLER_IDENTITY.json','PRE_RESOURCE_LEDGER.json','PRE_STATE.json','timemil_PRIOR_SUFFIX_PROOF.json','shuffle_PRIOR_SUFFIX_PROOF.json','timemil_NATIVE_RESTORE_VERIFIED.json','shuffle_NATIVE_RESTORE_VERIFIED.json','TIMEMIL_RESTORE.log','SHUFFLE_RESTORE.log','CAPTURE.log','CAPTURE_TIMEMIL.log','CAPTURE_SHUFFLE.log','timemil_PRE_RESUME_STEPS.jsonl','shuffle_PRE_RESUME_STEPS.jsonl']
write(B/'SOURCE_HASHES.json',{n:sha(B/n) for n in names});files={'recovery/'+n:B/n for n in names+['SOURCE_HASHES.json']}
for p in (B/'capture01').rglob('*'):
 if p.is_file():files['overlay/'+str(p.relative_to(B/'capture01'))]=p
m=build(B/'held3_checkpoint_backup','uwb_held3_after_second_boot_recovery_20261104',files,'held3_native_checkpoint27_57_second_boot_recovery_insurance_not_final_scoring_authority',chunk_size=7340032,limit=268435456)
m.update(state='intermediate_held3_native_recovery_checkpoint_streamed',complete_modelset=False,delta_dependency=False,initial_source_data_runtime_dependencies_required=True,final_scoring_authority=False);write(B/'held3_checkpoint_backup/MANIFEST.json',m);print(json.dumps(m))
