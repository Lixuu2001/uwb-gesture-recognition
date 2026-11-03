from pathlib import Path
import sys,json
from snapshot import ROOT,read,write,sha
B=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT));from stream_archive import build
m=read(B/'capture01/PAYLOAD_HASHES.json')
for n,h in m.items():assert sha(B/'capture01'/n)==h,n
for arm in ['timemil','shuffle']:
 c=read(B/'capture01'/f'{arm}_SNAPSHOT_VERIFIED.json');assert c['state']=='pass_captured_native_committed_model_optimizer_RNG_journal_and_log_prefix' and not c['diagnostic_labels_read'] and not c['new_model_fits_started']
assert (B/'SYNTHETIC_SNAPSHOT_TESTS.log').read_text().rstrip().endswith('OK')
assert sha(ROOT/'stream_archive.py')==read(ROOT/'FROZEN_HASHES.json')['stream_archive.py']
names=['snapshot.py','test_snapshot.py','verify_native.py','build_backup.py','RESUME_GUIDE.md','SYNTHETIC_SNAPSHOT_TESTS.log','CAPTURE.log','NATIVE_TIMEMIL_SNAPSHOT.log','NATIVE_SHUFFLE_SNAPSHOT.log']
write(B/'INSURANCE_SOURCE_HASHES.json',{n:sha(B/n) for n in names})
files={'insurance/'+n:B/n for n in names+['INSURANCE_SOURCE_HASHES.json']}
for p in (B/'capture01').rglob('*'):
 if p.is_file():files['overlay/'+str(p.relative_to(B/'capture01'))]=p
result=build(B/'progress01_backup','softvote_source6_held2_progress01_20261103',files,'source6_held2_intermediate_epoch15_TimeMIL_epoch30_Shuffle_resume_insurance_not_final_scoring_authority',chunk_size=7340032,limit=268435456)
print(json.dumps(result))
