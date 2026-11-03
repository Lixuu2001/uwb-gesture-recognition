from pathlib import Path
import sys,json
from snapshot import ROOT,read,write,sha
B=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));from stream_archive import build
for n,h in read(B/'capture01/PAYLOAD_HASHES.json').items():assert sha(B/'capture01'/n)==h,n
s=read(B/'capture01/SNAPSHOT.json');assert s['held']==2 and not s['diagnostic_labels_read']
epochs={a:s['members'][a]['progress']['epoch'] for a in ['timemil','shuffle']}
assert epochs==dict(timemil=48,shuffle=95)
for a in epochs:
 v=read(B/'capture01'/f'{a}_SNAPSHOT_VERIFIED.json');assert v['state']=='pass_captured_native_committed_model_optimizer_RNG_journal_and_log_prefix' and not v['new_model_fits_started'] and not v['diagnostic_labels_read']
for n,h in read(B/'UTILITY_LINEAGE.json')['unchanged_source'].items():assert sha(B/n)==h
assert sha(ROOT/'stream_archive.py')==read(ROOT/'FROZEN_HASHES.json')['stream_archive.py']
names=['snapshot.py','test_snapshot.py','verify_native.py','build_backup.py','verify_backup.py','README.md','PRIOR_INSURANCE_RESUME_GUIDE.md','UTILITY_LINEAGE.json','SYNTHETIC_SNAPSHOT_TESTS.log','CAPTURE.log','NATIVE_TIMEMIL_SNAPSHOT.log','NATIVE_SHUFFLE_SNAPSHOT.log']
write(B/'INSURANCE_SOURCE_HASHES.json',{n:sha(B/n) for n in names})
files={'insurance/'+n:B/n for n in names+['INSURANCE_SOURCE_HASHES.json']}
for p in (B/'capture01').rglob('*'):
 if p.is_file():files['overlay/'+str(p.relative_to(B/'capture01'))]=p
scope='source6_held2_intermediate_epoch48_TimeMIL_epoch95_Shuffle_resume_insurance_not_final_scoring_authority'
m=build(B/'progress02_backup','softvote_source6_held2_progress02_20261103',files,scope,chunk_size=7340032,limit=268435456)
m.update(state='complete_intermediate_native_checkpoint_overlay_streamed',source6_final_modelset_complete=False,final_scoring_authority=False)
write(B/'progress02_backup/MANIFEST.json',m);print(json.dumps(m))
