from pathlib import Path
import sys,shutil,json
from delta import read,write,sha
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');sys.path.insert(0,str(ROOT));from stream_archive import build
assert sha(B/'capture01/PAYLOAD_HASHES.json')==sha(B/'delta_payload/CURRENT_PAYLOAD_HASHES.json')
for n,h in read(B/'capture01/PAYLOAD_HASHES.json').items():assert sha(B/'capture01'/n)==sha(B/'reconstructed01'/n)==h,n
for arm in ['timemil','shuffle']:
 a=read(B/'capture01'/f'{arm}_SNAPSHOT_VERIFIED.json');b=read(B/'reconstructed01'/f'{arm}_SNAPSHOT_VERIFIED.json');assert a['state_sha256']==b['state_sha256'] and a['checkpoint_sha256']==b['checkpoint_sha256'] and not b['new_model_fits_started']
assert read(B/'capture01/AUGMENTED_VERIFIED.json')==read(B/'reconstructed01/AUGMENTED_VERIFIED.json')
assert (B/'DELTA_TESTS.log').read_text().rstrip().endswith('OK')
names=['snapshot.py','verify_native.py','verify_augmented.py','delta.py','test_delta.py','build_delta.py','build_archive.py','verify_backup.py','README.md','CAPTURE.log','DELTA_TESTS.log','DELTA_BUILD.log','NATIVE_TIMEMIL.log','NATIVE_SHUFFLE.log','AUGMENTED.log','RECONSTRUCTED_TIMEMIL.log','RECONSTRUCTED_SHUFFLE.log','RECONSTRUCTED_AUGMENTED.log','LOCAL_RECONSTRUCTION.json']
write(B/'SOURCE_HASHES.json',{n:sha(B/n) for n in names})
files={'insurance/'+n:B/n for n in names+['SOURCE_HASHES.json']}
for p in (B/'delta_payload').rglob('*'):
 if p.is_file():files['delta_payload/'+str(p.relative_to(B/'delta_payload'))]=p
for prefix,d in [('captured_verification','capture01'),('reconstructed_verification','reconstructed01')]:
 for n in ['timemil_SNAPSHOT_VERIFIED.json','shuffle_SNAPSHOT_VERIFIED.json','AUGMENTED_VERIFIED.json']:files[prefix+'/'+n]=B/d/n
scope='source6_held2_delta04_over_insurance03_completed_producers_own_Shuffle_replay23_not_final_scoring_authority'
m=build(B/'delta04_backup','softvote_source6_delta04_20261104',files,scope,chunk_size=7340032,limit=268435456)
m.update(state='complete_intermediate_delta_checkpoint_insurance_streamed',complete_modelset=False,base_archive_sha256=read(B/'delta_payload/DELTA.json')['base_archive_sha256'],standalone_recovery=False,final_scoring_authority=False);write(B/'delta04_backup/MANIFEST.json',m);print(json.dumps(m))
