from pathlib import Path
import sys,json
from delta import read,write,sha
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');sys.path.insert(0,str(ROOT));from stream_archive import build
for n,h in read(B/'capture01/PAYLOAD_HASHES.json').items():assert sha(B/'capture01'/n)==sha(B/'reconstructed01'/n)==h
for a in ['timemil','shuffle']:
 x=read(B/'capture01'/f'{a}_SNAPSHOT_VERIFIED.json');y=read(B/'reconstructed01'/f'{a}_SNAPSHOT_VERIFIED.json');assert x['state_sha256']==y['state_sha256'] and x['checkpoint_sha256']==y['checkpoint_sha256']
assert read(B/'capture01/AUGMENTED_VERIFIED.json')==read(B/'reconstructed01/AUGMENTED_VERIFIED.json')
for n,h in read(B/'SOURCE_LINEAGE.json')['unchanged_source'].items():assert sha(B/n)==h
names=['snapshot.py','verify_native.py','verify_augmented.py','delta.py','test_delta.py','seal_capture.py','build_delta.py','build_archive.py','verify_backup.py','README.md','SOURCE_LINEAGE.json','CAPTURE.log','SEAL.log','DELTA_TESTS.log','DELTA_BUILD.log','NATIVE_TIMEMIL.log','NATIVE_SHUFFLE.log','AUGMENTED.log','RECONSTRUCTED_TIMEMIL.log','RECONSTRUCTED_SHUFFLE.log','RECONSTRUCTED_AUGMENTED.log','LOCAL_RECONSTRUCTION.json']
write(B/'SOURCE_HASHES.json',{n:sha(B/n) for n in names});files={'insurance/'+n:B/n for n in names+['SOURCE_HASHES.json']}
for p in (B/'delta_payload').rglob('*'):
 if p.is_file():files['delta_payload/'+str(p.relative_to(B/'delta_payload'))]=p
for folder in ['capture01','reconstructed01']:
 for n in ['timemil_SNAPSHOT_VERIFIED.json','shuffle_SNAPSHOT_VERIFIED.json','AUGMENTED_VERIFIED.json']:files[folder+'_verification/'+n]=B/folder/n
p=read(B/'delta_payload/DELTA.json');m=build(B/'delta05_backup','softvote_source6_audit_delta05_20261104',files,'source6_held2_own_Shuffle_replay61_delta05_depth2_not_final_scoring_authority',chunk_size=7340032,limit=134217728)
m.update(state='complete_intermediate_audit_delta_streamed',complete_modelset=False,standalone_recovery=False,chain_depth=2,base_archive_sha256=p['base_archive_sha256'],ancestor_archive_sha256=p['ancestor_archive_sha256'],final_scoring_authority=False);write(B/'delta05_backup/MANIFEST.json',m);print(json.dumps(m))
