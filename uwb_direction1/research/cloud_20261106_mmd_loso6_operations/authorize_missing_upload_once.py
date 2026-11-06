"""Bounded recovery of a missing upload ID, never retry a known ID or an existing attempt."""
from pathlib import Path
import sys,json,os,time,re
base=Path(sys.argv[1]);i=int(sys.argv[2]);root=Path('/workspace/uwb-mmd-loso6-20261106/completed_fold_backups')
assert base.resolve()==base and base in [root/f'held{h}' for h in range(2,7)]
h=int(base.name.removeprefix('held'));suffix=f'{i:03}';manifest=json.loads((base/'MANIFEST.json').read_text());assert 0<=i<len(manifest['chunks'])
c=manifest['chunks'][i];assert c['name']==f'mmd_loso6_completed_held{h}_20261106.part{suffix}'
assert not(base/f'UPLOAD_{suffix}.json').exists() and not(base/f'READBACK_{suffix}.json').exists() and not(base/f'MISSING_ID_RETRY_{suffix}_INVOKED.json').exists()
errors=list(base.glob(f'ERROR_{suffix}_*.json'));missing=list(base.glob(f'MISSING_ID_RESPONSE_{suffix}_*.json'));assert len(errors)==len(missing)==1
e=json.loads(errors[0].read_text());assert e['i']==i and e['stage']=='upload' and e['file_id'] is None
folder='12AGMwT99yXCTYBQocQaX901Br-IelmOf';times={}
for round_id in [1,2]:
 for scope in ['parent','global']:
  q=json.loads((base/f'MISSING_ID_ABSENCE_{suffix}_round{round_id}_{scope}.json').read_text())
  expected="name = '"+c['name']+"' and trashed = false"+(" and '"+folder+"' in parents" if scope=='parent' else '')
  assert q['round']==round_id and q['scope']==scope and q['query']==expected and q['folder_id']==folder and q['exact_name']==c['name']
  assert q['results']==[] and q['provider_continuation_absent'] and q['provider_truncation_absent'] and not q['is_error'] and q['requested_topn']==100
  times[round_id,scope]=q['observed_unix']
assert all(times[2,s]>=times[1,s] for s in ['parent','global'])
marker=dict(state='one_bounded_additional_upload_after_two_complete_exactname_absence_rounds',index=i,held=h,part=c,observed_unix=time.time(),maximum_additional_attempts=1,original_errors_preserved=True,known_ID_absent=True,no_training_restart=True,scientific_negative=False)
with (base/f'MISSING_ID_RETRY_{suffix}_INVOKED.json').open('x') as f:json.dump(marker,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('one bounded additional upload marked')
