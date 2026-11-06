"""At most one extra producer-part upload after two complete delayed absence searches."""
from pathlib import Path
import sys,json,os,time,hashlib
base=Path(sys.argv[1]);i=int(sys.argv[2]);root=Path('/workspace/uwb-vrex-loso6-20261106/producer_backups')
assert base.resolve()==base and base in [root/f'held{h}' for h in range(1,7)]
h=int(base.name.removeprefix('held'));s=f'{i:03}';m=json.loads((base/'MANIFEST.json').read_text());assert 0<=i<len(m['chunks'])
assert m['complete_training_producer'] and not m['complete_modelset'] and not m['complete_sixfold_cohort']
c=m['chunks'][i];assert c['name']==f'vrex_loso6_producer_held{h}_20261106.part{s}'
assert (base/c['name']).stat().st_size==c['bytes'] and hashlib.sha256((base/c['name']).read_bytes()).hexdigest()==c['sha256']
assert not(base/f'UPLOAD_{s}.json').exists() and not(base/f'READBACK_{s}.json').exists() and not(base/f'MISSING_ID_RETRY_{s}_INVOKED.json').exists()
errors=list(base.glob(f'ERROR_{s}_*.json'));missing=list(base.glob(f'MISSING_ID_RESPONSE_{s}_*.json'));assert len(errors)==len(missing)==1
e=json.loads(errors[0].read_text());assert e['i']==i and e['stage']=='upload' and e['file_id'] is None
folder='12AGMwT99yXCTYBQocQaX901Br-IelmOf';times={}
for round_id in [1,2]:
 for scope in ['parent','global']:
  q=json.loads((base/f'MISSING_ID_ABSENCE_{s}_round{round_id}_{scope}.json').read_text())
  expected="name = '"+c['name']+"' and trashed = false"+(" and '"+folder+"' in parents" if scope=='parent' else '')
  assert q['round']==round_id and q['scope']==scope and q['query']==expected and q['folder_id']==folder and q['exact_name']==c['name']
  assert q['results']==[] and q['provider_continuation_absent'] and q['provider_truncation_absent'] and not q['is_error'] and q['requested_topn']==100
  times[round_id,scope]=q['observed_unix']
assert all(times[2,s]-times[1,s]>=15. for s in ['parent','global'])
marker=dict(state='one_bounded_additional_upload_after_two_complete_exactname_absence_rounds',index=i,held=h,part=c,observed_unix=time.time(),maximum_additional_attempts=1,original_errors_preserved=True,known_ID_absent=True,no_training_restart=True,scientific_negative=False)
with (base/f'MISSING_ID_RETRY_{i:03}_INVOKED.json').open('x') as f:json.dump(marker,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print('one bounded additional upload marked')
