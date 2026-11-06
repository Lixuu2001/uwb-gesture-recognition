from pathlib import Path
import json,os,time,hashlib
p=Path('/workspace/uwb-mmd-loso6-20261106/completed_fold_backups/held3')
c=json.loads((p/'MANIFEST.json').read_text())['chunks'][4];assert c['name']=='mmd_loso6_completed_held3_20261106.part004'
assert not(p/'UPLOAD_004.json').exists() and not(p/'READBACK_004.json').exists() and not(p/'MISSING_ID_RETRY_004_INVOKED.json').exists()
errors=list(p.glob('ERROR_004_*.json'));assert len(errors)==1
e=json.loads(errors[0].read_text());assert e['stage']=='upload' and e['file_id'] is None
for scope in ['parent','global']:
 q=json.loads((p/f'MISSING_ID_ABSENCE_004_{scope}.json').read_text());assert q['results']==[] and q['provider_continuation_absent'] and q['provider_truncation_absent'] and not q['is_error'] and q['exact_name']==c['name']
assert len(list(p.glob('MISSING_ID_RESPONSE_004_*.json')))==1
r=dict(state='one_bounded_missingID_retry_after_two_rounds_private_parent_and_global_exactname_absence',index=4,part=c,unix=time.time(),original_errors_preserved=True,known_ID=False,maximum_additional_upload_attempts=1,no_training_restart=True,no_scientific_negative=True)
with (p/'MISSING_ID_RETRY_004_INVOKED.json').open('x') as out:json.dump(r,out,indent=2);out.write('\n');out.flush();os.fsync(out.fileno())
print('one bounded retry marked')
