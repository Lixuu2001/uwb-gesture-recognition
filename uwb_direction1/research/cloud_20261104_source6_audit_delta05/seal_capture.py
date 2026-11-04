from pathlib import Path
import json,shutil,time,hashlib
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');S=B/'capture01'
for n in ['CERTIFICATE.json','TERMINATION.json']:
 d=S/'folds/held2/timemil/inference_audit'/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/'folds/held2/timemil/inference_audit'/n,d)
s=json.loads((S/'SNAPSHOT.json').read_text());e=s['members']['shuffle']['own_replay']['progress']['epoch'];required=(ROOT/'folds/held2/shuffle/replay'/f'journal{e:03}.json').stat().st_mtime
for _ in range(50):
 l=json.loads((ROOT/'RESOURCE_LEDGER.json').read_text())
 if l['last_observed_unix']>=required:break
 time.sleep(.2)
assert l['last_observed_unix']>=required
(S/'RESOURCE_LEDGER.json').write_text(json.dumps(l,sort_keys=True,indent=2)+'\n')
files={str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(S.rglob('*')) if p.is_file() and p.name!='PAYLOAD_HASHES.json'}
(S/'PAYLOAD_HASHES.json').write_text(json.dumps(files,sort_keys=True,indent=2)+'\n');print(json.dumps(dict(own_audit_epoch=e,payloads=len(files),cumulative_ledger_conservative=True)))
