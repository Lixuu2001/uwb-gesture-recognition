from pathlib import Path
import sys,shutil,json
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103')
sys.path.insert(0,str(ROOT));from common import read,write,sha
out=B/'capture01'
for n,h in read(out/'PAYLOAD_HASHES.json').items():assert sha(out/n)==h,n
d=out/'folds/held5/timemil';src=ROOT/'folds/held5/timemil'
for n in ['inference_audit/CERTIFICATE.json','inference_audit/TERMINATION.json']:
    dest=d/n;dest.parent.mkdir(exist_ok=True);assert not dest.exists();shutil.copyfile(src/n,dest);assert sha(dest)==sha(src/n)
cert=read(d/'inference_audit/CERTIFICATE.json');term=read(d/'inference_audit/TERMINATION.json')
assert cert['state']=='pass' and not cert['full_training_replayed'] and not cert['diagnostic_labels_read']
assert term['exit_code']==0 and term['resource_stop_reason'] is None and term['cgroup_OOM_kill_delta']==0
for n,h in cert['artifact_hashes'].items():assert sha(out/Path(n).relative_to(ROOT))==h,n
write(out/'PAYLOAD_HASHES.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='PAYLOAD_HASHES.json'})
print(json.dumps(dict(state='retained_actual_TimeMIL_output_certificate_and_termination_verified',payloads=len(read(out/'PAYLOAD_HASHES.json')),own_replay_epoch=read(out/'folds/held5/shuffle/replay/PROGRESS.json')['epoch'],new_fitting=False,diagnostic_labels_read=False)))
