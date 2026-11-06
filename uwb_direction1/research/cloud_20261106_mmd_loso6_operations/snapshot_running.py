"""Capture an immutable committed prefix while production keeps running; no model updates."""
from pathlib import Path
import sys,json,hashlib,shutil
P=Path('/workspace/uwb-mmd-loso6-20261106');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha
from stream_archive import build
import engine as e
s=P/'folds/held2/run/train';d=P/'first_running_checkpoint';assert not d.exists()
praw=(s/'PROGRESS.json').read_bytes();p=json.loads(praw);assert p['epoch']>=1 and read(s/'FIRST_RELEASE.json')['native_first_update_exact']
d.mkdir();shutil.copyfile(s/p['slot'],d/'committed_full_state.pt');assert sha(d/'committed_full_state.pt')==p['checkpoint_sha256']
raw=(s/'steps.jsonl').read_bytes()[:p['prefix_bytes']];assert hashlib.sha256(raw).hexdigest()==p['prefix_sha256'] and len(raw.splitlines())==p['updates'];(d/'steps_prefix.jsonl').write_bytes(raw)
for epoch in range(p['epoch']+1):
 for stem in ['epoch','journal']:shutil.copyfile(s/f'{stem}{epoch:03}.json',d/f'{stem}{epoch:03}.json')
for n in ['split.npz','FIRST_READY.json','FIRST_RELEASE.json','first_update.pt']:shutil.copyfile(s/n,d/n)
(d/'PROGRESS.json').write_bytes(praw)
st=e.torch.load(d/'committed_full_state.pt',map_location='cpu',weights_only=True);rec=read(d/f'epoch{p["epoch"]:03}.json')
assert st['epoch']==p['epoch'] and st['updates']==p['updates'] and e.semantic(st)==rec['state_sha256']
receipt=dict(state='actual_committed_native_model_optimizer_allRNG_and_exact_step_prefix_copied_while_training_runs',held=2,epoch=p['epoch'],updates=p['updates'],checkpoint_sha256=p['checkpoint_sha256'],native_state_sha256=e.semantic(st),step_prefix_sha256=p['prefix_sha256'],initial_semantic_keys=sorted(st),diagnostic_labels_read=False,complete_training_modelset=False,standalone_auto_resume_permitted=False,first_native_update_exact=True)
write(d/'SNAPSHOT_VERIFIED.json',receipt)
files={x.name:x for x in d.iterdir() if x.is_file()}
for n in ['REAL_RELEASE.json','LAUNCH_REQUEST.json','LAUNCH_RECEIPT.json','PHASE_IDENTITY.json']:files['phase/'+n]=P/n
result=build(P/'running_backup','mmd_loso6_running_held2_20261106',files,'first_observed_committed_held2_training_prefix_incomplete_private_checkpoint',chunk_size=7340032,limit=67108864)
result.update(state="partial_committed_training_checkpoint_streamed",complete_modelset=False,complete_training_modelset=False,partial_training_checkpoint=True,checkpoint_epoch=p['epoch'],checkpoint_updates=p['updates']);write(P/'running_backup/MANIFEST.json',result);print(json.dumps(result))
