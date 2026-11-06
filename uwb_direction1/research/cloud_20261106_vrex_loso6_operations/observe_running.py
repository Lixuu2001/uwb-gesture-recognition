"""Read actual identities/progress without model inference, truth pairing, or process changes."""
from pathlib import Path
import sys,time,json
P=Path('/workspace/uwb-vrex-loso6-20261106');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha
from controller import identity
old=read(P/'PHASE_IDENTITY.json');now=identity(old['pid']);assert now and now['state']!='Z' and all(now[k]==old[k] for k in ['pid','boot_id','startticks'])
ledger=read(P/'RESOURCE_LEDGER.json');assert ledger['wall']<43200 and ledger['cpu']<43200
h=ledger['active_held'];fold=P/f'folds/held{h}';stage=ledger['fold_stage'];progress=read(fold/f'run/{stage}/PROGRESS.json')
for n,v in read(P/'FROZEN_HASHES.json').items():assert sha(P/n)==v,n
assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
a=dict(state='one_actual_full_sixfold_VREx_phase_running_unscored',observed_unix=time.time(),phase_original_identity=old,phase_actual_identity=now,resource_ledger=ledger,current_held=h,stage=stage,committed_progress=progress,native_first_release=read(fold/'run/train/FIRST_RELEASE.json'),new_training_actually_started=True,new_real_release=read(P/'REAL_RELEASE.json'),all6_declared_phase_not_sixfold_completed=True,old_MMD_source1_once_invoked=False,old_VREx_singlefold_activated=False,diagnostic_labels_read=False,users7_8_labels_used=False,first_partial_backup_is_not_complete_modelset=True,scientific_sources_unchanged=True)
for root,name in [('controller','CONTROLLER_IDENTITY.json'),('worker',f'run/execution/{stage}/IDENTITY.json')]:
 p=fold/name
 if p.exists():old=read(p);a[root]=dict(original=old,actual=identity(old['pid']))
write(P/'ACTIVE_OBSERVATION.json',a);print(json.dumps(dict(actual_phase=now,held=h,stage=stage,epoch=progress['epoch'],updates=progress['updates'],wall=ledger['wall'],cpu=ledger['cpu'])))
