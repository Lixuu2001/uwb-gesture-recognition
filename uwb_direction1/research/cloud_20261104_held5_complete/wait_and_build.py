"""Bounded observation of the existing controller; never start training."""
from pathlib import Path
import sys,time,subprocess,json
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');sys.path.insert(0,str(ROOT))
from common import read,write,alive,process_identity,config
from controller import valid_fold,budget_reason
from checks import verify_audits
identity=read(ROOT/'CONTROLLER_IDENTITY.json');assert alive(identity) and identity['pid']==573 and identity['startticks']==34322
start=time.monotonic();limit=1200
while True:
    assert alive(identity),'Original controller not active: inspect failure, do not relaunch automatically'
    state=read(ROOT/'STATE.json');assert state['state'] in ['auditing_members_unscored','training_fixed_members_unscored'],state
    ledger=read(ROOT/'RESOURCE_LEDGER.json');_,p=config();held=state['held']
    assert held in [5,6] and budget_reason(ledger['folds'],held,p,ledger['last_observed_combined_RSS_KiB']) is None
    if valid_fold(5) and held==6:
        verify_audits(5)
        ready=all((ROOT/'folds/held6'/arm/n).exists() for arm in ['timemil','shuffle'] for n in ['PROGRESS.json','FIRST_RELEASE.json','first_audit/TERMINATION.json'])
        if ready:
            procs=read(ROOT/'attempts/attempt002/held6/PROCESSES.json')['identities']
            if sum(alive(i) for i in procs)==2:break
    if time.monotonic()-start>=limit:raise TimeoutError('Read-only completion wait expired; original training was not stopped or modified')
    time.sleep(10)
write(B/'WAIT_OBSERVATION.json',dict(state='original_held5_protocol_complete_held6_started_no_duplicate_or_restart',observed_unix=time.time(),wait_seconds=time.monotonic()-start,controller=process_identity(identity['pid']),scientific_configuration_changed=False,new_training_process_started=False,diagnostic_labels_read=False))
with (B/'BUILD.log').open('x') as log:
    result=subprocess.run([sys.executable,str(B/'pipeline.py'),'build'],stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0,'Preserve build failure evidence; do not overwrite or retrain'
print((B/'BUILD.log').read_text())
