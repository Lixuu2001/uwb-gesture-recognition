"""Bounded read-only wait for an existing producer, then capture without restart."""
from pathlib import Path
import sys,time,json,subprocess
B=Path(__file__).resolve().parent
ROOT=Path('/workspace/uwb-softvote-source6-real-20261103')
sys.path.insert(0,str(ROOT))
from common import read,write,alive,sha,config
from controller import budget_reason
identity=read(ROOT/'CONTROLLER_IDENTITY.json')
assert identity['pid']==573 and identity['boot_id']=='ad409b48-89ce-4704-beee-3658103d5856'
start=time.monotonic();limit=900
while not (ROOT/'folds/held3/shuffle/ARM_COMPLETE.json').exists():
    assert alive(identity),'Original controller no longer active: inspect failure, never automatically relaunch'
    _,protocol=config();ledger=read(ROOT/'RESOURCE_LEDGER.json')
    assert budget_reason(ledger['folds'],3,protocol,ledger['last_observed_combined_RSS_KiB']) is None
    state=read(ROOT/'STATE.json');assert state['held']==3 and state['state']=='training_fixed_members_unscored'
    if time.monotonic()-start>limit:raise TimeoutError('No completion within read-only wait; existing training was not stopped or modified')
    time.sleep(10)
assert alive(identity)
write(B/'CAPTURE_TRIGGER.json',dict(state='existing_shuffle_producer_completed_no_new_fit_or_restart',original_controller=identity,wait_seconds=time.monotonic()-start,observed_unix=time.time(),arm_complete_sha256=sha(ROOT/'folds/held3/shuffle/ARM_COMPLETE.json'),diagnostic_labels_read=False,scientific_configuration_changed=False))
with (B/'CAPTURE.log').open('x') as log:
    result=subprocess.run([sys.executable,str(B/'snapshot.py'),'3',str(B/'capture01')],stdout=log,stderr=subprocess.STDOUT)
assert result.returncode==0,'Capture failed; preserve the attempt, never overwrite or retrain'
print((B/'CAPTURE.log').read_text())
