"""Verify actual live identity and completed-producer receipts, without scoring."""
from pathlib import Path
import sys,time,json
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');sys.path.insert(0,str(ROOT))
from common import read,write,sha,config,alive,process_identity
from controller import valid_fold,budget_reason
from checks import verify_member,verify_audits
identity=read(ROOT/'CONTROLLER_IDENTITY.json');assert alive(identity)
assert identity['pid']==573 and identity['startticks']==34322 and identity['boot_id']=='ad409b48-89ce-4704-beee-3658103d5856'
c,p=config()
for n,h in read(ROOT/'FROZEN_HASHES.json').items():assert sha(ROOT/n)==h,n
assert valid_fold(2);verify_audits(2)
ledger=read(ROOT/'RESOURCE_LEDGER.json')
assert ledger['folds']['2']==dict(wall=22834.174004506,cpu=84851.68278)
assert budget_reason(ledger['folds'],3,p,ledger['last_observed_combined_RSS_KiB']) is None
state=read(ROOT/'STATE.json');assert state['held']==3 and state['state']=='training_fixed_members_unscored'
_,complete=verify_member(3,'shuffle')
assert complete['epochs']==complete['selected_epoch']==100 and complete['updates']==22500
term=read(ROOT/'attempts/attempt002/held3/shuffle_train_TERM.json')
assert term['exit_code']==0 and term['resource_stop_reason'] is None and not term['wall_timeout'] and term['cgroup_OOM_kill_delta']==0
capture=B/'capture01'
copiedterm=capture/'attempts/attempt002/held3/shuffle_train_TERM.json'
assert copiedterm.exists() and sha(copiedterm)==sha(ROOT/'attempts/attempt002/held3/shuffle_train_TERM.json')
snapshot=read(capture/'SNAPSHOT.json')
assert snapshot['members']['shuffle']['state']=='complete_immutable_member' and snapshot['members']['shuffle']['progress']['epoch']==100
members={a:read(ROOT/'folds/held3'/a/'PROGRESS.json') for a in ['timemil','shuffle']}
procs=read(ROOT/'attempts/attempt002/held3/PROCESSES.json')['identities']
active=[process_identity(x['pid']) for x in procs if alive(x)]
assert len(active)==1 and active[0]['pid']==574
out=dict(state='same_registered_held3_shuffle_producer_complete_timemil_still_running_unscored',observed_unix=time.time(),controller=process_identity(identity['pid']),active_children=active,controller_state=state,current_member_progress=members,captured_member_epochs={a:snapshot['members'][a]['progress']['epoch'] for a in ['timemil','shuffle']},resource_ledger=ledger,resource_stop_reason=None,budget_reset=False,configuration_seed_protocol_changed=False,core_frozen_files_verified=len(read(ROOT/'FROZEN_HASHES.json')),held2_completed_unscored_not_retrained=True,shuffle_producer_completion_sha256=sha(ROOT/'folds/held3/shuffle/ARM_COMPLETE.json'),shuffle_producer_termination=term,full_shuffle_replay_started=(ROOT/'folds/held3/shuffle/replay/PROGRESS.json').exists(),full_shuffle_replay_complete=False,full_modelset_or_final_scoring_authority=False,diagnostic_labels_read=False,users7_8_labels_used=False,new_accuracy_result=False,original_contribution_validated=False,data_redownloaded=False)
assert not out['full_shuffle_replay_started']
write(B/'LIVE_OBSERVATION.json',out)
write(B/'SHUFFLE_ARM_COMPLETE.json',complete)
print(json.dumps(out))
