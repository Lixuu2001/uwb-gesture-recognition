from pathlib import Path
import sys,time,json
B=Path(__file__).resolve().parent;ROOT=Path('/workspace/uwb-softvote-source6-real-20261103');sys.path.insert(0,str(ROOT))
from common import read,write,sha,alive,process_identity,config
from checks import verify_member,verify_audits
from controller import valid_fold,budget_reason
identity=read(ROOT/'CONTROLLER_IDENTITY.json');assert alive(identity)
assert identity['pid']==573 and identity['startticks']==34322 and identity['boot_id']=='ad409b48-89ce-4704-beee-3658103d5856'
for n,h in read(ROOT/'FROZEN_HASHES.json').items():assert sha(ROOT/n)==h,n
assert valid_fold(2);verify_audits(2)
_,p=config();ledger=read(ROOT/'RESOURCE_LEDGER.json')
assert ledger['folds']['2']==dict(wall=22834.174004506,cpu=84851.68278)
assert budget_reason(ledger['folds'],3,p,ledger['last_observed_combined_RSS_KiB']) is None
state=read(ROOT/'STATE.json');assert state==dict(state='auditing_members_unscored',held=3)
complete={};terms={}
for arm in ['timemil','shuffle']:
    _,a=verify_member(3,arm);complete[arm]={k:v for k,v in a.items() if k!='files'}
    t=read(ROOT/'attempts/attempt002/held3'/f'{arm}_train_TERM.json')
    assert t['exit_code']==0 and t['resource_stop_reason'] is None and t['cgroup_OOM_kill_delta']==0
    terms[arm]=t
cert=read(ROOT/'folds/held3/timemil/inference_audit/CERTIFICATE.json');term=read(ROOT/'folds/held3/timemil/inference_audit/TERMINATION.json')
assert cert['state']=='pass' and not cert['full_training_replayed'] and not cert['diagnostic_labels_read']
assert term['exit_code']==0 and term['resource_stop_reason'] is None and term['cgroup_OOM_kill_delta']==0
for n,h in cert['artifact_hashes'].items():assert sha(n)==h,n
procs=read(ROOT/'attempts/attempt002/held3/PROCESSES.json')['identities'];active=[process_identity(x['pid']) for x in procs if alive(x)]
assert len(active)==1
assert b'replay_shuffle.py' in Path(f'/proc/{active[0]["pid"]}/cmdline').read_bytes()
captured=read(B/'capture01/folds/held3/shuffle/replay/PROGRESS.json');current=read(ROOT/'folds/held3/shuffle/replay/PROGRESS.json')
assert captured['epoch']==7 and captured['updates']==1575 and current['epoch']>=captured['epoch']
o=dict(state='same_registered_held3_both_producers_complete_TimeMIL_output_audit_pass_own_Shuffle_replay_running_unscored',observed_unix=time.time(),controller=process_identity(identity['pid']),active_children=active,controller_state=state,complete_producers=complete,actual_producer_terminations=terms,timemil_output_audit={k:v for k,v in cert.items() if k!='artifact_hashes'},timemil_output_termination=term,captured_own_replay=captured,current_own_replay=current,resource_ledger=ledger,resource_stop_reason=None,budget_reset=False,scientific_configuration_changed=False,seed_changed=False,held2_completed_unscored_not_retrained=True,core_frozen_files_verified=len(read(ROOT/'FROZEN_HASHES.json')),full100_shuffle_replay_complete=False,diagnostic_labels_read=False,users7_8_labels_used=False,new_accuracy_result=False,original_contribution_validated=False,data_redownloaded=False)
write(B/'LIVE_OBSERVATION.json',o);print(json.dumps(o))
