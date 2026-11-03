"""Unselected cost draft. Require current full100 closure, not old SGDM40.

Only metadata/hashes and the original registered completion guard are used;
this module never decodes UWB labels, profiles a model or starts a child.
"""
from pathlib import Path
import hashlib,json,sys,importlib.util
B=Path(__file__).resolve().parent
CURRENT_ROOT=Path('/workspace/uwb-shufflenetv2-x1-adam100-recovery45-20261103')
CURRENT_FREEZE='fb5d9f587d7aa1b12638c62bacc74704e3c646c5cf1e7338e591f9d69a883b34'
CURRENT_SCOPE='complete_audited_full_ShuffleNetV2_x1_Adam100_B20_checkpoint45_recovery_source_held1'
CURRENT_REGISTRATION='142f3938798fae58a639f4d7b38b4b1eae90a1ec'
FAIL_ROUTE='close_exact_full_ShuffleNetV2_x1_Adam100_B20_noAug_final100_seed20261029_no_rescue'
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def term_ok(t):
    return (type(t.get('exit_code')) is int and t['exit_code']==0
        and t.get('wall_timeout') is False and t.get('resource_stop_reason') is None
        and 'resource_stop_reason' in t
        and type(t.get('cgroup_oom_kill_delta')) is int and t['cgroup_oom_kill_delta']==0)

def registered_complete(root):
    # Different module name prevents collision with the old Adam100 cost_guard
    # imported by the current trial's original dependency guards.
    spec=importlib.util.spec_from_file_location('registered_current_Adam100_complete',root/'trial_common.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.complete()

def validate_closed_artifacts(root,c):
    s=read(root/'run01/SUMMARY.json')
    assert s['predeclared_gate_passed'] is False
    assert s['complete_private_modelset_verified'] is True
    assert s['users7_8_labels_used'] is False and s['route']==FAIL_ROUTE
    a=read(root/'COMPLETE_AUDIT.json')
    assert a['state']=='pass' and a['diagnostic_labels_read'] is False
    assert a['all100_replay_epoch_journals_committed'] is True
    assert a['checkpoint45_operational_continuation_disclosed'] is True
    assert a['pixel_count']==1038643200
    p=root/'complete_backup';m=read(p/'MANIFEST.json');v=read(p/'READBACK_VERIFIED.json');r=read(p/'DRIVE_BACKUP_RECEIPT.json')
    assert m['scope']==v['scope']==r['scope']==CURRENT_SCOPE
    assert m['complete_modelset'] is True and v['complete_modelset_verified'] is True
    assert r['state']=='private_readback_verified' and r['complete_modelset_verified'] is True
    assert r['numeric_audit_certified'] is True and r['diagnostic_labels_scored'] is False
    assert v['state']=='remote_archive_and_all_members_verified'
    assert m['archive_sha256']==v['archive_sha256']==r['archive_sha256']
    assert sha(p/'MEMBERS.json')==m['members_manifest_sha256']==r['members_manifest_sha256']
    entries=read(p/'MEMBERS.json')['members'];members={e['path']:e for e in entries}
    assert entries and len(entries)==len(members)==m['tracked_payload_files']
    targets={'run01/'+n:root/'run01'/n for n in [
        'initial12_head.pt','final_model.pt','final_full.pt','split.npz',
        'steps.jsonl','TRAIN_ENDPOINT.json','validation_unscored.npz','diagnostic_unscored.npz']}
    targets['study/COMPLETE_AUDIT.json']=root/'COMPLETE_AUDIT.json'
    targets['audit01/PROGRESS.json']=root/'audit01/PROGRESS.json'
    for i in range(1,101):targets[f'audit01/epoch{i:03d}.json']=root/'audit01'/f'epoch{i:03d}.json'
    targets['dependencies/source_ids.npz']=Path(c['source_manifest'])
    for name,path in targets.items():
        assert members[name]['sha256']==sha(path) and members[name]['bytes']==path.stat().st_size
    assert members['official/weights/shufflenetv2_x1-5666bf0f80.pth']['sha256']==c['weights_sha256']
    return s

def prior_closed(p):
    root=Path(p['current_trial_root']);assert root==CURRENT_ROOT
    assert sha(root/'FROZEN_HASHES.json')==p['current_trial_frozen_sha256']==CURRENT_FREEZE
    frozen=read(root/'FROZEN_HASHES.json');assert len(frozen)==25
    for name,digest in frozen.items():
        q=Path(name);assert not q.is_absolute() and '..' not in q.parts
        assert sha(root/q)==digest
    marker=read(root/'REMOTE_REAL_PREREGISTRATION.json')
    assert marker['state']=='real_trial_exact_remote_readback_verified' and marker['branch_head_verified'] is True
    assert marker['commit']==CURRENT_REGISTRATION==p['previous_real_registration_commit']
    assert marker['frozen_sha256']==CURRENT_FREEZE
    c=registered_complete(root)
    return validate_closed_artifacts(root,c)

def ready():
    if sys.flags.optimize:raise RuntimeError('Do not disable assertions')
    p=read(B/'POLICY.json');assert p['draft_state']=='selected_frozen_cost_configuration'
    marker=read(B/'REMOTE_COST_PREREGISTRATION.json')
    assert marker['state']=='remote_cost_preregistration_exact_readback_verified'
    assert marker['branch_head_verified'] is True
    assert marker['cost_hashes_sha256']==sha(B/'COST_HASHES.json')
    assert marker['source_synthetic_tests_executed'] is True
    assert marker['source_tests_log_sha256']==sha(B/'NATIVE_SOURCE_TESTS.log')
    frozen=read(B/'COST_HASHES.json')
    required=['POLICY.json','DEPENDENCIES.json','CURRENT_TRIAL_ACTIVATION.json',
        'NATIVE_SOURCE_TESTS.log','activation_guard.py','cost_math.py','profile_cost.py',
        'bounded_screen.py','verify_result.py','test_cost_math.py','test_activation_guard.py']
    assert all(name in frozen for name in required)
    for name,digest in frozen.items():
        q=Path(name);assert not q.is_absolute() and '..' not in q.parts
        assert sha(B/q)==digest
    native=(B/p['native_module_directory']).resolve();deps=read(B/'DEPENDENCIES.json')['files']
    required_paths=[native/'native_resnet18_adam100.py',native/'test_native.py',
        Path('/workspace/uwb-resnet18-finetune-source-20261102/native_finetune.py'),
        Path('/workspace/uwb-resnet18-imagenet-source-20261102/native_resnet18.py'),
        Path('/workspace/uwb-resnet18-imagenet-source-20261102/local_author_sources/resnet.py'),
        Path('/workspace/uwb-resnet18-imagenet-source-20261102/weights/resnet18-f37072fd.pth'),
        Path(sys.executable).resolve()]
    assert all(str(path) in deps for path in required_paths)
    for name,digest in deps.items():assert sha(name)==digest
    a=read(B/'CURRENT_TRIAL_ACTIVATION.json')
    assert a['state']==p['required_activation_state'] and a['current_trial_closed_without_rescore'] is True
    assert marker['activation_sha256']==sha(B/'CURRENT_TRIAL_ACTIVATION.json')
    summary=prior_closed(p);root=Path(p['current_trial_root'])
    assert a['source_summary_sha256']==sha(root/'run01/SUMMARY.json')
    assert a['source_private_receipt_sha256']==sha(root/'complete_backup/DRIVE_BACKUP_RECEIPT.json')
    assert (p['trainable_parameters'],p['epochs'],p['batch_size'],p['seed'])==(11182668,100,20,20261029)
    return p
