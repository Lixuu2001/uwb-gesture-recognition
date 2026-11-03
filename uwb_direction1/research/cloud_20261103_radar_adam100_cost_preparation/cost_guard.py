"""UNEXECUTED fail-closed promotion guard. Draft files alone never authorize a run."""
from pathlib import Path
import json,hashlib,sys
B=Path(__file__).resolve().parent
CURRENT_FROZEN='1616f2fe7c642d721d9e2af984f203aa3f2ddd7889659eb0dbb2dc75ed61129b'
CURRENT_SCOPE='complete_audited_full_ShuffleNetV2_x1_SGDM40_B15_source_held1'
def read(path):return json.loads(Path(path).read_bytes())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for a in iter(lambda:f.read(4194304),b''):h.update(a)
    return h.hexdigest()
def term_ok(t):
    return (type(t.get('exit_code')) is int and t['exit_code']==0
            and t.get('wall_timeout') is False
            and 'resource_stop_reason' in t and t['resource_stop_reason'] is None
            and type(t.get('cgroup_oom_kill_delta')) is int
            and t['cgroup_oom_kill_delta']==0)
def ready():
    if sys.flags.optimize:
        raise RuntimeError('Assertions must remain enabled; do not use python -O.')
    # POLICY.json, activation, source-test evidence and the remote marker must be
    # created only in a separately selected/frozen/registered future cost study.
    # COST_POLICY_DRAFT.json is intentionally not a fallback.
    p=read(B/'POLICY.json')
    assert p['draft_state']=='selected_frozen_cost_configuration'
    marker=read(B/'REMOTE_COST_PREREGISTRATION.json')
    assert marker['state']=='remote_cost_preregistration_exact_readback_verified'
    assert marker['branch_head_verified'] is True
    assert marker['cost_hashes_sha256']==sha(B/'COST_HASHES.json')
    assert marker['source_synthetic_tests_executed'] is True
    assert marker['source_tests_log_sha256']==sha(B/'NATIVE_SOURCE_TESTS.log')
    frozen=read(B/'COST_HASHES.json')
    assert all(n in frozen for n in ['POLICY.json','DEPENDENCIES.json',
        'CURRENT_TRIAL_ACTIVATION.json','NATIVE_SOURCE_TESTS.log',
        'cost_math.py','cost_guard.py','profile_cost.py','bounded_screen.py',
        'verify_result.py','test_cost_math.py'])
    for name,digest in frozen.items():
        q=Path(name);assert not q.is_absolute() and '..' not in q.parts
        assert sha(B/q)==digest,name
    deps=read(B/'DEPENDENCIES.json')['files']
    native=(B/p['native_module_directory']).resolve()
    required=[native/'native_adam100.py',native/'test_adam100_native.py',
        Path('/workspace/uwb-shufflenetv2-x1-source-20261103/native_shufflenet.py'),
        Path('/workspace/uwb-shufflenetv2-x1-source-20261103/local_author_sources/shufflenetv2.py'),
        Path('/workspace/uwb-shufflenetv2-x1-source-20261103/weights/shufflenetv2_x1-5666bf0f80.pth'),
        Path('/workspace/uwb-resnet18-finetune-source-20261102/native_finetune.py'),
        Path('/workspace/uwb-resnet18-imagenet-source-20261102/native_resnet18.py'),
        Path(sys.executable).resolve()]
    assert all(str(path) in deps for path in required)
    for path,digest in deps.items():
        assert sha(path)==digest,path
    activation=read(B/'CURRENT_TRIAL_ACTIVATION.json')
    assert activation['state']==p['required_activation_state']
    assert activation['current_trial_closed_without_rescore'] is True
    assert marker['activation_sha256']==sha(B/'CURRENT_TRIAL_ACTIVATION.json')
    root=Path(p['current_trial_root'])
    assert root==Path('/workspace/uwb-shufflenetv2-x1-pilot-20261103')
    assert sha(root/'FROZEN_HASHES.json')==CURRENT_FROZEN==p['current_trial_frozen_sha256']
    for name,digest in read(root/'FROZEN_HASHES.json').items():
        q=Path(name);assert not q.is_absolute() and '..' not in q.parts
        assert sha(root/q)==digest
    assert activation['source_summary_sha256']==sha(root/'run01/SUMMARY.json')
    summary=read(root/'run01/SUMMARY.json')
    assert summary.get('predeclared_gate_passed') is False
    assert summary.get('complete_private_modelset_verified') is True
    assert summary.get('users7_8_labels_used') is False
    assert summary.get('route')=='close_exact_full_ShuffleNetV2_x1_SGDM40_B15_seed20261029_no_rescue'
    for n in ['supervisor01','audit_supervisor01']:
        assert term_ok(read(root/n/'TERMINATION.json'))
    cert=read(root/'COMPLETE_AUDIT.json')
    assert cert.get('state')=='pass' and cert.get('diagnostic_labels_read') is False
    assert cert.get('users7_8_labels_used') is False
    for key in ['full40_epoch12560_update_replay_exact',
        'all40_validation_scores_selection_exact',
        'best_full_model_and_final_model_optimizer_momentum_rng_exact',
        'all2200_selected_scores_predictions_exact',
        'all6900_raw_RMS_cache_FP64_verified']:
        assert cert.get(key) is True
    assert cert.get('pixel_count')==1038643200
    for path,digest in cert['artifact_hashes'].items():
        q=Path(path).resolve();assert q.is_relative_to(root.resolve())
        assert sha(q)==digest
    backup=root/'complete_backup'
    assert activation['source_private_receipt_sha256']==sha(backup/'DRIVE_BACKUP_RECEIPT.json')
    receipt=read(backup/'DRIVE_BACKUP_RECEIPT.json')
    verified=read(backup/'READBACK_VERIFIED.json')
    manifest=read(backup/'MANIFEST.json')
    assert receipt['state']=='private_readback_verified'
    assert verified['state']=='remote_archive_and_all_members_verified'
    assert receipt['scope']==verified['scope']==manifest['scope']==CURRENT_SCOPE
    assert receipt['complete_modelset_verified'] is True
    assert verified['complete_modelset_verified'] is True
    assert receipt['numeric_audit_certified'] is True
    assert receipt['archive_sha256']==verified['archive_sha256']==manifest['archive_sha256']
    assert receipt['diagnostic_labels_scored'] is False
    return p
