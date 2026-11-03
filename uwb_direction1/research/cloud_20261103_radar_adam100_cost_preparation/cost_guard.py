"""Fail-closed promotion guard; virtual evidence fixtures executed. Draft files alone never authorize a run."""
from pathlib import Path
import json,hashlib,sys,re
B=Path(__file__).resolve().parent
CURRENT_FROZEN='22f3772d42371b2994fab86658e53d1dd006a6f75724b29ba11fb23686d5994e'
CURRENT_SCOPE='complete_audited_full_ShuffleNetV2_x1_SGDM40_B15_recovery_20261103_source_held1'
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
        'verify_result.py','test_cost_math.py','test_cost_guard.py'])
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
    assert root==Path('/workspace/uwb-shufflenetv2-x1-recovery-pilot-20261103')
    assert sha(root/'FROZEN_HASHES.json')==CURRENT_FROZEN==p['current_trial_frozen_sha256']
    prior_frozen=read(root/'FROZEN_HASHES.json')
    assert len(prior_frozen)==26==p['previous_expected_frozen_file_count']
    for name,digest in prior_frozen.items():
        q=Path(name);assert not q.is_absolute() and '..' not in q.parts
        assert sha(root/q)==digest
    registration=read(root/'REMOTE_PREREGISTRATION.json')
    assert registration['state']=='real_pilot_remote_exact_readback_verified'
    assert registration['branch_head_verified'] is True
    assert registration['commit']==p['previous_real_registration_commit']==(
        '994727de9a04f8029f6a683fe917c587a81d726d')
    assert registration['frozen_sha256']==CURRENT_FROZEN
    assert registration['all45_changed_files_exact_before_Popen'] is True
    insurance=read(root/'INITIAL_PRIVATE_INSURANCE.json')
    assert insurance['state']=='owner_only_raw_allmember_initial_insurance_verified'
    assert insurance['frozen_sha256']==CURRENT_FROZEN
    assert insurance['shared'] is False and insurance['diagnostic_labels_scored'] is False
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
    artifacts=cert['artifact_hashes']
    required_artifacts=['initial12_head.pt','best_model.pt','final_full.pt',
        'split.npz','steps.jsonl','TRAIN_SELECTION.json','FILES.json',
        'validation_unscored.npz','diagnostic_unscored.npz']
    required_artifacts += ['validation_epoch%02d.npz'%i for i in range(1,41)]
    assert all(str(root/'run01'/n) in artifacts for n in required_artifacts)
    for path,digest in artifacts.items():
        assert re.fullmatch('[0-9a-f]{64}',digest)
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
    archive=manifest['archive_sha256']
    assert re.fullmatch('[0-9a-f]{64}',archive)
    members_file=backup/'MEMBERS.json'
    assert sha(members_file)==manifest['members_manifest_sha256']
    assert receipt['members_manifest_sha256']==manifest['members_manifest_sha256']
    rows=read(members_file)['members']
    assert rows and len(rows)==manifest['tracked_payload_files']
    members={r['path']:r for r in rows}
    assert len(members)==len(rows)
    for name,row in members.items():
        q=Path(name)
        assert not q.is_absolute() and '..' not in q.parts
        assert type(row['bytes']) is int and row['bytes']>=0
        assert re.fullmatch('[0-9a-f]{64}',row['sha256'])
    # Match the original finalizer's archived scientific-artifact bindings.
    bindings={str('run01/'+n):root/'run01'/n for n in [
        'best_model.pt','final_full.pt','validation_unscored.npz',
        'diagnostic_unscored.npz']}
    bindings['study/COMPLETE_AUDIT.json']=root/'COMPLETE_AUDIT.json'
    bindings['dependencies/source_ids.npz']=Path(
        '/workspace/uwb-accuracy-loop-20261029/run01/source_ids.npz')
    for name,path in bindings.items():
        assert name in members and members[name]['sha256']==sha(path)
    weights='official/weights/shufflenetv2_x1-5666bf0f80.pth'
    assert members[weights]['sha256']==(
        '5666bf0f80d1e7f8fb61f6a65742b9c27c37f3208c1f087cdcd7c1ec1aa8b848')
    return p
