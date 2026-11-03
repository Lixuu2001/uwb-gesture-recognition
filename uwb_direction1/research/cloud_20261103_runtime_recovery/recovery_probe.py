"""Read-only recovery observation. Never starts, resumes, scores or downloads.

Unexecuted at authoring: the cloud executor was unavailable. This probe cannot
replace the original frozen complete()/stream verification/finalization guards.
"""
from pathlib import Path
import argparse,datetime,hashlib,json,math
DEFAULT_ROOT=Path('/workspace/uwb-shufflenetv2-x1-pilot-20261103')
EXPECTED_FROZEN='1616f2fe7c642d721d9e2af984f203aa3f2ddd7889659eb0dbb2dc75ed61129b'
EXPECTED_REGISTRATION='7fd8d88d82b1570f90358dca43fc964b9b7eba88'
CERTIFICATE_KEYS=(
 'full40_epoch12560_update_replay_exact',
 'all40_validation_scores_selection_exact',
 'best_full_model_and_final_model_optimizer_momentum_rng_exact',
 'all2200_selected_scores_predictions_exact',
 'all6900_raw_RMS_cache_FP64_verified',
)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4194304),b''):h.update(block)
    return h.hexdigest()
def read_json(path,errors):
    path=Path(path)
    if not path.exists():return None
    try:return json.loads(path.read_bytes())
    except (OSError,ValueError,UnicodeError) as e:
        errors.append(dict(path=str(path),error_type=type(e).__name__))
        return None
def successful_term(value):
    return (
        isinstance(value,dict)
        and type(value.get('exit_code')) is int
        and value['exit_code']==0
        and value.get('wall_timeout') is False
        and 'resource_stop_reason' in value
        and value['resource_stop_reason'] is None
        and type(value.get('cgroup_oom_kill_delta')) is int
        and value['cgroup_oom_kill_delta']==0
    )
def process_observation(pid,script,expected_ticks=None,expected_parent=None):
    r=dict(pid=pid,verified_live=False,exists=False)
    if type(pid) is not int or pid<=0:return r
    p=Path('/proc')/str(pid)
    try:
        # comm may contain spaces or closing parentheses; do not split it.
        stat=p.joinpath('stat').read_text()
        tail=stat.rsplit(')',1)[1].split()
        state=tail[0];parent=int(tail[1]);ticks=tail[19]
        args=p.joinpath('cmdline').read_bytes().split(b'\0')
        matches=str(script).encode() in args
        identity=expected_ticks is None or ticks==str(expected_ticks)
        parent_matches=expected_parent is None or parent==expected_parent
        r.update(exists=True,process_state=state,start_ticks=ticks,
                 expected_start_ticks=expected_ticks,start_ticks_match=identity,
                 argv_script_matches=matches,parent_matches=parent_matches,
                 verified_live=state not in ('Z','X') and matches
                    and identity and parent_matches)
    except (OSError,ValueError,IndexError):
        r['process_observation_incomplete']=True
    return r
def budget_ok(producer,audit):
    if not all(successful_term(x) for x in (producer,audit)):return False
    try:
        wall=sum(float(x['supervisor_wall_seconds']) for x in (producer,audit))
        cpu=sum(float(x['child_cpu_seconds']) for x in (producer,audit))
        rss=[int(x['peak_observed_RSS_KiB']) for x in (producer,audit)]
        return (math.isfinite(wall) and math.isfinite(cpu)
                and 0<=wall<=43200 and 0<=cpu<=172800
                and all(0<=n<=14680064 for n in rss))
    except (KeyError,ValueError,TypeError):return False
def choose_route(s):
    # These are review routes, never automatic mutations or label release.
    if s['summary_exists']:return 'already_finalized_inspect_existing_provenance_do_not_rescore'
    if not s['root_exists']:return 'workspace_identity_unverified_inspect_persistent_index_not_data_loss'
    if not s['frozen_controls_verified']:return 'blocked_frozen_or_registration_guard'
    p,a=s['producer_term'],s['audit_term']
    if p is None:
        return ('wait_existing_producer_no_duplicate_start' if s['producer_live']
                else 'producer_not_verified_inspect_evidence_no_auto_resume')
    if not successful_term(p):return 'producer_execution_failure_no_scientific_negative_no_retry'
    if a is None:
        return ('wait_existing_full_replay_no_duplicate_start' if s['audit_live']
                else 'audit_not_verified_inspect_evidence_no_auto_resume')
    if not successful_term(a):return 'audit_execution_failure_diagnostic_remains_sealed'
    if not s['certificate_verified']:return 'blocked_missing_or_incomplete_full_audit_certificate'
    if not s['budget_verified']:return 'blocked_observed_resource_budget'
    if not s['dependencies_verified']:return 'verify_dependencies_before_original_complete_guard'
    if s['private_directory_exists']:
        return 'inspect_existing_private_backup_reuse_known_IDs_original_stream_verify_required'
    return 'original_complete_guard_then_backup_build_only_diagnostic_still_sealed'
def collect(root,verify_dependencies=False):
    root=Path(root);errors=[]
    s=dict(root=str(root),root_exists=root.is_dir(),
           utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
           summary_exists=(root/'run01/SUMMARY.json').exists(),
           private_directory_exists=(root/'complete_backup').exists(),
           producer_term=None,audit_term=None,producer_live=False,audit_live=False,
           frozen_controls_verified=False,dependencies_verified=False,
           certificate_verified=False,budget_verified=False,
           diagnostic_values_deserialized=False,read_only=True,
           training_started_by_probe=False,downloads_started_by_probe=False)
    if not s['root_exists']:
        s['route']=choose_route(s);return s
    frozen=read_json(root/'FROZEN_HASHES.json',errors)
    registration=read_json(root/'REMOTE_PREREGISTRATION.json',errors)
    try:
        assert sha(root/'FROZEN_HASHES.json')==EXPECTED_FROZEN
        assert isinstance(frozen,dict) and len(frozen)==22
        assert isinstance(registration,dict)
        assert registration.get('commit')==EXPECTED_REGISTRATION
        assert registration.get('state')=='real_pilot_remote_exact_readback_verified'
        assert registration.get('branch_head_verified') is True
        assert registration.get('frozen_sha256')==EXPECTED_FROZEN
        for name,digest in frozen.items():
            q=Path(name);assert not q.is_absolute() and '..' not in q.parts
            assert sha(root/q)==digest
        s['frozen_controls_verified']=True
    except (OSError,AssertionError,TypeError,KeyError):
        errors.append(dict(stage='frozen_or_registration_guard',verified=False))
    launch=read_json(root/'LAUNCH.json',errors) or {}
    launch_valid=(launch.get('commit')==EXPECTED_REGISTRATION
                  and launch.get('frozen_sha256')==EXPECTED_FROZEN)
    controller=process_observation(launch.get('controller_pid'),
                 root/'controller.py',launch.get('proc_start_ticks'))
    s['controller']=controller
    observation=read_json(root/'LIVE_OBSERVATION.json',errors) or {}
    for kind,directory,script in (
          ('producer','supervisor01','run_pilot.py'),
          ('audit','audit_supervisor01','audit_complete.py')):
        pid=read_json(root/directory/'PID.json',errors) or {}
        term=read_json(root/directory/'TERMINATION.json',errors)
        expected=(observation.get('worker_start_ticks')
                  if observation.get('worker_pid')==pid.get('pid') else None)
        proc=process_observation(pid.get('pid'),root/script,expected,
                                launch.get('controller_pid'))
        s[kind+'_process']=proc
        s[kind+'_live']=bool(launch_valid and controller['verified_live']
                            and proc['verified_live'])
        s[kind+'_term']=term
    s['last_state_record']=read_json(root/'STATE.json',errors)
    if verify_dependencies and s['frozen_controls_verified']:
        deps=read_json(root/'DEPENDENCIES.json',errors)
        try:
            assert isinstance(deps,dict) and isinstance(deps.get('files'),dict)
            for path,digest in deps['files'].items():assert sha(path)==digest
            s['dependencies_verified']=True
            s['dependency_count_verified']=len(deps['files'])
        except (OSError,AssertionError,TypeError):
            errors.append(dict(stage='dependency_hash_guard',verified=False))
    certificate=read_json(root/'COMPLETE_AUDIT.json',errors)
    if isinstance(certificate,dict):
        try:
            assert certificate.get('state')=='pass'
            assert all(certificate.get(k) is True for k in CERTIFICATE_KEYS)
            assert certificate.get('diagnostic_labels_read') is False
            assert certificate.get('users7_8_labels_used') is False
            assert certificate.get('pixel_count')==1038643200
            bound=0.00023389505667192252
            assert certificate.get('pixel_bound')==bound
            error=float(certificate['pixel_maximum_error'])
            assert math.isfinite(error) and 0<=error<=bound
            artifacts=certificate.get('artifact_hashes')
            assert isinstance(artifacts,dict) and artifacts
            for path,digest in artifacts.items():
                resolved=Path(path).resolve()
                assert resolved.is_relative_to(root.resolve())
                assert sha(resolved)==digest
            s['certificate_verified']=True
            s['certificate_SHA256']=sha(root/'COMPLETE_AUDIT.json')
        except (OSError,AssertionError,KeyError,TypeError,ValueError):
            errors.append(dict(stage='full_audit_certificate_guard',verified=False))
    s['budget_verified']=budget_ok(s['producer_term'],s['audit_term'])
    s['errors']=errors;s['route']=choose_route(s)
    return s
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--verify-dependencies',action='store_true')
    parser.add_argument('--write-observation',action='store_true')
    args=parser.parse_args();s=collect(args.root,args.verify_dependencies)
    if args.write_observation:
        assert args.root.is_dir(),'No new workspace may be invented by this probe'
        folder=args.root/'recovery_checks';folder.mkdir(exist_ok=True)
        name=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        with (folder/(name+'.json')).open('x') as f:json.dump(s,f,indent=2)
    print(json.dumps(s,indent=2))
if __name__=='__main__':main()
