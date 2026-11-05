"""Wait for ORIGINAL pilot; audit completed evidence and build once, never fit/score/upload."""
from pathlib import Path
import fcntl,hashlib,json,os,subprocess,sys,time,traceback
M=Path(__file__).resolve().parent;B=Path('/workspace/uwb-mixstyle-gap64-20261105')
sys.path.insert(0,str(B))
def save(n,q):
    p=M/n;tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(q,indent=2)+'\n');tmp.replace(p)
def optional_progress(path):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return None
def main():
    lock=(M/'audit_build.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from common import read,ready,sha
    from controller import identity
    import journal,engine as e
    ready(True);old=read(Path('/workspace/uwb-mixstyle-gap-meta-20261105')/'RECOVERY_CONTROLLER_IDENTITY.json');start=time.monotonic();save('OBSERVER_IDENTITY.json',identity(os.getpid()))
    while True:
        state=read(B/'STATE.json') if (B/'STATE.json').exists() else None
        if state and state['state']=='pilot_audited_frozen_unscored_requires_complete_private_backup_before_once_gate':break
        now=identity(old['pid'])
        if now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['pid','startticks','boot_id']):
            state=read(B/'STATE.json') if (B/'STATE.json').exists() else None
            assert state and state['state']=='pilot_audited_frozen_unscored_requires_complete_private_backup_before_once_gate','supervisor unexpectedly inactive before terminal state'
            break
        ledger=read(B/'RESOURCE_LEDGER.json');assert ledger['wall']<=43200 and ledger['cpu']<=43200 and ledger['RSS_KiB']<=12582912
        assert not (B/'SUMMARY.json').exists() and not (B/'FINALIZE_ONCE.lock').exists()
        p=B/'run'/('replay' if ledger['stage']=='replay' else 'train')/'PROGRESS.json';progress=optional_progress(p)
        save('OBSERVER_PROGRESS.json',dict(observed_unix=time.time(),stage=ledger['stage'],progress=progress,progress_file_ready=progress is not None,resource_ledger=ledger,new_fit_started=False,diagnostic_labels_read=False))
        assert time.monotonic()-old['startticks']/os.sysconf('SC_CLK_TCK')<43200,'bounded observation timeout; retain original healthy job'
        time.sleep(5)
    state=read(B/'STATE.json');assert state['state']=='pilot_audited_frozen_unscored_requires_complete_private_backup_before_once_gate',state
    for _ in range(10):
        now=identity(old['pid'])
        if now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['pid','startticks','boot_id']):break
        time.sleep(.5)
    else:raise AssertionError('controller terminal state written but controller remains active; do not archive live state')
    c=ready(True);root=B/'run';assert not (B/'SUMMARY.json').exists() and not (B/'FINALIZE_ONCE.lock').exists()
    terms={n:read(root/'execution'/n/'TERMINATION.json') for n in ['train','replay','output']}
    assert all(x['exit_code']==0 and x['resource_stop_reason'] is None for x in terms.values())
    assert terms['output']['cumulative']['wall']<=43200 and terms['output']['cumulative']['cpu']<=43200
    a,pa=journal.load(root/'train');b,pb=journal.load(root/'replay')
    assert a['epoch']==b['epoch']==64 and a['updates']==b['updates']==4736 and e.semantic(a)==e.semantic(b)
    for side,st,pr in [('train',a,pa),('replay',b,pb)]:
        out=root/side;v=read(out/'COMPLETE.json');assert v['epochs']==64 and v['updates']==4736 and not v['diagnostic_labels_read']
        assert v['final_state_sha256']==e.semantic(st)==read(out/'epoch064.json')['state_sha256']
        assert e.semantic(e.torch.load(out/'final_full.pt',map_location='cpu',weights_only=True))==e.semantic(st)
        m,o,g=e.make();e.restore(st,m,o,g);assert e.semantic(e.capture(m,o,g,64,4736,st['best_key'],st['best_model'],st['best_epoch']))==e.semantic(st)
        selected=e.torch.load(out/'selected.pt',map_location='cpu',weights_only=True);assert selected['selected_epoch']==st['best_epoch'] and e.semantic(selected['model'])==e.semantic(st['best_model'])
        raw=(out/'steps.jsonl').read_bytes();assert len(raw)==pr['prefix_bytes'] and hashlib.sha256(raw).hexdigest()==pr['prefix_sha256'] and len(raw.splitlines())==4736
    assert (root/'train/steps.jsonl').read_bytes()==(root/'replay/steps.jsonl').read_bytes()
    for ep in range(65):assert read(root/f'train/epoch{ep:03}.json')==read(root/f'replay/epoch{ep:03}.json')
    frozen=read(root/'PREDICTIONS_FROZEN.json');assert frozen['full_training_replayed'] and not frozen['diagnostic_labels_read']
    for n,h in frozen['files'].items():assert sha(root/n)==h
    release=read(B/'REAL_RELEASE.json');assert sha(B/'operator/finalize_once.py')==release['once_scorer_sha256']=='2fba1d84bae43a4f8ed893e43b8eb776b944a56830737998c2a6c426a6571324'
    q=dict(state='full64_producer_own_fresh64_replay_native_states_all65records_all4736steps_selected_output_audit_verified_unscored',observed_unix=time.time(),supervising_recovery_controller=old,original_failed_controller=read(B/'CONTROLLER_IDENTITY.json'),controller_inactive=True,producer_epochs=64,replay_epochs=64,updates_per_trajectory=4736,epoch_records_exact=65,native_final_sha256=e.semantic(a),canonical_terminal_digest_verified=True,full_optimizer_all_RNG_restore_recapture_exact=True,selected_epoch=a['best_epoch'],actual_terminations=terms,scientific_manifest_sha256=sha(B/'FROZEN_HASHES.json'),original_once_scorer_sha256=release['once_scorer_sha256'],diagnostic_labels_read=False,users7_8_labels_used=False,new_model_fit_started_by_observer=False,full_private_modelset_not_yet_uploaded=True)
    save('PILOT_AUDIT.json',q)
    # Include the actual recovered supervision source and failure/termination lineage.
    recovery=Path('/workspace/uwb-mixstyle-gap-meta-20261105');target=B/'operator/runtime_recovery';target.mkdir(exist_ok=False)
    import shutil
    names=['reattach_supervision.py','MONITOR_RECOVERY.json','RECOVERY_CONTROLLER_IDENTITY.json','RECOVERY_CONTROLLER_TERMINATION.json','REATTACH_LAUNCH.json','RECOVERY_CONTROLLER.log']
    for n in names:shutil.copyfile(recovery/n,target/n)
    for src in (recovery/'original_monitor_failure').rglob('*'):
        if src.is_file():
            dst=target/'original_monitor_failure'/src.relative_to(recovery/'original_monitor_failure');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    save('RECOVERY_EVIDENCE_BUNDLED.json',dict(files={str(f.relative_to(target)):sha(f) for f in target.rglob('*') if f.is_file()},scientific18_unchanged=True,once_scorer_unchanged=True))
    if not (B/'full_backup').exists():
        r=subprocess.run([sys.executable,str(B/'operator/full_archive.py'),'build'],text=True,capture_output=True)
        (M/'FULL_ARCHIVE_BUILD.log').write_text(r.stdout+r.stderr);save('FULL_ARCHIVE_BUILD_TERMINATION.json',dict(exit_code=r.returncode,new_fit_started=False,diagnostic_labels_read=False));assert r.returncode==0,r.stderr[-2000:]
    else:assert (B/'full_backup/MANIFEST.json').exists(),'partial archive exists; inspect, never blindly rebuild'
    print(json.dumps(dict(state='full_pilot_audited_and_complete_unscored_archive_built_not_uploaded_or_scored',archive_manifest=str(B/'full_backup/MANIFEST.json'))),flush=True)
if __name__=='__main__':
    try:main();save('OBSERVER_TERMINATION.json',dict(exit_code=0,new_fit_started=False,diagnostic_labels_read=False))
    except BaseException as exc:
        save('OBSERVER_TERMINATION.json',dict(exit_code=1,error=repr(exc),new_fit_started=False,diagnostic_labels_read=False));raise
