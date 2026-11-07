"""Read-only consistent committed training snapshot; never pause, mutate or resume production."""
from pathlib import Path
import sys,json,hashlib,io,time,shutil
P=Path('/workspace/uwb-irm-loso6-20261107');sys.path.insert(0,str(P/'shared'))
from common import read,write,sha
from controller import identity
import engine as e
from journal import load
from stream_archive import build,verify

def capture():
    w=time.monotonic();cpu=time.process_time();h=1
    phase=read(P/'PHASE_IDENTITY.json');now=identity(phase['pid'])
    assert now and now['state']!='Z' and all(now[k]==phase[k] for k in ['pid','boot_id','startticks'])
    assert read(P/'RESOURCE_LEDGER.json')['active_held']==h
    assert not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
    for n,v in read(P/'FROZEN_HASHES.json').items():assert sha(P/n)==v
    out=P/'folds/held1/run/train';dest=P/'progress_insurance/held1_train01';assert not dest.exists();dest.mkdir(parents=True);snap=dest/'snapshot/run/train';snap.mkdir(parents=True)
    for attempt in range(10):
        a=(out/'PROGRESS.json').read_bytes();p=json.loads(a);blob=(out/p['slot']).read_bytes();b=(out/'PROGRESS.json').read_bytes()
        if a==b and hashlib.sha256(blob).hexdigest()==p['checkpoint_sha256']:break
        time.sleep(.05)
    else:raise AssertionError('No stable committed pointer; evidence preserved, no live files changed')
    (snap/p['slot']).write_bytes(blob);(snap/'PROGRESS.json').write_bytes(a)
    raw=(out/'steps.jsonl').read_bytes();prefix=raw[:p['prefix_bytes']]
    assert len(prefix.splitlines())==p['updates'] and hashlib.sha256(prefix).hexdigest()==p['prefix_sha256']
    complete=raw if raw.endswith(b'\n') else raw[:raw.rfind(b'\n')+1]
    (snap/'steps.jsonl').write_bytes(complete)
    if len(complete)!=len(raw):(dest/'partial_last_line_evidence.bin').write_bytes(raw[len(complete):])
    for ep in range(p['epoch']+1):
        for name in [f'epoch{ep:03}.json',f'journal{ep:03}.json']:shutil.copyfile(out/name,snap/name)
    for name in ['initial.pt','first_update.pt','FIRST_READY.json','FIRST_RELEASE.json','FIRST_INDEPENDENT_REFERENCE.json']:shutil.copyfile(out/name,snap/name)
    state,q=load(snap);assert q==p
    for i,line in enumerate(complete.splitlines()):
        r=json.loads(line);assert r['update']==i+1 and r['completed_updates_before']==i and r['completed_updates_after']==i+1
        assert r['irm_penalty_weight']==(1. if i<500 else 100.) and r['optimizer_reset_this_update']==(i==500)
        assert r['optimizer_reset_events']==([] if i<500 else [500]) and r['classification_loss']==r['global_BCE']
        assert abs(r['loss']-(r['global_BCE']+r['irm_penalty_weight']*r['penalty']))<=1e-6*max(1.,abs(r['loss']))
    n=state['updates'];opt=state['optimizer'];assert opt['algorithm_updates']==n
    assert opt['reset_events']==([] if n<=500 else [500]);inner=opt['native_optimizer'];expected=n-(500 if n>500 else 0);assert inner['count']==expected
    assert all(float(v['step'])==expected for v in inner['fast']['state'].values())
    assert all(e.torch.isfinite(v).all() for v in state['model'].values())
    ledger=read(P/'RESOURCE_LEDGER.json')
    audit=dict(state='stable_native_training_checkpoint_and_entire_committed_journal_prefix_audited_without_pausing_live_job',held=h,epoch=p['epoch'],committed_updates=n,captured_complete_step_records=len(complete.splitlines()),uncommitted_suffix_records=len(complete.splitlines())-n,checkpoint_sha256=p['checkpoint_sha256'],native_state_sha256=e.semantic(state),selected_epoch_validation_only=state['best_epoch'],actual_native_optimizer_steps=expected,actual_reset_events=opt['reset_events'],all_captured_step_schedule_loss_and_counter_checks_passed=True,original_phase_identity=phase,actual_original_phase=now,cost_observation_after_capture=ledger,prior_and_unknown_interruption_costs_must_be_preserved=True,complete_trajectory=False,complete_sixfold_modelset=False,automatic_restart_authorized=False,labels_or_UWB_maps_opened_by_snapshot=False,new_models_trained_or_inference_run_by_snapshot=False,snapshot_capture_wall=time.monotonic()-w,snapshot_capture_CPU=time.process_time()-cpu)
    write(dest/'CAPTURE_AUDIT.json',audit)
    files={str(f.relative_to(dest)):f for f in (dest/'snapshot').rglob('*') if f.is_file()}
    files['CAPTURE_AUDIT.json']=dest/'CAPTURE_AUDIT.json'
    for n in [*read(P/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','REAL_RELEASE.json','INITIAL_ASSET.json','INITIAL_COMPOSITE_RECOVERY_INDEX.json']:
        files['source/phase/'+n]=P/n
    files['private_IDs/source_ids.npz']=Path(e.c['ids']);files['private_IDs/held1_roles.npz']=P/'folds/held1/roles.npz'
    m=build(dest/'archive','irm_loso6_held1_train_prefix01_20261107',files,'IRM_held1_training_committed_native_checkpoint_only_not_complete_fold_or_cohort',chunk_size=7340032,limit=268435456)
    m.update(complete_modelset=False,complete_training_trajectory=False,complete_sixfold_cohort=False,restorable_single_native_checkpoint=True,checkpoint_epoch=p['epoch'],checkpoint_updates=state['updates'],scope_costs_not_reset_on_restore=True)
    write(dest/'archive/MANIFEST.json',m);print(json.dumps(dict(archive_parts=len(m['chunks']),archive_bytes=m['archive_bytes'],payloads=m['tracked_payload_files'],epoch=p['epoch'],updates=state['updates'],state='live_job_unpaused_snapshot_built_unverified_private_upload')))

def check():
    d=P/'progress_insurance/held1_train01/archive';m=read(d/'MANIFEST.json');rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
    for r in rows:
        v=r['metadata'];assert v['shared'] is False and v['permissions']==[dict(type='user',role='owner')] and v['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf'];assert r['raw_fetch'] and r['download_verified'] and r['include_base64'] is False
    w=time.monotonic();cpu=time.process_time();v=verify(m,rows);write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
    write(d/'READBACK_COST.json',dict(wall=time.monotonic()-w,CPU=time.process_time()-cpu,scope='whole_archive_raw_allmember_verification'))
    receipt=dict(state='single_current_native_checkpoint_private_raw_allmember_verified_not_completed_training',archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],scope=m['scope'],complete_training_trajectory=False,complete_sixfold_cohort=False,restorable_single_native_checkpoint=True,automatic_restart_authorized=False,checkpoint_epoch=m['checkpoint_epoch'],checkpoint_updates=m['checkpoint_updates'],chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows]);write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**receipt));print(json.dumps(v))

if __name__=='__main__':
    assert sys.argv[1:] in [['build'],['verify']]
    capture() if sys.argv[1]=='build' else check()
