"""Read-only insurance of one completed unscored fold; never train, infer or score."""
from pathlib import Path
import sys,json,time,math
P=Path('/workspace/uwb-irm-loso6-20261107');sys.path.insert(0,str(P/'shared'))
from common import ready,read,write,sha
from controller import identity
from journal import load
from stream_archive import build,verify
import engine as e

def create(h):
    started=time.monotonic();cpu=time.process_time();ready(False)
    assert h in range(1,7) and not(P/'SUMMARY.json').exists() and not(P/'FINALIZE_ONCE.lock').exists()
    fold=P/f'folds/held{h}';c=read(fold/'PROTOCOL.json');root=fold/'run'
    assert read(fold/'STATE.json')['state']=='pilot_audited_frozen_unscored_requires_complete_private_backup_before_once_gate'
    assert h in read(P/'STATE.json')['completed_folds']
    for name in ['CONTROLLER_IDENTITY.json','LAUNCH_IDENTITY.json']:
        old=read(fold/name);now=identity(old['pid'])
        assert now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['boot_id','startticks'])
    for q in (root/'execution').rglob('*IDENTITY.json'):
        old=read(q);now=identity(old['pid'])
        assert now is None or now['state']=='Z' or any(now[k]!=old[k] for k in ['boot_id','startticks'])
    for side in ['train','replay','output']:
        term=read(root/f'execution/{side}/TERMINATION.json');assert term['exit_code']==0 and term['resource_stop_reason'] is None
    assert read(fold/'PHASE_TERMINATION.json')['exit_code']==0
    frozen=read(root/'PREDICTIONS_FROZEN.json');assert frozen['epochs']==64 and frozen['updates']==c['expected_updates'] and frozen['full_training_replayed'] and not frozen['diagnostic_labels_read']
    for n,v in frozen['files'].items():assert sha(root/n)==v
    completed=[read(root/f'{side}/COMPLETE.json') for side in ['train','replay']]
    for r in completed:assert r['epochs']==64 and r['updates']==c['expected_updates'] and not r['diagnostic_labels_read'] and not r['users7_8_labels_used']
    for k in completed[0]:
        if k!='state':assert completed[0][k]==completed[1][k],k
    for n in ['initial.pt','first_update.pt','selected.pt','final_full.pt','validation_unscored.npz','diagnostic_unscored.npz','steps.jsonl']:
        # replay reuses the exact initial/first checkpoint and independently constructs its trajectory.
        if (root/'replay'/n).exists():assert sha(root/'train'/n)==sha(root/'replay'/n),n
    for epoch in range(65):
        for kind in ['epoch','journal']:assert sha(root/f'train/{kind}{epoch:03}.json')==sha(root/f'replay/{kind}{epoch:03}.json')
    states=[]
    for side in ['train','replay']:
        s,p=load(root/side);states.append(s)
        assert p['epoch']==64 and s['updates']==c['expected_updates'] and e.semantic(s)==completed[0]['final_state_sha256']
        raw=(root/side/'steps.jsonl').read_bytes();assert len(raw)==p['prefix_bytes'] and sha(root/side/'steps.jsonl')==p['prefix_sha256']
        rows=[json.loads(x) for x in raw.splitlines()];assert len(rows)==c['expected_updates']
        for i,r in enumerate(rows):
            assert r['update']==i+1 and r['completed_updates_before']==i and r['completed_updates_after']==i+1
            assert r['irm_penalty_weight']==(1. if i<500 else 100.) and r['optimizer_reset_this_update']==(i==500) and r['optimizer_reset_events']==([] if i<500 else [500])
            assert r['global_BCE']==r['classification_loss'] and all(math.isfinite(r[k]) for k in ['global_BCE','penalty','loss'])
            assert abs(r['loss']-(r['global_BCE']+r['irm_penalty_weight']*r['penalty']))<=1e-6*max(1.,abs(r['loss']))
        o=s['optimizer'];native=o['native_optimizer'];assert o['algorithm_updates']==c['expected_updates'] and o['reset_events']==[500]
        assert native['count']==c['expected_updates']-500 and all(float(v['step'])==native['count'] for v in native['fast']['state'].values())
        assert all(e.torch.isfinite(v).all() for v in s['model'].values())
    ref=read(root/'train/FIRST_INDEPENDENT_REFERENCE.json');assert ref['parameters']==617612 and ref['training_labels_only'] and not ref['diagnostic_labels_read']
    assert read(root/'train/FIRST_RELEASE.json')['native_first_update_exact']
    dest=P/f'completed_fold_insurance/held{h}';assert not dest.exists();dest.mkdir(parents=True)
    audit=dict(state='one_complete_fold_full64_fresh64_native_journals_and_unscored_outputs_verified',held=h,epochs_per_trajectory=64,updates_per_trajectory=c['expected_updates'],epoch_records_compared=65,all_step_records_compared=c['expected_updates'],first_independent_reference=ref,selected_epoch_validation_only=completed[0]['selected_epoch'],selected_sha256=completed[0]['selected_sha256'],final_native_sha256=completed[0]['final_state_sha256'],training_and_replay_native_optimizer_steps=c['expected_updates']-500,reset_events=[500],complete_trajectory_for_this_fold=True,complete_all_six=False,diagnostic_labels_read=False,users7_8_labels_used=False,new_training_inference_or_scoring_by_this_operation=False,fold_original_cost=read(fold/'STATE.json'),phase_current_cost_observation=read(P/'RESOURCE_LEDGER.json'),audit_wall=time.monotonic()-started,audit_CPU=time.process_time()-cpu)
    write(dest/'FOLD_AUDIT.json',audit)
    files={'source/phase/'+n:P/n for n in [*read(P/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','REAL_RELEASE.json','INITIAL_ASSET.json','INITIAL_COMPOSITE_RECOVERY_INDEX.json']}
    for f in fold.rglob('*'):
        if f.is_file() and '__pycache__' not in f.parts and not f.name.endswith(('.tmp','.lock')):files['completed/'+str(f.relative_to(P))]=f
    for n in read(P/'shared/DEPENDENCIES.json')['files']:files['native_source/'+str(Path(n).relative_to(c['native_source']))]=Path(n)
    files['private_IDs/source_ids.npz']=Path(c['ids']);files['FOLD_AUDIT.json']=dest/'FOLD_AUDIT.json'
    files['operational/archive_completed_fold.py']=Path(__file__)
    m=build(dest/'archive',f'irm_native_loso6_held{h}_complete_20261107',files,f'IRM_LOSO6_held{h}_complete_native_production_and_fresh_replay_unscored_single_fold_only',chunk_size=7340032,limit=268435456)
    m.update(state='one_complete_fold_streamed_not_complete_sixfold_modelset',complete_modelset=True,modelset_scope=f'held{h} only; not all six',held=h,complete_trajectory=True,complete_all_six=False,automatic_restart_authorized=False)
    write(dest/'archive/MANIFEST.json',m)
    write(dest/'BUILD_COST.json',dict(wall=time.monotonic()-started,CPU=time.process_time()-cpu,includes_audit=True,overlaps_live_phase_wall=True,network_cost_not_measured=True))
    return dict(base=str(dest/'archive'),chunks=len(m['chunks']),archive_bytes=m['archive_bytes'],payloads=m['tracked_payload_files'],archive_sha256=m['archive_sha256'],audit=audit)

def check(h):
    started=time.monotonic();cpu=time.process_time();d=P/f'completed_fold_insurance/held{h}/archive';m=read(d/'MANIFEST.json');rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
    for r in rows:
        q=r['metadata'];assert not q['shared'] and q['permissions']==[dict(type='user',role='owner')] and q['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
        assert r['raw_fetch'] and r['download_verified'] and r['include_base64'] is False
    v=verify(m,rows);write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
    write(d/'READBACK_COST.json',dict(wall=time.monotonic()-started,CPU=time.process_time()-cpu,overlaps_live_phase_wall=True,scope='single_complete_fold_raw_allmember_verification'))
    receipt=dict(state='one_complete_IRM_fold_raw_allmember_verified_not_all_six',held=h,complete_all_six=False,complete_this_fold_trajectory=True,automatic_restart_authorized=False,archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],scope=m['scope'],chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows])
    write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',raw_fetch_options=dict(download_raw_file=True,include_base64=False),max_concurrency=3,**receipt));return v

if __name__=='__main__':
    mode,h=sys.argv[1:];h=int(h);assert mode in ['build','verify'];print(json.dumps(create(h) if mode=='build' else check(h)))
