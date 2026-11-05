"""Complete original producer plus coherent OWN replay prefix; never fit or score."""
from pathlib import Path
import sys,json,hashlib,shutil,time
S=Path('/workspace/uwb-cutmix-temporal-gap64-20261105');D=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
from common import read,write,sha,ready
from controller import identity
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
def create():
    c=ready(True);controller=read(S/'CONTROLLER_IDENTITY.json');worker=read(S/'run/execution/replay/IDENTITY.json')
    for old in [controller,worker]:
        now=identity(old['pid']);assert now and now['state']!='Z' and all(now[k]==old[k] for k in ['pid','boot_id','startticks'])
    assert not (S/'SUMMARY.json').exists() and not (S/'FINALIZE_ONCE.lock').exists()
    term=read(S/'run/execution/train/TERMINATION.json');assert term['exit_code']==0 and term['resource_stop_reason'] is None
    assert read(S/'run/train/COMPLETE.json')['epochs']==64 and not (S/'run/replay/COMPLETE.json').exists()
    p=read(S/'run/replay/PROGRESS.json');raw=(S/'run/replay'/p['slot']).read_bytes();assert hashlib.sha256(raw).hexdigest()==p['checkpoint_sha256']
    log=(S/'run/replay/steps.jsonl').read_bytes();assert log.endswith(b'\n');prefix=log[:p['prefix_bytes']];assert len(prefix.splitlines())==p['updates'] and hashlib.sha256(prefix).hexdigest()==p['prefix_sha256']
    rows=[json.loads(x) for x in log.splitlines()];assert all(x['update']==i+1 for i,x in enumerate(rows))
    out=D/'capture01';out.mkdir(exist_ok=False);overlay=out/'study';train=overlay/'run/train';replay=overlay/'run/replay';replay.mkdir(parents=True)
    for src in (S/'run/train').rglob('*'):
        if src.is_file() and not src.name.endswith(('.lock','.tmp')):copy(src,train/src.relative_to(S/'run/train'))
    (replay/p['slot']).write_bytes(raw);(replay/'steps.jsonl').write_bytes(log);write(replay/'PROGRESS.json',p)
    for ep in range(p['epoch']+1):
        for n in [f'epoch{ep:03}.json',f'journal{ep:03}.json']:copy(S/'run/replay'/n,replay/n)
    for n in ['initial.pt','split.npz']:copy(S/'run/replay'/n,replay/n)
    for n in [*read(S/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','REAL_RELEASE.json','INITIAL_ASSET.json','initial.pt','roles.npz','SYNTHETIC_RESULT.json','RESOURCE_RESULT.json','CONTROLLER_IDENTITY.json','LAUNCH_STARTED.json','LAUNCH_OBSERVATION.json','ACTIVATION.json','AUTHOR_SCOPE_CORRECTION.json','OPERATIONAL_ARCHIVE_REPAIR.json','RESTORE_GUIDE.md']:copy(S/n,overlay/n)
    for sub in ['operator','run/execution','initial_backup']:
        for src in (S/sub).rglob('*'):
            if src.is_file() and '__pycache__' not in src.parts and not src.name.endswith(('.lock','.tmp')):copy(src,overlay/sub/src.relative_to(S/sub))
    ledger=read(S/'RESOURCE_LEDGER.json');assert ledger['stage']=='replay' and ledger['wall']<43200 and ledger['cpu']<43200 and ledger['RSS_KiB']<12582912;write(overlay/'RESOURCE_LEDGER.json',ledger)
    for n,h in read(S/'DEPENDENCIES.json')['files'].items():assert sha(n)==h;copy(Path(n),out/'native_source'/Path(n).relative_to(c['native_source']))
    for n,h in c['author_reference_files'].items():assert sha(n)==h;copy(Path(n),out/'author_review'/Path(n).name)
    prior=Path(c['prior_RegMixup_summary']).parent
    for src,dst in [('SUMMARY.json','prior_RegMixup_SUMMARY.json'),('STATE.json','prior_RegMixup_POST_SCORE_STATE.json'),('FINALIZE_ONCE.lock','prior_RegMixup_FINALIZE_ONCE.lock')]:copy(prior/src,out/'lineage'/dst)
    copy(Path(c['ids']),out/'private_IDs/source_ids.npz');copy(Path(c['source1_reference_summary']),out/'lineage/noMixup_reference_SUMMARY.json');copy(S/'initial_backup/CLOUD_DOWNLOAD_INDEX.json',out/'lineage/INITIAL_ARCHIVE_INDEX.json')
    write(out/'PAYLOAD_HASHES.json',{str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in out.rglob('*') if f.is_file()})
    q=dict(state='complete_temporal_CutMix_producer64_and_coherent_OWN_fresh_replay_native_prefix_captured_unscored',observed_unix=time.time(),controller=identity(controller['pid']),replay_worker=identity(worker['pid']),producer_complete=read(S/'run/train/COMPLETE.json'),producer_actual_termination=term,replay_progress=p,replay_captured_log_rows=len(rows),replay_uncommitted_rows=len(rows)-p['updates'],replay_suffix_bytes=len(log)-p['prefix_bytes'],resource_ledger=ledger,diagnostic_labels_read=False,users7_8_labels_used=False,new_experiment_or_duplicate_fit_started=False,scientific20_changed=False,budget_reset=False)
    write(D/'OBSERVATION.json',q);print(json.dumps(q))
def native():
    import engine as e
    from journal import load
    O=D/'capture01';B=O/'study'
    for n,r in read(O/'PAYLOAD_HASHES.json').items():assert sha(O/n)==r['sha256'] and (O/n).stat().st_size==r['bytes']
    states={};progress={}
    for side in ['train','replay']:
        st,p=load(B/'run'/side);states[side]=st;progress[side]=p;m,o,g=e.make();e.restore(st,m,o,g)
        assert e.semantic(e.capture(m,o,g,st['epoch'],st['updates'],st['best_key'],st['best_model'],st['best_epoch']))==e.semantic(st)
    st=states['train'];complete=read(B/'run/train/COMPLETE.json');assert st['epoch']==64 and st['updates']==4736
    assert e.semantic(st)==complete['final_state_sha256']==read(B/'run/train/epoch064.json')['state_sha256']==e.semantic(e.torch.load(B/'run/train/final_full.pt',map_location='cpu',weights_only=True))
    selected=e.torch.load(B/'run/train/selected.pt',map_location='cpu',weights_only=True);assert selected['selected_epoch']==st['best_epoch'] and e.semantic(selected['model'])==e.semantic(st['best_model'])
    allrows=(B/'run/train/steps.jsonl').read_bytes();p=progress['train'];assert len(allrows)==p['prefix_bytes'] and hashlib.sha256(allrows).hexdigest()==p['prefix_sha256'] and len(allrows.splitlines())==4736
    p=progress['replay'];raw=(B/'run/replay/steps.jsonl').read_bytes();assert raw[:p['prefix_bytes']]==allrows[:p['prefix_bytes']]
    refrows=allrows.splitlines();reprows=raw.splitlines();assert all(json.loads(x)==json.loads(refrows[i]) for i,x in enumerate(reprows))
    for ep in range(p['epoch']+1):assert read(B/f'run/train/epoch{ep:03}.json')==read(B/f'run/replay/epoch{ep:03}.json')
    assert e.semantic(states['replay'])==read(B/f'run/train/epoch{p["epoch"]:03}.json')['state_sha256']
    first=read(B/'run/train/FIRST_RELEASE.json');assert first['first_sha256']==sha(B/'run/train/first_update.pt') and first['native_first_update_exact']
    q=dict(state='pass_complete_producer64_and_OWN_partial_fresh_replay_native_insurance_unscored',producer_epochs=64,producer_updates=4736,producer_canonical_final_sha256=e.semantic(st),producer_complete_digest_not_selected_alias=True,selected_epoch=st['best_epoch'],native_full_optimizer_all_RNG_restore_recapture_both_lanes_exact=True,replay_epoch=p['epoch'],replay_committed_updates=p['updates'],replay_all_captured_uncommitted_rows_equal_producer=True,payload_files_verified=len(read(O/'PAYLOAD_HASHES.json')),full64_fresh_replay_complete=False,diagnostic_labels_read=False,users7_8_labels_used=False,no_forward_or_optimizer_update=True)
    write(D/'NATIVE_SNAPSHOT_VERIFIED.json',q);print(json.dumps(q))
def archive():
    from stream_archive import build
    assert read(D/'NATIVE_SNAPSHOT_VERIFIED.json')['state'].startswith('pass_')
    files={'capture/'+str(p.relative_to(D/'capture01')):p for p in (D/'capture01').rglob('*') if p.is_file()}
    for n in ['snapshot.py','RESTORE_GUIDE.md','OBSERVATION.json','NATIVE_SNAPSHOT_VERIFIED.json']:files['insurance/'+n]=D/n
    m=build(D/'private_backup','temporal_cutmix_gap64_producer64_replay_insurance01_20261105',files,'complete_temporal_CutMix_producer64_plus_OWN_partial_fresh_replay_not_final_pilot_or_scoring_authority',chunk_size=7340032,limit=268435456)
    m.update(state='complete_producer64_and_own_partial_fresh_replay_streamed_to_chunks_not_final_modelset',complete_modelset=False,producer64_complete=True,full64_fresh_replay_complete=False,initial_guard_raw_chunks_bundled=True,final_scoring_authority=False);m.pop('shared_initial_encoder_included_once');write(D/'private_backup/MANIFEST.json',m);return m
def verify_private():
    from stream_archive import verify
    d=D/'private_backup';m=read(d/'MANIFEST.json');rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
    for r in rows:
        z=r['metadata'];assert not z['shared'] and z['permissions']==[dict(role='owner',type='user')] and z['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf'];assert r['raw_fetch'] and r['download_verified'] and not r['include_base64']
    v=verify(m,rows);write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
    receipt=dict(state='complete_producer64_and_partial_own_replay_owner_only_raw_every_member_verified_not_final_modelset',scope=m['scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],producer64_complete=True,complete_pilot_modelset=False,final_scoring_authority=False,initial_guard_raw_chunks_bundled=True,raw_data_maps_runtime_external=True,diagnostic_labels_read=False,chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows])
    write(d/'DRIVE_BACKUP_RECEIPT.json',receipt);write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**receipt));return v
if __name__=='__main__':
    if sys.argv[1]=='capture':create()
    elif sys.argv[1]=='native':native()
    else:print(json.dumps(archive() if sys.argv[1]=='archive' else verify_private()))
