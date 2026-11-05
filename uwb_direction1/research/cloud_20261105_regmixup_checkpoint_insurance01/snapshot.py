"""Coherent committed native RegMixup/GAP state and append-only suffix insurance."""
from pathlib import Path
import sys,json,hashlib,shutil,time
S=Path('/workspace/uwb-regmixup-gap64-20261105');D=Path(__file__).resolve().parent
sys.path.insert(0,str(S))
from common import read,write,sha,ready
from controller import identity
def digest(raw):return hashlib.sha256(raw).hexdigest()
def create():
    c=ready(True);controller=read(S/'CONTROLLER_IDENTITY.json');worker=read(S/'run/execution/train/IDENTITY.json')
    for q in [controller,worker]:
        a=identity(q['pid']);assert a and a['state']!='Z' and a['boot_id']==q['boot_id'] and a['startticks']==q['startticks']
    assert not (S/'run/train/COMPLETE.json').exists(),'Choose complete-producer preservation after completion'
    p=read(S/'run/train/PROGRESS.json');raw=(S/'run/train'/p['slot']).read_bytes();assert digest(raw)==p['checkpoint_sha256']
    log=(S/'run/train/steps.jsonl').read_bytes();assert log.endswith(b'\n')
    prefix=log[:p['prefix_bytes']];assert digest(prefix)==p['prefix_sha256'] and len(prefix.splitlines())==p['updates']
    rows=[json.loads(x) for x in log.splitlines()];assert all(x['update']==i+1 for i,x in enumerate(rows))
    out=D/'capture01';out.mkdir(exist_ok=False);overlay=out/'study';t=overlay/'run/train';t.mkdir(parents=True)
    (t/p['slot']).write_bytes(raw);(t/'steps.jsonl').write_bytes(log);write(t/'PROGRESS.json',p)
    for epoch in range(p['epoch']+1):
        for n in [f'epoch{epoch:03}.json',f'journal{epoch:03}.json']:shutil.copyfile(S/'run/train'/n,t/n)
    for n in ['initial.pt','first_update.pt','FIRST_READY.json','FIRST_RELEASE.json','split.npz']:shutil.copyfile(S/'run/train'/n,t/n)
    for n in [*read(S/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','REAL_RELEASE.json','INITIAL_ASSET.json','initial.pt','roles.npz','SYNTHETIC_RESULT.json','RESOURCE_RESULT.json','CONTROLLER_IDENTITY.json','LAUNCH_STARTED.json','LAUNCH_OBSERVATION.json','INITIAL_VERIFY_DISPLAY_CORRECTION.json']:
        f=overlay/n;f.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(S/n,f)
    for sub in ['operator','run/execution','initial_backup']:
        for src in (S/sub).rglob('*'):
            if src.is_file() and '__pycache__' not in src.parts and not src.name.endswith(('.lock','.tmp')):
                dst=overlay/sub/src.relative_to(S/sub);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    ledger=read(S/'RESOURCE_LEDGER.json');assert ledger['stage']=='train' and ledger['wall']<c['real_wall_cap'] and ledger['cpu']<c['real_CPU_cap']
    write(overlay/'RESOURCE_LEDGER.json',ledger)
    native=out/'native_source'
    for n,h in read(S/'DEPENDENCIES.json')['files'].items():
        assert sha(n)==h;dst=native/Path(n).relative_to(c['native_source']);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(n,dst)
    author=out/'author_review';author.mkdir()
    for ref,h in c['author_reference_files'].items():
        assert sha(ref)==h;shutil.copyfile(ref,author/Path(ref).name)
    private=out/'private_IDs';private.mkdir();shutil.copyfile(c['ids'],private/'source_ids.npz')
    lineage=out/'lineage';lineage.mkdir();shutil.copyfile(c['source1_reference_summary'],lineage/'noMixup_reference_SUMMARY.json')
    shutil.copyfile(S/'initial_backup/CLOUD_DOWNLOAD_INDEX.json',lineage/'INITIAL_ARCHIVE_INDEX.json')
    table={str(f.relative_to(out)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in out.rglob('*') if f.is_file()}
    write(out/'PAYLOAD_HASHES.json',table)
    observation=dict(state='coherent_committed_RegMixup_GAP_native_checkpoint_and_uncommitted_suffix_captured',observed_unix=time.time(),controller=identity(controller['pid']),worker=identity(worker['pid']),progress=p,total_log_rows=len(rows),uncommitted_log_rows=len(rows)-p['updates'],suffix_bytes=len(log)-p['prefix_bytes'],resource_ledger=ledger,diagnostic_labels_read=False,users7_8_labels_used=False,new_fit_started_by_capture=False,scientific_configuration_changed=False,budget_reset=False,live_source_modified=False)
    write(D/'OBSERVATION.json',observation);print(json.dumps(observation))
def verify_capture():
    out=D/'capture01';overlay=out/'study'
    for n,row in read(out/'PAYLOAD_HASHES.json').items():assert sha(out/n)==row['sha256'] and (out/n).stat().st_size==row['bytes'],n
    # Load the captured journal independently, never the live mutable checkpoint.
    import engine as e
    from journal import load
    state,p=load(overlay/'run/train');m,o,g=e.make();e.restore(state,m,o,g)
    restored=e.capture(m,o,g,state['epoch'],state['updates'],state['best_key'],state['best_model'],state['best_epoch'])
    assert e.semantic(restored)==e.semantic(state)
    first=e.torch.load(overlay/'run/train/first_update.pt',map_location='cpu',weights_only=True)
    release=read(overlay/'run/train/FIRST_RELEASE.json');assert e.semantic(first)==release['state_sha256']
    assert sha(overlay/'run/train/first_update.pt')==release['first_sha256']
    record=read(overlay/f'run/train/epoch{state["best_epoch"]:03}.json')
    assert e.semantic(state['best_model'])==record['model_sha256']
    candidates=[read(overlay/f'run/train/epoch{x:03}.json') for x in range(1,state['epoch']+1)]
    chosen=max(candidates,key=lambda r:(r['validation_macro'],-r['validation_loss'],-r['epoch']))
    assert chosen['epoch']==state['best_epoch']
    result=dict(state='pass',native_model_optimizer_all_RNG_restore_recapture_exact=True,full_native_backbone_parameters=sum(p.numel() for p in m.parameters()),checkpoint_sha256=p['checkpoint_sha256'],state_sha256=e.semantic(state),committed_epoch=state['epoch'],committed_updates=state['updates'],selected_epoch=state['best_epoch'],best_inner_validation_selection_verified=True,first_real_update_release_verified=True,journal_and_committed_prefix_verified=True,payload_files_verified=len(read(out/'PAYLOAD_HASHES.json')),diagnostic_labels_read=False,users7_8_labels_used=False,full64_training_replayed=False,no_forward_or_optimizer_update=True)
    write(D/'NATIVE_SNAPSHOT_VERIFIED.json',result);print(json.dumps(result))
def archive():
    assert read(D/'NATIVE_SNAPSHOT_VERIFIED.json')['state']=='pass'
    from stream_archive import build
    files={'capture/'+str(f.relative_to(D/'capture01')):f for f in (D/'capture01').rglob('*') if f.is_file()}
    for n in ['snapshot.py','OBSERVATION.json','NATIVE_SNAPSHOT_VERIFIED.json','RESTORE_GUIDE.md']:files['insurance/'+n]=D/n
    m=build(D/'private_backup','regmixup_gap64_checkpoint_insurance01_20261105',files,'intermediate_RegMixup_GAP64_producer_native_checkpoint_not_complete_pilot_or_scoring_authority',chunk_size=7340032,limit=134217728)
    m.update(state='intermediate_RegMixup_native_checkpoint_and_initial_insurance_streamed',complete_modelset=False,final_scoring_authority=False,full64_training_complete=False,initial_private_guard_chunks_included=True);m.pop('shared_initial_encoder_included_once');write(D/'private_backup/MANIFEST.json',m)
    return m
def verify_private():
    from stream_archive import verify
    d=D/'private_backup';m=read(d/'MANIFEST.json');rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
    for r in rows:
        z=r['metadata'];assert not z['shared'] and z['permissions']==[dict(role='owner',type='user')] and z['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
        assert r['raw_fetch'] and r['download_verified'] and not r['include_base64']
    v=verify(m,rows);write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
    receipt=dict(state='intermediate_native_checkpoint_owner_only_raw_allmember_verified',scope=m['scope'],archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],complete_pilot_modelset=False,final_scoring_authority=False,diagnostic_labels_read=False,raw_data_maps_runtime_external=True,chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows])
    write(d/'DRIVE_BACKUP_RECEIPT.json',receipt);write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**receipt));return v
if __name__=='__main__':
    if sys.argv[1]=='capture':create()
    elif sys.argv[1]=='native':verify_capture()
    else:print(json.dumps(archive() if sys.argv[1]=='archive' else verify_private()))
