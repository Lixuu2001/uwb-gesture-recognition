"""Protect one matched-model untrained IRM state and all6 private roles before fitting."""
from pathlib import Path
import sys,shutil,json
P=Path('/workspace/uwb-irm-loso6-20261107');sys.path.insert(0,str(P/'shared'))
from common import ready,read,write,sha
from stream_archive import build,verify

def create():
    c=ready();registration=read(P/'REMOTE_PREREGISTRATION.json');assert registration['branch_head_verified'] and registration['force'] is False and registration['frozen_sha256']==sha(P/'FROZEN_HASHES.json')
    assert read(P/'SYNTHETIC_SUMMARY.json')['all_passed'] and read(P/'RESOURCE_GATE.json')['gate_passed']
    assert not(P/'PHASE_IDENTITY.json').exists() and not(P/'initial.pt').exists()
    import engine as e
    m,o,g=e.make();st=e.capture(m,o,g,0,0)
    assert e.semantic(st['model'])=='2aae73c58e38ca324027e888955b130a82fbaece7167df3465522ec51986d2d3'
    assert st['optimizer']['algorithm_updates']==0 and st['optimizer']['reset_events']==[] and o.count==0 and not o.slow
    e.atomic_save(P/'initial.pt',st)
    from loso_roles_metrics import roles
    with e.np.load(c['ids']) as z:ids=z['ids'];users=z['users']
    assert sha(c['ids'])==c['ids_sha256']
    files={'phase/'+n:P/n for n in [*read(P/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','SYNTHETIC_SUMMARY.json','RESOURCE_GATE.json','initial.pt']}
    for h in range(1,7):
        fold=P/f'folds/held{h}';cfg=read(fold/'PROTOCOL.json');assert cfg['seed']==20261029 and cfg['native_source']==c['native_source']
        shutil.copyfile(P/'initial.pt',fold/'initial.pt');_,loc=roles(users,h);e.np.savez_compressed(fold/'roles.npz',**{k:ids[ix] for k,ix in loc.items()})
        files[f'phase/folds/held{h}/roles.npz']=fold/'roles.npz'
        for sub in ['SYNTHETIC_RESULT.json','synthetic_execution/TERMINATION.json','synthetic_execution/stdout.log']:files[f'phase/folds/held{h}/'+sub]=fold/sub
    for n in read(P/'shared/DEPENDENCIES.json')['files']:files['native_source/'+str(Path(n).relative_to(c['native_source']))]=Path(n)
    files['private_IDs/source_ids.npz']=Path(c['ids'])
    q=dict(initial_model_file_sha256=sha(P/'initial.pt'),initial_native_state_sha256=e.semantic(st),initial_model_native_sha256=e.semantic(st['model']),all_six_initial_states_identical=True,private_roles_for_all_six_folds=True,new_UWB_fit_started=False,initial_optimizer_is_new_IRM_wrapper_not_MMD_optimizer=True)
    write(P/'INITIAL_ASSET.json',q);files['phase/INITIAL_ASSET.json']=P/'INITIAL_ASSET.json'
    result=build(P/'initial_backup','irm_native_loso6_initial_20261107',files,'fixed_full_sixfold_IRM_initial_state_and_all_private_roles_before_any_fit',chunk_size=7340032,limit=67108864)
    result.update(state='untrained_initial_native_state_roles_sources_streamed',complete_modelset=False,complete_initial_model=True,complete_training_modelset=False)
    write(P/'initial_backup/MANIFEST.json',result);return result

def check():
    d=P/'initial_backup';m=read(d/'MANIFEST.json');rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
    for r in rows:
        q=r['metadata'];assert not q['shared'] and q['permissions']==[dict(type='user',role='owner')] and q['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
        assert r['raw_fetch'] and r['download_verified'] and not r['include_base64']
    v=verify(m,rows);write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
    receipt=dict(state='all_six_initial_native_roles_sources_owner_only_raw_allmember_verified',archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],scope=m['scope'],complete_untrained_initial_state=True,complete_trained_modelset=False,chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows])
    write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',raw_fetch_options=dict(download_raw_file=True,include_base64=False),max_concurrency=3,**receipt));return v
if __name__=='__main__':print(json.dumps(create() if sys.argv[1]=='build' else check()))
