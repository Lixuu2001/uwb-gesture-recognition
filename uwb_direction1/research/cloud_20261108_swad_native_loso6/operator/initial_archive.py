"""Protect six role-specific untrained native/SWAD states before any actual fitting."""
from pathlib import Path
import sys,json,subprocess,os
P=Path('/workspace/uwb-swad-loso6-20261108');sys.path.insert(0,str(P/'shared'))
from common import ready,read,write,sha
from stream_archive import build,verify

def create():
    c=ready();r=read(P/'REMOTE_PREREGISTRATION.json');assert r['branch_head_verified'] and r['force'] is False and r['frozen_sha256']==sha(P/'FROZEN_HASHES.json')
    assert read(P/'SYNTHETIC_SUMMARY.json')['all_passed'] and read(P/'RESOURCE_GATE.json')['gate_passed']
    assert not(P/'PHASE_IDENTITY.json').exists() and not(P/'INITIAL_ASSET.json').exists()
    # Binary hash checks do not decode held labels. Only source IDs/subjects opened below.
    assert sha(c['ids'])==c['ids_sha256'] and sha(c['maps'])==c['maps_sha256']
    for n,h in c['input_sha256'].items():assert sha(Path(c['data'])/'cache'/(n+'.npy'))==h
    import numpy as np
    from loso_roles_metrics import roles
    with np.load(c['ids']) as z:ids=z['ids'];users=z['users']
    subj=np.load(Path(c['data'])/'cache/subj_clutterremoved.npy',mmap_mode='r');np.testing.assert_array_equal(subj[ids],users)
    files={'phase/'+n:P/n for n in [*read(P/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','SYNTHETIC_SUMMARY.json','RESOURCE_GATE.json']};states={}
    for h in range(1,7):
        fold=P/f'folds/held{h}';assert not(fold/'initial.pt').exists() and not(fold/'roles.npz').exists()
        env=dict(os.environ,UWB_SWAD_LOSO_FOLD_ROOT=str(fold));subprocess.run([sys.executable,__file__,'state'],env=env,check=True)
        states[str(h)]=read(fold/'INITIAL_STATE_METADATA.json');_,loc=roles(users,h)
        np.savez_compressed(fold/'roles.npz',**{k:ids[ix] for k,ix in loc.items()})
        for n in ['initial.pt','roles.npz','INITIAL_STATE_METADATA.json','SYNTHETIC_RESULT.json','synthetic_execution/TERMINATION.json','synthetic_execution/stdout.log']:files[f'phase/folds/held{h}/'+n]=fold/n
    assert len({x['model_native_sha256'] for x in states.values()})==1 and len({x['full_native_sha256'] for x in states.values()})==1
    q=dict(initial_model_file_sha256_by_held={h:x['file_sha256'] for h,x in states.items()},initial_native_state_sha256_by_held={h:x['full_native_sha256'] for h,x in states.items()},initial_model_native_sha256=states['1']['model_native_sha256'],all_six_initial_models_identical=True,all_six_full_states_identical=True,reason='Native and SWAD initial tensors, optimizer and RNG are identical; roles are separate private files',states=states,private_roles_for_all_six_folds=True,new_UWB_fit_started=False)
    write(P/'INITIAL_ASSET.json',q);files['phase/INITIAL_ASSET.json']=P/'INITIAL_ASSET.json'
    for n in read(P/'shared/DEPENDENCIES.json')['files']:files['native_source/'+str(Path(n).relative_to(c['native_source']))]=Path(n)
    files['private_IDs/source_ids.npz']=Path(c['ids'])
    result=build(P/'initial_backup','swad_native_loso6_initial_20261108',files,'fixed_fullsix_SWAD_untrained_native_SWAD_states_all_private_roles_before_fit',chunk_size=7340032,limit=100663296)
    result.update(state='all6_untrained_initial_native_SWAD_roles_sources_streamed',complete_modelset=False,complete_initial_model=True,complete_training_modelset=False);write(P/'initial_backup/MANIFEST.json',result);return result

def state():
    import engine as e
    from common import B
    m,o,g=e.make();st=e.capture(m,o,g,0,0)
    assert e.semantic(st['model'])=='2aae73c58e38ca324027e888955b130a82fbaece7167df3465522ec51986d2d3'
    assert o.native.count==o.swad.updates==o.swad.last_epoch==0 and not o.native.slow and not o.native.fast.state
    ss=o.swad.state_dict();assert int(ss['segment']['n_averaged'])==int(ss['tail']['n_averaged'])==0 and ss['converge_Q']==ss['smooth_Q']==[] and ss['final_model'] is None and ss['threshold'] is None and not ss['finalized']
    e.atomic_save(B/'initial.pt',st);write(B/'INITIAL_STATE_METADATA.json',dict(file_sha256=sha(B/'initial.pt'),full_native_sha256=e.semantic(st),model_native_sha256=e.semantic(st['model']),updates=0,SWAD_all_queues_empty=True,SWAD_average_counts_zero=True,native_optimizer_empty=True))

def check():
    d=P/'initial_backup';m=read(d/'MANIFEST.json');rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
    for r in rows:
        q=r['metadata'];assert not q['shared'] and q['permissions']==[dict(type='user',role='owner')] and q['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf']
        assert r['raw_fetch'] and r['download_verified'] and not r['include_base64']
    v=verify(m,rows);write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
    receipt=dict(state='all6_initial_native_SWAD_roles_sources_owneronly_raw_allmember_verified',archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],members_manifest_sha256=m['members_manifest_sha256'],scope=m['scope'],complete_untrained_initial_state=True,complete_trained_modelset=False,chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows])
    write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',raw_fetch_options=dict(download_raw_file=True,include_base64=False),max_concurrency=3,**receipt));return v
if __name__=='__main__':
    if sys.argv[1]=='state':state()
    else:print(json.dumps(create() if sys.argv[1]=='build' else check()))
