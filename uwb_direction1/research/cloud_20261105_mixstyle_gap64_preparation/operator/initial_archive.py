"""Initial model/roles/source private insurance; no fitting or diagnostic truth."""
import sys,json
from pathlib import Path
sys.path.insert(0,'/workspace/uwb-mixstyle-gap64-20261105')
from common import B,ready,read,write,sha
import stream_archive as ar
def build():
    c=ready();assert read(B/'SYNTHETIC_RESULT.json')['state']=='pass' and read(B/'RESOURCE_RESULT.json')['gate_passed']
    assert read(B/'ACTIVATION.json')['prior_summary_sha256']==sha(c['prior_CutMix_summary'])
    prior=read(c['prior_CutMix_summary']);assert prior['predeclared_gate_passed'] is False and prior['route']==c['prior_CutMix_exact_failed_route'] and prior['finalizer_invocations']==1
    assert not (B/'run').exists() and not (B/'initial.pt').exists()
    import engine as e
    from data import load
    m,o,g=e.make();state=e.capture(m,o,g,0,0);e.atomic_save(B/'initial.pt',state)
    ids,u,loc,maps,_,_=load(False)
    e.np.savez_compressed(B/'roles.npz',**{k:ids[v] for k,v in loc.items()})
    write(B/'INITIAL_ASSET.json',dict(initial_model_file_sha256=sha(B/'initial.pt'),initial_native_state_sha256=e.semantic(state),roles_sha256=sha(B/'roles.npz'),source_ids_sha256=sha(c['ids']),maps_sha256=sha(c['maps']),diagnostic_labels_read=False,new_UWB_fit_started=False))
    files={'study/'+n:B/n for n in ['ACTIVATION.json','SOURCE_REVIEW.json','ARCHIVE_DEPENDENCIES.json',*read(B/'FROZEN_HASHES.json'),'FROZEN_HASHES.json','REMOTE_PREREGISTRATION.json','SYNTHETIC_RESULT.json','RESOURCE_RESULT.json','initial.pt','roles.npz','INITIAL_ASSET.json']}
    files['private_IDs/source_ids.npz']=Path(c['ids'])
    for n in read(B/'DEPENDENCIES.json')['files']:files['native_source/'+str(Path(n).relative_to(Path(c['native_source'])))]=Path(n)
    for ref in c['author_reference_files']:files['author_review/'+Path(ref).name]=Path(ref)
    files['lineage/noMixup_reference_SUMMARY.json']=Path(c['source1_reference_summary'])
    prior=Path(c['prior_CutMix_summary']).parent
    for src,dst in [('SUMMARY.json','prior_CutMix_SUMMARY.json'),('STATE.json','prior_CutMix_POST_SCORE_STATE.json'),('FINALIZE_ONCE.lock','prior_CutMix_FINALIZE_ONCE.lock')]:files['lineage/'+dst]=prior/src
    for ref in (B/'operator').glob('*'):
        if ref.is_file() and ref.suffix in ['.py','.json']:files['operator/'+ref.name]=ref
    m=ar.build(B/'initial_backup','mixstyle_gap64_initial_20261105',files,'fixed_MixStyle_GAP64_initial_model_roles_and_sources_before_UWB_fit',chunk_size=7340032,limit=67108864)
    m.update(state='initial_seed_roles_sources_streamed_before_new_fits',complete_modelset=False,complete_initial_model=True,complete_training_modelset=False);m.pop('shared_initial_encoder_included_once',None);write(B/'initial_backup/MANIFEST.json',m)
    return m
def verify():
    d=B/'initial_backup';m=read(d/'MANIFEST.json');rows=[read(d/f'READBACK_{i:03}.json') for i in range(len(m['chunks']))]
    for r in rows:
        assert r['raw_fetch'] and r['download_verified'] and not r['include_base64']
        q=r['metadata'];assert not q['shared'] and q['parent_ids']==['12AGMwT99yXCTYBQocQaX901Br-IelmOf'] and q['permissions']==[dict(role='owner',type='user')]
    v=ar.verify(m,rows);write(d/'READBACK_INPUT.json',rows);write(d/'READBACK_VERIFIED.json',v)
    receipt=dict(state='initial_model_roles_sources_owner_only_raw_allmember_verified',archive_sha256=m['archive_sha256'],archive_bytes=m['archive_bytes'],scope=m['scope'],members_manifest_sha256=m['members_manifest_sha256'],complete_initial_seed_not_trained_modelset=True,diagnostic_labels_read=False,chunks=[{k:v for k,v in r.items() if k!='path'} for r in rows])
    write(d/'DRIVE_BACKUP_RECEIPT.json',receipt);write(d/'CLOUD_DOWNLOAD_INDEX.json',dict(folder_id='12AGMwT99yXCTYBQocQaX901Br-IelmOf',max_concurrency=3,raw_fetch_options=dict(download_raw_file=True,include_base64=False),**receipt));return v
if __name__=='__main__':
    out=build() if sys.argv[1]=='build' else verify()
    parts=len(out['chunks']) if 'chunks' in out else int(out.get('chunks_verified',0))
    print(json.dumps({k:v for k,v in out.items() if k!='chunks'}|dict(parts=parts)))
