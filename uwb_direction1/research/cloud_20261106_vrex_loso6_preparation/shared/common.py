from pathlib import Path
import json,hashlib,os
PHASE=Path('/workspace/uwb-vrex-loso6-20261106')
B=Path(os.environ.get('UWB_VREX_LOSO_FOLD_ROOT',str(PHASE/'folds/held1'))).resolve()
assert B in [(PHASE/'folds'/f'held{h}').resolve() for h in range(1,7)]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(4194304),b''):h.update(x)
    return h.hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');assert not tmp.exists()
    with tmp.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def protocol():return read(B/'PROTOCOL.json')
def ready(real=False):
    for n,h in read(PHASE/'FROZEN_HASHES.json').items():assert sha(PHASE/n)==h,n
    for n,h in read(PHASE/'shared/DEPENDENCIES.json')['files'].items():assert sha(n)==h,n
    c=protocol();assert c['held'] in range(1,7) and c['validation_user']==c['held']%6+1
    assert c['train_users']==[u for u in range(1,7) if u not in [c['held'],c['validation_user']]]
    for n,h in c['author_reference_files'].items():assert sha(n)==h,n
    prior=read(PHASE/'PRIOR_CLOSURE_AND_REUSE.json')
    for n,h in prior['files'].items():assert sha(n)==h,n
    source=Path('/workspace/uwb-mmd-loso6-20261106');old=Path('/workspace/uwb-mmd-gap64-20261106')
    assert read(source/'SUMMARY.json')['cohort_spending_gate_passed'] is False and (source/'FINALIZE_ONCE.lock').exists()
    assert not(old/'SUMMARY.json').exists() and not(old/'FINALIZE_ONCE.lock').exists()
    legacy=Path('/workspace/uwb-vrex-gap64-20261106');assert not(legacy/'STATE.json').exists() and not(legacy/'run').exists()
    q=read(PHASE/'REMOTE_PREREGISTRATION.json');assert q['branch_head_verified'] and q['force'] is False and q['frozen_sha256']==sha(PHASE/'FROZEN_HASHES.json')
    if real:
        q=read(PHASE/'REAL_RELEASE.json');assert q['branch_head_verified'] and q['force'] is False and q['frozen_sha256']==sha(PHASE/'FROZEN_HASHES.json')
        assert read(PHASE/'SYNTHETIC_SUMMARY.json')['all_passed'] and read(PHASE/'RESOURCE_GATE.json')['gate_passed']
        assert sha(B/'initial.pt')==q['initial_model_file_sha256']
        from stream_archive import verify
        d=PHASE/'initial_backup';assert verify(read(d/'MANIFEST.json'),read(d/'READBACK_INPUT.json'))==read(d/'READBACK_VERIFIED.json')
        assert read(d/'MANIFEST.json')['archive_sha256']==q['initial_archive_sha256']
        assert sha(c['ids'])==c['ids_sha256'] and sha(c['maps'])==c['maps_sha256']
        for n,h in c['input_sha256'].items():assert sha(Path(c['data'])/'cache'/(n+'.npy'))==h,n
    return c
