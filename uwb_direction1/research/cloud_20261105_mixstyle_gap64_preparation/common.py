from pathlib import Path
import json,hashlib,os,time
B=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp')
    assert not tmp.exists(),str(tmp)
    with tmp.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for a in iter(lambda:f.read(4194304),b''):h.update(a)
    return h.hexdigest()
def protocol():return read(B/'PROTOCOL.json')
def ready(real=False):
    for n,h in read(B/'FROZEN_HASHES.json').items():assert sha(B/n)==h,n
    for n,h in read(B/'DEPENDENCIES.json')['files'].items():assert sha(n)==h,n
    for p,h in protocol()['author_reference_files'].items():assert sha(p)==h,p
    q=read(B/'REMOTE_PREREGISTRATION.json');assert q['branch_head_verified'] and q['frozen_sha256']==sha(B/'FROZEN_HASHES.json')
    if real:
        assert read(B/'RESOURCE_RESULT.json')['gate_passed'] and read(B/'SYNTHETIC_RESULT.json')['state']=='pass'
        q=read(B/'REAL_RELEASE.json');assert q['branch_head_verified'] and q['frozen_sha256']==sha(B/'FROZEN_HASHES.json')
        assert q['initial_archive_raw_allmember_verified']
        c=protocol()
        prior=read(c['prior_CutMix_summary']);assert prior['predeclared_gate_passed'] is False and prior['route']==c['prior_CutMix_exact_failed_route'] and prior['finalizer_invocations']==1
        assert Path(c['prior_CutMix_once_lock']).exists() and sha(c['prior_CutMix_summary'])==q['activation_prior_summary_sha256']
        assert read(B/'ACTIVATION.json')['prior_summary_sha256']==q['activation_prior_summary_sha256']
        assert sha(c['ids'])==c['ids_sha256'] and sha(c['maps'])==c['maps_sha256']
        for n,h in c['input_sha256'].items():assert sha(Path(c['data'])/'cache'/(n+'.npy'))==h,n
        assert sha(c['source1_reference_summary'])==c['source1_reference_sha256']
        assert sha(B/'initial.pt')==q['initial_model_file_sha256']
        import stream_archive
        d=B/'initial_backup';v=read(d/'READBACK_VERIFIED.json')
        assert stream_archive.verify(read(d/'MANIFEST.json'),read(d/'READBACK_INPUT.json'))==v
        assert v['archive_sha256']==q['initial_archive_sha256']
    return protocol()
