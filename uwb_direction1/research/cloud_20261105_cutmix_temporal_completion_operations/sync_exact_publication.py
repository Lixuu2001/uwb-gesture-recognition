"""Sync local index only after exact physical-file and remote publication proofs."""
from pathlib import Path
import subprocess,json,sys,hashlib
def main():
    mp=Path(sys.argv[1]);m=json.loads(mp.read_text());proof=json.loads(mp.with_name(mp.stem+'_VERIFIED.json').read_text());g=Path(m['root']);commit=proof['commit']
    assert proof['force'] is False and proof['branch_head_verified'] and proof['parent']==m['parent']
    assert proof['exact_readback_verified']==len(m['files'])
    def git(*a):return subprocess.check_output(['git',*a],cwd=g,text=True).strip()
    assert git('rev-parse','HEAD')==m['parent']
    r=subprocess.run(['git','fetch','--quiet','origin','codex/uwb-direction1-cloud-20260930'],cwd=g,capture_output=True,text=True);assert r.returncode==0,'Fetch failed; inspect transport without exposing credentials'
    assert git('rev-parse','FETCH_HEAD')==commit and git('rev-parse',commit+'^')==m['parent']
    assert not git('diff','--name-only',m['parent'],commit,'--','uwb_direction1/original')
    allowed={f['path'] for f in m['files']};changed=set(git('diff','--name-only').splitlines())|set(git('ls-files','--others','--exclude-standard').splitlines());assert changed<=allowed
    for f in m['files']:
        raw=(g/f['path']).read_bytes();assert len(raw)==f['bytes'] and hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==f['blob_sha1']
        assert raw==subprocess.check_output(['git','show',commit+':'+f['path']],cwd=g)
    subprocess.run(['git','reset','--mixed',commit],cwd=g,capture_output=True,check=True)
    assert git('rev-parse','HEAD')==commit and not git('status','--porcelain')
    print(json.dumps(dict(HEAD=commit,physical_files_verified=len(m['files']),clean=True,original789_unchanged=True,force=False)))
if __name__=='__main__':main()
