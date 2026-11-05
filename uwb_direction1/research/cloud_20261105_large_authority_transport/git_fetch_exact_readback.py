"""Read the SAME connector-created pending commit over Git pack transport, never push."""
from pathlib import Path
import subprocess,sys,json,hashlib,re
def main():
 manifest=Path(sys.argv[1]);m=json.loads(manifest.read_text());g=Path(m['root']);commit=sys.argv[2]
 def run(args):return subprocess.check_output(['git',*args],cwd=g)
 branch='refs/heads/codex/uwb-direction1-cloud-20260930'
 assert run(['ls-remote','origin',branch]).decode().split()[0]==m['parent']
 p=subprocess.run(['git','fetch','--quiet','origin',commit],cwd=g,capture_output=True,text=True);assert p.returncode==0,re.sub(r'https?://\S+','[redacted URL]',p.stderr)
 assert run(['rev-parse','FETCH_HEAD']).decode().strip()==commit and run(['show','-s','--format=%P',commit]).decode().strip()==m['parent']
 assert run(['rev-parse',commit+':uwb_direction1/original'])==run(['rev-parse',m['parent']+':uwb_direction1/original'])
 allowed={r['path'] for r in m['files']};assert set(run(['diff-tree','--no-commit-id','--name-only','-r',commit]).decode().splitlines())<=allowed
 for f in m['files']:
  b=run(['show',commit+':'+f['path']]);assert b==(g/f['path']).read_bytes() and len(b)==f['bytes'] and hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==f['blob_sha1'],f['path']
 proof=dict(parent=m['parent'],commit=commit,exact_readback_verified=len(m['files']),transport='same_remote_commit_Git_pack_bytes_independently_hashed_and_compared_before_ref_move',force=False,original789_unchanged=True)
 manifest.with_name(manifest.stem+'_GIT_READBACK_VERIFIED.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
