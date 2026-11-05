"""Exact content, sole latest parent, ordinary nonforce Git publication fallback."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,re
def main():
 mpath=Path(sys.argv[1]);m=json.loads(mpath.read_text());g=Path(m['root']);branch='refs/heads/codex/uwb-direction1-cloud-20260930';env=os.environ.copy()
 def run(args,**kw):return subprocess.check_output(['git',*args],cwd=g,env=env,**kw)
 def head():return run(['ls-remote','origin',branch],text=True).split()[0]
 parent=m['parent'];assert run(['rev-parse','HEAD'],text=True).strip()==parent and head()==parent
 assert len({f['path'] for f in m['files']})==len(m['files'])
 allowed={f['path'] for f in m['files']};existing=set(run(['diff','--name-only',parent],text=True).splitlines());assert existing<=allowed,existing-allowed
 fd,index=tempfile.mkstemp(prefix='uwb-publication-index-');os.close(fd);os.unlink(index);env['GIT_INDEX_FILE']=index
 try:
  run(['read-tree',parent])
  for row in m['files']:
   path=row['path'];assert path.startswith('uwb_direction1/') and not path.startswith('uwb_direction1/original/') and not path.endswith(('.pt','.pth','.npy','.npz','.bin','.jsonl'))
   raw=(g/path).read_bytes();expected=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest();assert len(raw)==row['bytes'] and expected==row['blob_sha1']
   actual=run(['hash-object','-w','--stdin'],input=raw).decode().strip();assert actual==expected
   run(['update-index','--add','--cacheinfo','100644',actual,path])
  tree=run(['write-tree'],text=True).strip();env.update(GIT_AUTHOR_NAME='Codex',GIT_AUTHOR_EMAIL='codex@openai.com',GIT_COMMITTER_NAME='Codex',GIT_COMMITTER_EMAIL='codex@openai.com')
  commit=run(['commit-tree',tree,'-p',parent],input=sys.argv[2]+'\n',text=True).strip()
  assert run(['show','-s','--format=%P',commit],text=True).strip()==parent
  assert set(run(['diff-tree','--no-commit-id','--name-only','-r',commit],text=True).splitlines())<=allowed
  assert run(['rev-parse',parent+':uwb_direction1/original'],text=True)==run(['rev-parse',commit+':uwb_direction1/original'],text=True)
  for row in m['files']:assert run(['show',commit+':'+row['path']])==(g/row['path']).read_bytes()
  pending=dict(parent=parent,commit=commit,files=len(m['files']),local_blob_and_tree_exact_verified=True,transport='ordinary_nonforce_git_after_connector_internal_errors',force=False)
  mpath.with_name(mpath.stem+'_CLI_PENDING.json').write_text(json.dumps(pending,indent=2)+'\n')
  assert head()==parent
  p=subprocess.run(['git','push','--porcelain','origin',commit+':'+branch],cwd=g,env=env,capture_output=True,text=True)
  pending.update(push_exit_code=p.returncode,stdout=re.sub(r'https?://\S+','[redacted URL]',p.stdout),stderr=re.sub(r'https?://\S+','[redacted URL]',p.stderr))
  mpath.with_name(mpath.stem+'_CLI_TRANSPORT.json').write_text(json.dumps(pending,indent=2)+'\n')
  assert p.returncode==0 and head()==commit,pending
  print(json.dumps({k:v for k,v in pending.items() if k not in ['stdout','stderr']}))
 finally:
  if Path(index).exists():Path(index).unlink()
if __name__=='__main__':main()
