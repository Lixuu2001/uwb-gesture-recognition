async function publish(manifestPath,message){
 const quote=x=>"'"+x.replaceAll("'","'\\''")+"'";
 const local=async path=>{let body='';for(let offset=0;;offset+=4000){
  const code='import json,pathlib;s=pathlib.Path('+JSON.stringify(path)+').read_text();print(json.dumps({"part":s['+offset+':'+(offset+4000)+'],"end":len(s)<='+(offset+4000)+'},ensure_ascii=False))';
  const r=await tools.exec_command({cmd:'python -c '+quote(code),max_output_tokens:10000});if(r.exit_code!==0)throw Error('Local read failed '+path);const q=JSON.parse(r.output);body+=q.part;if(q.end)return body;
 }};
 const M=JSON.parse(await local(manifestPath)),repo='Lixuu2001/uwb-gesture-recognition',branch='codex/uwb-direction1-cloud-20260930';
 async function head(){for(let n=0;n<3;n++){const r=await tools.mcp__codex_apps__github_fetch_commit({repo_full_name:repo,commit_sha:branch});const h=r.structuredContent?.commit?.sha;if(h)return h;}throw Error('No readable current branch head');}
 const parent=await head();if(parent!==M.parent)throw Error('Parent changed; inspect latest head '+parent);
 let next=0;const files=[];
 async function worker(){for(;;){const i=next++;if(i>=M.files.length)return;const f=M.files[i];let body;
  if(f.remote_transform){
   let prior=load('HeadNow')===parent?load('Remote:'+f.path):undefined;
   for(let n=0;typeof prior!=='string'&&n<3;n++)prior=(await tools.mcp__codex_apps__github_fetch_file({repository_full_name:repo,path:f.path,ref:parent})).structuredContent?.content;
   if(typeof prior!=='string')throw Error('Cannot read immutable authority '+f.path);
   if(f.remote_transform==='prepend')body=M.note+prior;
   else{
    body=prior;
    for(const [k,v] of Object.entries(M.status_updates)){
     const rx=new RegExp('("'+k+'"\\s*:\\s*)("(?:[^"\\\\]|\\\\.)*"|true|false|null|-?\\d+(?:\\.\\d+)?)');
     if(!rx.test(body))throw Error('Missing root scalar '+k);body=body.replace(rx,(_,prefix)=>prefix+JSON.stringify(v));
    }
    body=body.trimEnd();if(!body.endsWith('}'))throw Error('Invalid root JSON');body=body.slice(0,-1)+',\n  "'+M.status_insert_key+'": '+M.status_insert_text+'\n}\n';
   }
  }else body=await local(M.root+'/'+f.path);
  let blob=await tools.mcp__codex_apps__github_create_blob({repository_full_name:repo,content:body,encoding:'utf-8'});
  if(!blob.structuredContent?.sha)blob=await tools.mcp__codex_apps__github_create_blob({repository_full_name:repo,content:body,encoding:'utf-8'});
  if(blob.structuredContent?.sha!==f.blob_sha1)throw Error('Independent local/Git blob hash mismatch '+f.path);
  files[i]={...f,body,sha:blob.structuredContent.sha};
 }}
 await Promise.all([worker(),worker(),worker(),worker()]);
 const t=await tools.mcp__codex_apps__github_create_tree({repository_full_name:repo,base_tree_sha:parent,tree_elements:files.map(f=>({path:f.path,mode:'100644',type:'blob',sha:f.sha}))});
 if(!t.structuredContent?.sha)throw Error('No created tree SHA');
 const c=await tools.mcp__codex_apps__github_create_commit({repository_full_name:repo,parent_sha:parent,tree_sha:t.structuredContent.sha,message});
 const commit=c.structuredContent?.sha;if(!commit)throw Error('No created commit SHA; do not retry blind');store('PendingPublicationCommit',commit);
 let index=0;
 async function verifier(){for(;;){const i=index++;if(i>=files.length)return;const f=files[i];let body;
  for(let n=0;n<3;n++){body=(await tools.mcp__codex_apps__github_fetch_file({repository_full_name:repo,path:f.path,ref:commit})).structuredContent?.content;if(body===f.body)break;}
  if(body!==f.body)throw Error('Exact remote readback failed '+f.path);
 }}
 await Promise.all([verifier(),verifier(),verifier(),verifier()]);
 const u=await tools.mcp__codex_apps__github_update_ref({repository_full_name:repo,branch_name:branch,sha:commit,expected_sha:parent,force:false});
 let refRecovery=null;
 if(u.isError){
  const observed=await head();
  refRecovery={state:'ref_API_error_inspected_before_nonforce_recovery',parent,commit,observed_head:observed,API_error:u.structuredContent,force:false};
  const evidence=manifestPath.replace(/\.json$/,'_REF_TRANSPORT_RECOVERY.json');
  await tools.apply_patch('*** Begin Patch\n*** Add File: '+evidence+'\n+'+JSON.stringify(refRecovery,null,2).replace(/\n/g,'\n+')+'\n*** End Patch');
  if(observed!==commit){
   if(observed!==parent)throw Error('Ref changed after API error; do not publish stale parent '+observed);
   const code='import subprocess,json,re;g='+JSON.stringify(M.root)+';commit='+JSON.stringify(commit)+';parent='+JSON.stringify(parent)+';branch='+JSON.stringify('refs/heads/'+branch)+';r=subprocess.run(["git","fetch","--quiet","origin",commit],cwd=g,text=True,capture_output=True);assert r.returncode==0,re.sub(r"https?://\\S+","[redacted URL]",r.stderr);assert subprocess.check_output(["git","rev-parse",commit+"^"],cwd=g,text=True).strip()==parent;assert subprocess.check_output(["git","ls-remote","origin",branch],cwd=g,text=True).split()[0]==parent;r=subprocess.run(["git","push","--porcelain","origin",commit+":"+branch],cwd=g,text=True,capture_output=True);print(json.dumps(dict(exit_code=r.returncode,stdout=re.sub(r"https?://\\S+","[redacted URL]",r.stdout),stderr=re.sub(r"https?://\\S+","[redacted URL]",r.stderr),force=False)));assert r.returncode==0';
   const pushed=await tools.exec_command({cmd:'python -c '+quote(code),yield_time_ms:1000,max_output_tokens:2000});
   let done=pushed;
   while(done.session_id)done=await tools.write_stdin({session_id:done.session_id,chars:'',yield_time_ms:1000,max_output_tokens:2000});
   if(done.exit_code!==0)throw Error('Ordinary non-force Git push failed; inspect persisted API evidence and CLI status');
   refRecovery.recovery='ordinary_nonforce_Git_push_same_commit_after_unchanged_parent_guard';
  }else refRecovery.recovery='API_error_but_branch_already_at_exact_verified_commit';
 }
 if(await head()!==commit)throw Error('Moved head not verified');
 const proof={parent,commit,files:files.length,exact_readback_verified:files.length,branch_head_verified:true,force:false,ref_transport_recovery:refRecovery};
 const p=manifestPath.replace(/\.json$/,'_VERIFIED.json');await tools.apply_patch('*** Begin Patch\n*** Add File: '+p+'\n+'+JSON.stringify(proof,null,2).replace(/\n/g,'\n+')+'\n*** End Patch');
 store('HeadNow',commit);for(const f of files)if(f.remote_transform)store('Remote:'+f.path,f.body);return proof;
}
