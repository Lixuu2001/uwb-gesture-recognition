async function fetchPart(i){
 const base=load('GAPRestoreBase'),c=load('GAPRestoreManifest').chunks[i];
 const quote=x=>"'"+x.replaceAll("'","'\\''")+"'";
 const run=async code=>{const r=await tools.exec_command({cmd:'python -c '+quote(code),max_output_tokens:700});if(r.exit_code!==0)throw Error('Local raw verification/materialization failed');return r.output;};
 const save=async(name,value)=>tools.apply_patch('*** Begin Patch\n*** Add File: '+base+'/'+name+'\n+'+JSON.stringify(value,null,2).replace(/\n/g,'\n+')+'\n*** End Patch');
 try{
  const check=JSON.parse(await run('from pathlib import Path;import json,hashlib;b=Path('+JSON.stringify(base)+');p=b/"parts"/'+JSON.stringify(c.name)+';r=b/'+JSON.stringify('READBACK_'+String(i).padStart(3,'0')+'.json')+';print(json.dumps(dict(done=r.exists(),part=p.exists(),sha256=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None)))'));
  if(check.done){if(check.sha256!==c.sha256)throw Error('Existing verified part changed');return {i,held:c.held,reused:true};}
  if(check.part)throw Error('Part exists without final readback; inspect/reuse known ID, no blind overwrite');
  const md=await tools.mcp__codex_apps__google_drive_get_file_metadata({fileId:c.file_id,fields:'id,name,size,mimeType,parents,shared,permissions,trashed,webViewLink'});const m=md.structuredContent;
  if(m?.shared!==false||m?.trashed===true||m?.permissions?.length!==1||m.permissions[0].role!=='owner'||m.permissions[0].type!=='user'||Number(m.size)!==c.bytes||JSON.stringify(m.parent_ids)!==JSON.stringify(['12AGMwT99yXCTYBQocQaX901Br-IelmOf'])||m.mime_type?.startsWith('application/vnd.google-apps.'))throw Error('Metadata privacy/size/parent/MIME guard');
  const raw=await tools.mcp__codex_apps__google_drive_fetch({url:m.url,download_raw_file:true,include_base64:false});if(raw.isError)throw Error('Provider raw fetch failed; retain known ID');
  const ref=raw.structuredContent?.file_uri??raw.file_uri;const id=typeof ref==='object'?ref.file_id:ref;if(typeof id!=='string')throw Error('No materialization reference');
  const down=await tools.download_file({file_id:id.replace('sediment://','')});
  const result=JSON.parse(await run('from pathlib import Path;import hashlib,json,os;s=Path('+JSON.stringify(down.path)+');t=Path('+JSON.stringify(base+'/parts/'+c.name)+');assert s.stat().st_size=='+c.bytes+' and hashlib.sha256(s.read_bytes()).hexdigest()=='+JSON.stringify(c.sha256)+';assert not t.exists();os.replace(s,t);print(json.dumps(dict(bytes=t.stat().st_size,sha256=hashlib.sha256(t.read_bytes()).hexdigest())))'));
  if(result.bytes!==c.bytes||result.sha256!==c.sha256)throw Error('Raw verified output mismatch');
  await save('READBACK_'+String(i).padStart(3,'0')+'.json',{...c,path:base+'/parts/'+c.name,metadata:{shared:false,parent_ids:m.parent_ids,mime_type:m.mime_type,permissions:[{type:'user',role:'owner'}]},raw_fetch:true,include_base64:false,immediate_materialization_SHA_verified:true,download_verified:true,new_upload:false});return {i,held:c.held,verified:true};
 }catch(err){await save('ERROR_'+String(i).padStart(3,'0')+'_'+Date.now()+'.json',{i,held:c.held,file_id:c.file_id,error:String(err).replace(/https?:\/\/\S+/g,'[redacted URL]'),new_upload:false});throw Error('Known Drive ID restore failed for part '+i+'; inspect durable receipt, never upload or blindly repeat verified parts');}
}
