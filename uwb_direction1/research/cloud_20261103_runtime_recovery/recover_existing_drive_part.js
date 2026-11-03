async ({base,index},api) => {
  // Exactly one raw fetch of a known existing upload. Never uploads or shares.
  const quote=s=>"'"+s.replaceAll("'","'\\''")+"'";
  const clean=s=>String(s).replace(/https?:\/\/\S+/g,'[redacted URL]').replace(/Bearer\s+\S+/gi,'[redacted bearer]');
  const suffix=String(index).padStart(3,'0');let stage='local_evidence',file_id=null;
  const save=async(name,value)=>api.apply_patch('*** Begin Patch\n*** Add File: '+base+'/'+name+'\n+'+JSON.stringify(value,null,2).replace(/\n/g,'\n+')+'\n*** End Patch');
  try {
    if(!Number.isInteger(index)||index<0||!base.startsWith('/workspace/'))throw Error('Invalid index/root');
    const py='from pathlib import Path;import json;b=Path('+JSON.stringify(base)+');i='+index+';assert not (b/("READBACK_"+str(i).zfill(3)+".json")).exists(),"Already verified: inspect receipt before another fetch";m=json.loads((b/"MANIFEST.json").read_text());u=json.loads((b/("UPLOAD_"+str(i).zfill(3)+".json")).read_text());c=m["chunks"][i];assert all(c[k]==u[k] for k in ("name","bytes","sha256"));print(json.dumps(u))';
    const r=await api.exec_command({cmd:'python -c '+quote(py),max_output_tokens:1000});
    if(r.exit_code!==0)throw Error('Missing/changed upload evidence or existing readback');
    const c=JSON.parse(r.output);file_id=c.file_id;if(typeof file_id!=='string'||!file_id)throw Error('Known upload ID absent');
    stage='metadata';const response=await api.mcp__codex_apps__google_drive_get_file_metadata({fileId:file_id,fields:'id,name,mimeType,size,parents,shared,permissions(type,role),trashed,webViewLink'});
    const m=response.structuredContent;
    if(response.isError||!m||m.id!==file_id||m.shared!==false||m.trashed===true||Number(m.size)!==c.bytes||m.parent_ids?.length!==1||m.parent_ids[0]!=='12AGMwT99yXCTYBQocQaX901Br-IelmOf'||m.permissions?.length!==1||m.permissions[0].role!=='owner'||m.permissions[0].type!=='user'||!m.mime_type||m.mime_type.startsWith('application/vnd.google-apps.')||typeof m.url!=='string')throw Error('Metadata/privacy/parent/size/MIME mismatch');
    stage='raw_fetch';const raw=await api.mcp__codex_apps__google_drive_fetch({url:m.url,download_raw_file:true,include_base64:false});
    if(raw.isError){const message=(raw.content??[]).filter(x=>x.type==='text').map(x=>x.text).join('\n');throw Error('Raw provider failure: '+clean(message).slice(0,1200));}
    const uri=raw.structuredContent?.file_uri??raw.file_uri;
    const id=typeof uri==='object'?uri?.file_id:uri;
    if(typeof id!=='string'||!(/^(sediment:\/\/)?file_/.test(id)))throw Error('Raw download reference missing; no materialization or readback certificate issued');
    stage='materialize';const downloaded=await api.download_file({file_id:id.replace('sediment://','')});
    if(typeof downloaded.path!=='string')throw Error('Materialized path missing');
    stage='SHA_and_identical_replacement';
    const verify='from pathlib import Path;import hashlib,os;p=Path('+JSON.stringify(base+'/'+c.name)+');q=Path('+JSON.stringify(downloaded.path)+');assert p!=q and not p.is_symlink() and not q.is_symlink();assert p.stat().st_size==q.stat().st_size=='+c.bytes+';assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(q.read_bytes()).hexdigest()=='+JSON.stringify(c.sha256)+';os.replace(q,p);assert hashlib.sha256(p.read_bytes()).hexdigest()=='+JSON.stringify(c.sha256)+';print("verified")';
    const checked=await api.exec_command({cmd:'python -c '+quote(verify),max_output_tokens:1000});
    if(checked.exit_code!==0||checked.output.trim()!=='verified')throw Error('Materialized bytes/SHA or identical transport replacement failed');
    const row={...c,path:base+'/'+c.name,remote_raw_bytes_replaced_identical_transport:true,metadata:{shared:false,parent_ids:m.parent_ids,mime_type:m.mime_type,size_bytes:Number(m.size),permissions:m.permissions.map(x=>({role:x.role,type:x.type}))},raw_fetch:true,include_base64:false,download_verified:true,recovered_known_uploaded_ID_without_reupload:true};
    await save('READBACK_'+suffix+'.json',row);return {index,file_id,verified:true,upload_retried:false};
  } catch(e) {
    const failure={index,file_id,stage,error:clean(e).slice(0,1600),upload_retried:false,whole_archive_verified:false};
    await save('RECOVERY_ERROR_'+suffix+'_'+Date.now()+'.json',failure);throw Error(JSON.stringify(failure));
  }
}
