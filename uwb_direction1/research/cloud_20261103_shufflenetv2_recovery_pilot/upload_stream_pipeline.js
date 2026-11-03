async(i)=>{
const base=load("ShuffleNetRecoveryCompleteDir"),c=load("ShuffleNetRecoveryCompleteManifest").chunks[i];let stage="upload",file_id=null;
try{
const preflight=await tools.exec_command({cmd:"python -c 'from pathlib import Path;import json;b=Path("+JSON.stringify(base)+");i="+i+";assert not list(b.glob(\"UPLOAD_RESPONSE_\"+str(i).zfill(3)+\"_*.json\")) and not (b/(\"UPLOAD_\"+str(i).zfill(3)+\".json\")).exists(),\"Prior response/known ID requires explicit reuse/inspection, no upload retry\"'",max_output_tokens:200});
if(preflight.exit_code!==0)throw Error('Prior response or known ID must be inspected/reused; automatic upload retry blocked');
const u=await tools.mcp__codex_apps__google_drive_upload_file({file_uri:base+"/"+c.name,file_name:c.name,mime_type:"application/octet-stream",parent_folder_id:"12AGMwT99yXCTYBQocQaX901Br-IelmOf"});
const sanitized=JSON.parse(JSON.stringify(u,(k,v)=>/url|token|secret|authorization|email/i.test(k)?'[redacted]':typeof v==='string'?v.replace(/https?:\/\/\S+/g,'[redacted URL]'):v));
await tools.apply_patch('*** Begin Patch\n*** Add File: '+base+'/UPLOAD_RESPONSE_'+String(i).padStart(3,'0')+'_'+Date.now()+'.json\n+'+JSON.stringify(sanitized,null,2).replace(/\n/g,'\n+')+'\n*** End Patch');
let d=u.structuredContent;if(!d)for(const q of u.content||[])if(q.type==="text"){try{d=JSON.parse(q.text);break;}catch{}}
if(!d?.id&&d?.structuredContent?.id)d=d.structuredContent;
if(!d?.id&&d?.file?.id)d=d.file;
if(!d?.id)throw Error("No uploadedID; exact-name search required before retry");file_id=d.id;
const up={...c,file_id};await tools.apply_patch("*** Begin Patch\n*** Add File: "+base+"/UPLOAD_"+String(i).padStart(3,"0")+".json\n+"+JSON.stringify(up,null,2).replace(/\n/g,"\n+")+"\n*** End Patch");
stage="metadata";const r=await tools.mcp__codex_apps__google_drive_get_file_metadata({fileId:file_id,fields:"id,name,mimeType,size,parents,permissions,shared,trashed,webViewLink"});const m=r.structuredContent;
if(m.shared!==false||m.permissions.length!==1||m.permissions[0].role!=="owner"||Number(m.size)!==c.bytes||!m.parent_ids.includes("12AGMwT99yXCTYBQocQaX901Br-IelmOf")||(!m.mime_type||m.mime_type.startsWith("application/vnd.google-apps."))||m.trashed===true)throw Error("Privacy/parent/size/MIME check failed");
stage="raw_fetch";const raw=await tools.mcp__codex_apps__google_drive_fetch({url:m.url,download_raw_file:true,include_base64:false});
stage="materialize";const down=await tools.download_file({file_id:raw.structuredContent.file_uri.file_id.replace("sediment://","")});
stage="SHA";const py="from pathlib import Path;import hashlib,json;p=Path("+JSON.stringify(down.path)+");print(json.dumps(dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))";
const v=await tools.exec_command({cmd:"python -c '"+py.replaceAll("'","'\\''")+"'",max_output_tokens:300});const check=JSON.parse(v.output);if(check.bytes!==c.bytes||check.sha256!==c.sha256)throw Error("Materialized size/SHA mismatch");
const beforeReplace={stage:'verified_raw_download_before_identical_transport_replacement',i,local_part:base+"/"+c.name,download:down.path,bytes:c.bytes,sha256:c.sha256,file_id};
await tools.apply_patch("*** Begin Patch\n*** Add File: "+base+"/TRANSPORT_REPLACE_"+String(i).padStart(3,"0")+".json\n+"+JSON.stringify(beforeReplace,null,2).replace(/\n/g,"\n+")+"\n*** End Patch");
const replacepy="import sys;sys.path.insert(0,'/workspace/uwb-shufflenetv2-x1-recovery-pilot-20261103');from stream_archive import replace_verified_download;print(replace_verified_download("+JSON.stringify(base+"/"+c.name)+","+JSON.stringify(down.path)+","+c.bytes+","+JSON.stringify(c.sha256)+"))";
const replaced=await tools.exec_command({cmd:"python -c '"+replacepy.replaceAll("'","'\\''")+"'",max_output_tokens:200});if(replaced.exit_code!==0||replaced.output.trim()!==base+"/"+c.name)throw Error('Identical raw download transport replacement failed');
const row={...up,path:base+"/"+c.name,remote_raw_bytes_replaced_identical_transport:true,metadata:{shared:false,parent_ids:m.parent_ids,mime_type:m.mime_type,size_bytes:Number(m.size),permissions:m.permissions.map(x=>({role:x.role,type:x.type}))},raw_fetch:true,include_base64:false,download_verified:true};
await tools.apply_patch("*** Begin Patch\n*** Add File: "+base+"/READBACK_"+String(i).padStart(3,"0")+".json\n+"+JSON.stringify(row,null,2).replace(/\n/g,"\n+")+"\n*** End Patch");return{i,verified:true};
}catch(e){const row={i,name:c.name,file_id,stage,error:String(e).replace(/https?:\/\/\S+/g,"[redacted URL]")};await tools.apply_patch("*** Begin Patch\n*** Add File: "+base+"/ERROR_"+String(i).padStart(3,"0")+"_"+Date.now()+".json\n+"+JSON.stringify(row,null,2).replace(/\n/g,"\n+")+"\n*** End Patch");throw Error(JSON.stringify(row));}
}



