async(i)=>{
const base=load("Adam100ArchiveDir"),c=load("Adam100ArchiveManifest").chunks[i];let stage="upload",file_id=null;
try{
if(i!==11)throw Error('Only explicitly reviewed part permitted');
const preflight=await tools.exec_command({cmd:"python -c 'import pathlib,json,datetime;p=pathlib.Path(\"/workspace/uwb-swad-loso6-20261108/held5_completed_backup\");assert (p/'\\''EXPLICIT_PART011_RECOVERY_PLAN.json'\\'').exists() and not(p/'\\''UPLOAD_011.json'\\'').exists() and not(p/'\\''READBACK_011.json'\\'').exists();f=(p/'\\''EXPLICIT_PART011_ATTEMPT_STARTED.json'\\'').open('\\''x'\\'');json.dump(dict(part=11,started=datetime.datetime.now(datetime.timezone.utc).isoformat(),explicit_single_attempt=True),f);f.close()'",max_output_tokens:200});
if(preflight.exit_code!==0)throw Error('Prior response or known ID must be inspected/reused; automatic upload retry blocked');
const u=await tools.mcp__codex_apps__google_drive_upload_file({file_uri:base+"/"+c.name,file_name:c.name,mime_type:"application/octet-stream",parent_folder_id:"12AGMwT99yXCTYBQocQaX901Br-IelmOf"});
let d=u.structuredContent;if(!d)for(const q of u.content||[])if(q.type==="text"){try{d=JSON.parse(q.text);break;}catch{}}
if(!d?.id&&d?.structuredContent?.id)d=d.structuredContent;
if(!d?.id&&d?.file?.id)d=d.file;
if(!d?.id){
const cleaned=JSON.parse(JSON.stringify(u,(k,v)=>/url|token|secret|authorization|email/i.test(k)?'[redacted]':typeof v==='string'?v.replace(/https?:\/\/\S+/g,'[redacted URL]').replace(/Bearer\s+\S+/gi,'[redacted bearer]'):v));
const serializedMissing=JSON.stringify(cleaned,null,2),missing=serializedMissing.length>16000?{bounded_response:true,response_characters:serializedMissing.length,top_level_keys:Object.keys(u),is_error:u.isError===true,uploaded_ID_absent:true}:cleaned;
await tools.apply_patch('*** Begin Patch\n*** Add File: '+base+'/MISSING_ID_RESPONSE_'+String(i).padStart(3,'0')+'_'+Date.now()+'.json\n+'+JSON.stringify(missing,null,2).replace(/\n/g,'\n+')+'\n*** End Patch');
throw Error("No uploadedID; response evidence preserved; exact-name search required before retry");
}file_id=d.id;
const up={...c,file_id};await tools.apply_patch("*** Begin Patch\n*** Add File: "+base+"/UPLOAD_"+String(i).padStart(3,"0")+".json\n+"+JSON.stringify(up,null,2).replace(/\n/g,"\n+")+"\n*** End Patch");
const sanitized=JSON.parse(JSON.stringify(u,(k,v)=>/url|token|secret|authorization|email/i.test(k)?'[redacted]':typeof v==='string'?v.replace(/https?:\/\/\S+/g,'[redacted URL]'):v));
const serialized=JSON.stringify(sanitized,null,2),summary=serialized.length>16000?{bounded_response:true,response_characters:serialized.length,file_id,top_level_keys:Object.keys(u),is_error:u.isError===true}:sanitized;
try{await tools.apply_patch('*** Begin Patch\n*** Add File: '+base+'/UPLOAD_RESPONSE_'+String(i).padStart(3,'0')+'_'+Date.now()+'.json\n+'+JSON.stringify(summary,null,2).replace(/\n/g,'\n+')+'\n*** End Patch');}
catch(err){await tools.apply_patch('*** Begin Patch\n*** Add File: '+base+'/UPLOAD_RESPONSE_PERSIST_WARNING_'+String(i).padStart(3,'0')+'_'+Date.now()+'.json\n+'+JSON.stringify({i,file_id,known_ID_already_persisted:true,error:String(err).replace(/https?:\/\/\S+/g,'[redacted URL]').slice(0,1200)},null,2).replace(/\n/g,'\n+')+'\n*** End Patch');}
stage="metadata";const r=await tools.mcp__codex_apps__google_drive_get_file_metadata({fileId:file_id,fields:"id,name,mimeType,size,parents,permissions,shared,trashed,webViewLink"});const m=r.structuredContent;
if(m.shared!==false||m.permissions.length!==1||m.permissions[0].role!=="owner"||Number(m.size)!==c.bytes||!m.parent_ids.includes("12AGMwT99yXCTYBQocQaX901Br-IelmOf")||(!m.mime_type||m.mime_type.startsWith("application/vnd.google-apps."))||m.trashed===true)throw Error("Privacy/parent/size/MIME check failed");
stage="raw_fetch";const raw=await tools.mcp__codex_apps__google_drive_fetch({url:m.url,download_raw_file:true,include_base64:false});
if(raw.isError)throw Error("Raw provider failure: "+(raw.content||[]).filter(x=>x.type==="text").map(x=>x.text).join(" ").replace(/https?:\/\/\S+/g,"[redacted URL]"));const uri=raw.structuredContent?.file_uri??raw.file_uri;const id=typeof uri==="object"?uri?.file_id:uri;if(typeof id!=="string")throw Error("No raw materialization reference");stage="materialize";const down=await tools.download_file({file_id:id.replace("sediment://","")});
stage="SHA";const py="from pathlib import Path;import hashlib,json;p=Path("+JSON.stringify(down.path)+");print(json.dumps(dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())))";
const v=await tools.exec_command({cmd:"python -c '"+py.replaceAll("'","'\\''")+"'",max_output_tokens:300});const check=JSON.parse(v.output);if(check.bytes!==c.bytes||check.sha256!==c.sha256)throw Error("Materialized size/SHA mismatch");
const beforeReplace={stage:'verified_raw_download_before_identical_transport_replacement',i,local_part:base+"/"+c.name,download:down.path,bytes:c.bytes,sha256:c.sha256,file_id};
await tools.apply_patch("*** Begin Patch\n*** Add File: "+base+"/TRANSPORT_REPLACE_"+String(i).padStart(3,"0")+".json\n+"+JSON.stringify(beforeReplace,null,2).replace(/\n/g,"\n+")+"\n*** End Patch");
const replacepy="import sys;sys.path.insert(0,'/workspace/uwb-shufflenetv2-x1-adam100-recovery45-20261103');from stream_archive import replace_verified_download;print(replace_verified_download("+JSON.stringify(base+"/"+c.name)+","+JSON.stringify(down.path)+","+c.bytes+","+JSON.stringify(c.sha256)+"))";
const replaced=await tools.exec_command({cmd:"python -c '"+replacepy.replaceAll("'","'\\''")+"'",max_output_tokens:200});if(replaced.exit_code!==0||replaced.output.trim()!==base+"/"+c.name)throw Error('Identical raw download transport replacement failed');
const row={...up,path:base+"/"+c.name,remote_raw_bytes_replaced_identical_transport:true,metadata:{shared:false,parent_ids:m.parent_ids,mime_type:m.mime_type,size_bytes:Number(m.size),permissions:m.permissions.map(x=>({role:x.role,type:x.type}))},raw_fetch:true,include_base64:false,download_verified:true};
await tools.apply_patch("*** Begin Patch\n*** Add File: "+base+"/READBACK_"+String(i).padStart(3,"0")+".json\n+"+JSON.stringify(row,null,2).replace(/\n/g,"\n+")+"\n*** End Patch");return{i,verified:true};
}catch(e){const row={i,name:c.name,file_id,stage,error:String(e).replace(/https?:\/\/\S+/g,"[redacted URL]")};await tools.apply_patch("*** Begin Patch\n*** Add File: "+base+"/ERROR_"+String(i).padStart(3,"0")+"_"+Date.now()+".json\n+"+JSON.stringify(row,null,2).replace(/\n/g,"\n+")+"\n*** End Patch");throw Error(JSON.stringify(row));}
}
