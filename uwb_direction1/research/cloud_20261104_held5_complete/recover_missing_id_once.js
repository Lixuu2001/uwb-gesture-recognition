async function recoverOnce(base,index,api,knownRecovery){
const quote=s=>"'"+s.replaceAll("'","'\\''")+"'";
const clean=x=>JSON.parse(JSON.stringify(x,(k,v)=>/url|token|secret|authorization|email/i.test(k)?"[redacted]":typeof v==="string"?v.replace(/https?:\/\/\S+/g,"[redacted URL]").replace(/Bearer\s+\S+/gi,"[redacted bearer]"):v));
const suffix=String(index).padStart(3,"0");
const save=async(name,x)=>api.apply_patch("*** Begin Patch\n*** Add File: "+base+"/"+name+"\n+"+JSON.stringify(x,null,2).replace(/\n/g,"\n+")+"\n*** End Patch");
if(!Number.isInteger(index)||index<0||!base.startsWith("/workspace/"))throw Error("Invalid part/root");
const py="from pathlib import Path;import json;b=Path("+JSON.stringify(base)+");s="+JSON.stringify(suffix)+";assert list(b.glob('ERROR_'+s+'_*.json'));assert not (b/('UPLOAD_'+s+'.json')).exists() and not (b/('READBACK_'+s+'.json')).exists();assert not list(b.glob('EXPLICIT_RETRY*'+s+'*.json')) and not (b/('ABSENCE_BEFORE_EXPLICIT_RETRY_'+s+'.json')).exists(),'Prior explicit retry/evidence must be inspected; no second upload';m=json.loads((b/'MANIFEST.json').read_text());print(json.dumps(m['chunks']["+index+"]))";
const pre=await api.exec_command({cmd:"python -c "+quote(py),max_output_tokens:1000});if(pre.exit_code!==0)throw Error("Retry preflight blocked");
const c=JSON.parse(pre.output);if(!/^[A-Za-z0-9_.-]+$/.test(c.name))throw Error("Unsafe part name");
for(const q of ["'12AGMwT99yXCTYBQocQaX901Br-IelmOf' in parents and name = '"+c.name+"' and trashed = false","name = '"+c.name+"' and trashed = false"]){
 const r=await api.mcp__codex_apps__google_drive_search({special_filter_query_str:q,best_effort_fetch:false,topn:100});
 if(r.isError||!Array.isArray(r.structuredContent?.results)||r.structuredContent.results.length||r.structuredContent.next_page_token)throw Error("Absence unproved; inspect/reuse matches, no retry");
}
await save("ABSENCE_BEFORE_EXPLICIT_RETRY_"+suffix+".json",{state:"two_successful_exact_name_provider_searches_returned_no_file_before_one_explicit_retry",name:c.name,observed_at:new Date().toISOString(),folder_filter_results:0,global_exact_name_results:0,provider_reported_no_next_page:true,unlimited_retry_allowed:false});
await save("EXPLICIT_RETRY_STARTED_"+suffix+".json",{index,name:c.name,maximum_explicit_upload_retries:1,observed_at:new Date().toISOString()});
const u=await api.mcp__codex_apps__google_drive_upload_file({file_uri:base+"/"+c.name,file_name:c.name,mime_type:"application/octet-stream",parent_folder_id:"12AGMwT99yXCTYBQocQaX901Br-IelmOf"});
const id=u.structuredContent?.id||u.structuredContent?.file?.id||u.structuredContent?.structuredContent?.id;
if(id)await save("UPLOAD_"+suffix+".json",{...c,file_id:id});
const sanitized=clean(u);const s=JSON.stringify(sanitized);await save("EXPLICIT_RETRY_RESPONSE_"+suffix+".json",s.length>16000?{bounded_response:true,characters:s.length,is_error:u.isError===true,known_ID:id||null}:sanitized);
if(!id)throw Error("Single explicit retry failed; evidence retained, no loop");
const r=await knownRecovery({base,index},api);
await save("EXPLICIT_RETRY_RECOVERED_"+suffix+".json",{state:"one_explicit_retry_after_proven_absence_then_known_ID_private_raw_verified",index,file_id:id,upload_retries:1,known_ID_readback_reuploaded:false,verified:r.verified});
return r;
}
