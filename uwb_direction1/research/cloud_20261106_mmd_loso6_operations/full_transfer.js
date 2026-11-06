async (index) => {
 const base=load('Adam100ArchiveDir'),c=load('Adam100ArchiveManifest').chunks[index];
 const quote=s=>"'"+s.replaceAll("'","'\\''")+"'";
 const clean=s=>String(s).replace(/https?:\/\/\S+/g,'[redacted URL]').replace(/Bearer\s+\S+/gi,'[redacted bearer]');
 if(base!=='/workspace/uwb-mmd-loso6-20261106/full_backup'||!Number.isInteger(index)||index<0)throw Error('Unexpected archive/index');
 const suffix=String(index).padStart(3,'0');
 const save=async(name,value)=>tools.apply_patch('*** Begin Patch\n*** Add File: '+base+'/'+name+'\n+'+JSON.stringify(value,null,2).replace(/\n/g,'\n+')+'\n*** End Patch');
 const inspect=async()=>{
  const py='from pathlib import Path;import json,hashlib;b=Path('+JSON.stringify(base)+');i='+index+';s=str(i).zfill(3);m=json.loads((b/"MANIFEST.json").read_text());c=m["chunks"][i];assert c["name"]=='+JSON.stringify(c.name)+' and c["sha256"]=='+JSON.stringify(c.sha256)+' and c["bytes"]=='+c.bytes+';u=b/("UPLOAD_"+s+".json");r=b/("READBACK_"+s+".json");up=json.loads(u.read_text()) if u.exists() else None;row=json.loads(r.read_text()) if r.exists() else None;errors=list(b.glob("ERROR_"+s+"_*.json"));assert not up or all(up[k]==c[k] for k in ["name","bytes","sha256"]);assert not row or (all(row[k]==c[k] for k in ["name","bytes","sha256"]) and row["raw_fetch"] and row["download_verified"] and not row["include_base64"] and row["metadata"]["shared"] is False and hashlib.sha256((b/c["name"]).read_bytes()).hexdigest()==c["sha256"]);print(json.dumps(dict(verified=row is not None,known=up is not None,error_count=len(errors),retry_marked=(b/("MISSING_ID_RETRY_"+s+"_INVOKED.json")).exists(),absence_files=[q.name for q in b.glob("MISSING_ID_ABSENCE_"+s+"_*.json")])))';
  const r=await tools.exec_command({cmd:'python -c '+quote(py),max_output_tokens:1000});if(r.exit_code!==0)throw Error('Archive/receipt inspection failed');return JSON.parse(r.output);
 };
 const known=async()=>{await eval('('+load('KnownIDRecoverySource').trim()+')')({base,index},tools);return {i:index,verified:true,recovery:'known_ID_raw_readback_without_upload'};};
 try {
  let state=await inspect();if(state.verified)return {i:index,verified:true,recovery:'existing_verified_receipt_reused'};
  if(state.known)return await known();
  let originalError=null;
  if(state.error_count===0){
   try{return await eval('('+load('UploadSource').trim()+')')(index);}
   catch(ex){originalError=clean(ex);notify({event:'private_transfer_error_evidence_preserved',index,error:originalError.slice(0,500)});}
  }
  state=await inspect();if(state.verified)return {i:index,verified:true};
  if(state.known){const result=await known();await save('FULL_RECOVERY_'+suffix+'.json',{index,type:result.recovery,original_error:originalError,upload_retries:0});return result;}
  if(state.retry_marked||state.error_count!==1||state.absence_files.length)throw Error('Existing unknown-ID attempt/evidence requires explicit inspection; no blind retry');
  const folder='12AGMwT99yXCTYBQocQaX901Br-IelmOf';
  for(let round=1;round<=2;round++){
   if(round===2)await new Promise(resolve=>setTimeout(resolve,20000));
   let found=new Set();
   for(const scope of ['parent','global']){
    const query="name = '"+c.name+"' and trashed = false"+(scope==='parent'?" and '"+folder+"' in parents":'');
    const r=await tools.mcp__codex_apps__google_drive_search({special_filter_query_str:query,best_effort_fetch:false,topn:100});const v=r.structuredContent;
    if(r.isError||!Array.isArray(v?.results)||Object.entries(v).some(([k,x])=>/next.*page|more|truncat/i.test(k)&&x))throw Error('Complete exact-name search unavailable; no retry');
    const ids=v.results.map(x=>x.id??x.file_id).filter(x=>typeof x==='string');if(ids.length!==v.results.length)throw Error('Unknown search result identity schema');for(const id of ids)found.add(id);
    await save('MISSING_ID_ABSENCE_'+suffix+'_round'+round+'_'+scope+'.json',{round,scope,query,exact_name:c.name,folder_id:folder,results:ids.map(id=>({id})),provider_keys:Object.keys(v),provider_continuation_absent:true,provider_truncation_absent:true,requested_topn:100,is_error:false,observed_unix:Date.now()/1000});
   }
   if(found.size){
    if(found.size!==1)throw Error('Multiple exact-name IDs require explicit inspection; no upload');
    await save('UPLOAD_'+suffix+'.json',{...c,file_id:[...found][0],recovered_from_exactname_search_without_upload:true});const result=await known();await save('FULL_RECOVERY_'+suffix+'.json',{index,type:'existing_unknown_ID_found_by_exact_name_then_raw_verified',upload_retries:0,original_error:originalError});return result;
   }
  }
  const src=load('UploadSource'),start=src.indexOf('const preflight='),end=src.indexOf('\nif(preflight.exit_code',start);if(start<0||end<start)throw Error('Unexpected uploader preflight source');
  const cmd='python /workspace/uwb-mmd-loso6-meta-20261106/authorize_missing_full_upload_once.py '+quote(base)+' '+index;
  const patched=src.slice(0,start)+'const preflight=await tools.exec_command({cmd:'+JSON.stringify(cmd)+',max_output_tokens:500});'+src.slice(end);
  let result;
  try{result=await eval('('+patched.trim()+')')(index);}
  catch(ex){state=await inspect();if(!state.known||state.verified)throw ex;result=await known();}
  await save('FULL_RECOVERY_'+suffix+'.json',{index,type:'one_missing_ID_upload_retry_after_two_complete_exactname_absence_rounds',upload_retries:1,original_error:originalError,raw_readback_verified:true,original_error_and_search_evidence_retained:true});
  return {...result,recovery:'one_absence_guarded_missing_ID_retry'};
 }catch(ex){await save('FULL_RECOVERY_FAILURE_'+suffix+'_'+Date.now()+'.json',{index,error:clean(ex).slice(0,1800),whole_archive_verified:false,no_further_automatic_retry:true});throw Error('Part '+index+' recovery failed; inspect preserved evidence before continuing');}
}
