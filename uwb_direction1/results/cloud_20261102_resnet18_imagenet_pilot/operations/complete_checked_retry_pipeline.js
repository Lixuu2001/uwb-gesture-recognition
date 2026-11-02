async(i)=>{
const fn=eval('('+load('ResNet18Complete84Pipeline')+')');
try{return await fn(i);}catch(first){
 const base=load('ResNet18Complete84Dir'),part=load('ResNet18Complete84Manifest').chunks[i];
 // Retry only a retained explicit connector-internal upload failure with no ID.
 // A known ID, later-stage failure or nonempty discovery always requires reuse
 // or inspection; this wrapper must never reupload those automatically.
 const py="from pathlib import Path;import json,hashlib;b=Path("+JSON.stringify(base)+");i="+i+";e=sorted(b.glob('ERROR_'+str(i).zfill(3)+'_*.json'))[-1];p=sorted(b.glob('UPLOAD_RESPONSE_'+str(i).zfill(3)+'_*.json'))[-1];a=json.loads(e.read_bytes());r=json.loads(p.read_bytes());assert a['stage']=='upload' and a['file_id'] is None;assert not (b/('UPLOAD_'+str(i).zfill(3)+'.json')).exists();assert not (b/('READBACK_'+str(i).zfill(3)+'.json')).exists();assert r.get('isError') is True and not r.get('structuredContent');assert any('Mcp error: -32603: Internal error' in x.get('text','') for x in r.get('content',[]));print(json.dumps(dict(retained_response=p.name,response_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),explicit_connector_internal_error=True)))";
 const check=await tools.exec_command({cmd:"python -c '"+py.replaceAll("'","'\\''")+"'",max_output_tokens:500});if(check.exit_code!==0)throw first;
 const evidence=JSON.parse(check.output);
 const search=await tools.mcp__codex_apps__google_drive_search({query:part.name,special_filter_query_str:"'12AGMwT99yXCTYBQocQaX901Br-IelmOf' in parents and trashed = false",topn:100,best_effort_fetch:false});
 const result=search.structuredContent;
 const receipt={i,name:part.name,...evidence,exact_name_search_results:JSON.parse(JSON.stringify(result,(k,v)=>/url|token|secret|authorization|email/i.test(k)?'[redacted]':v)),empty_search_is_not_absence_proof:true,prior_unidentified_object_possible:true,maximum_same_part_retry:1,scientific_training_or_audit_retry:false};
 await tools.apply_patch('*** Begin Patch\n*** Add File: '+base+'/CHECKED_RETRY_'+String(i).padStart(3,'0')+'_'+Date.now()+'.json\n+'+JSON.stringify(receipt,null,2).replace(/\n/g,'\n+')+'\n*** End Patch');
 if(!Array.isArray(result?.results)||result.results.length!==0)throw Error('Discovery requires ID inspection/reuse, no reupload');
 const r=await fn(i);return {...r,recovered_after_retained_internal_error_and_exact_name_search:true};
}
}



