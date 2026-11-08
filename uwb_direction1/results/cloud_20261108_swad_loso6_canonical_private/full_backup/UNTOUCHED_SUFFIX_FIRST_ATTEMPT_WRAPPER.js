async function uploadUntouchedPart(i){
 const base=load("Adam100ArchiveDir");
 if(!Number.isInteger(i)||i<85||i>246)throw Error("Outside reviewed untouched suffix");
 const py="import pathlib,json,datetime;b=pathlib.Path("+JSON.stringify(base)+");i="+i+";assert not(b/('UPLOAD_'+str(i).zfill(3)+'.json')).exists() and not(b/('READBACK_'+str(i).zfill(3)+'.json')).exists();f=(b/('FIRST_ATTEMPT_STARTED_'+str(i).zfill(3)+'.json')).open('x');json.dump(dict(part=i,started=datetime.datetime.now(datetime.timezone.utc).isoformat(),first_attempt_only=True),f);f.close()";
 const q=s=>"'"+s.replaceAll("'","'\\''")+"'";
 const r=await tools.exec_command({cmd:"python -c "+q(py),max_output_tokens:300});
 if(r.exit_code!==0)throw Error("Prior attempt marker or known ID requires reconciliation, never replay");
 const upload=eval("("+load("UploadHelperSource")+")");return upload(i);
}
