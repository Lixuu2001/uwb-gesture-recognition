async function run(source){
const factory=eval("(function(tools,load){return ("+source.trim()+")})");
const cases=[];
for(const kind of ["missing_ID_response_redacted_before_error","oversized_missing_ID_response_bounded","known_ID_persisted_before_metadata_failure"]){
 const events=[];let uploads=0,metadata=0,raw=0;
 const response=kind==="known_ID_persisted_before_metadata_failure"?{structuredContent:{id:"known-upload-id"}}:kind==="oversized_missing_ID_response_bounded"?{isError:true,content:[{type:"text",text:"x".repeat(40000)}]}:{isError:true,content:[{type:"text",text:"provider failed at https://example.invalid/secret Bearer fake-credential"}],structuredContent:{url:"https://example.invalid/private",email:"private@example.invalid"}};
 const mock={
 exec_command:async()=>({exit_code:0,output:""}),
 mcp__codex_apps__google_drive_upload_file:async()=>{uploads++;return response;},
 apply_patch:async p=>{events.push(p);return {};},
 mcp__codex_apps__google_drive_get_file_metadata:async()=>{metadata++;throw Error("mock metadata failure");},
 mcp__codex_apps__google_drive_fetch:async()=>{raw++;throw Error("Unexpected fetch");}
 };
 const load=k=>k==="Adam100ArchiveDir"?"/workspace/mock-no-filesystem-writes":{chunks:[{name:"part000",bytes:7,sha256:"mock"}]};
 let failed=false;try{await factory(mock,load)(0);}catch(e){failed=true;}
 if(!failed||uploads!==1||raw!==0)throw Error("Invalid fail-closed behavior "+kind);
 const known=events.findIndex(p=>p.includes("/UPLOAD_000.json"));
 const missing=events.findIndex(p=>p.includes("/MISSING_ID_RESPONSE_000_"));
 const error=events.findIndex(p=>p.includes("/ERROR_000_"));
 if(error<0)throw Error("Missing failure evidence");
 if(kind==="known_ID_persisted_before_metadata_failure"){
  if(known<0||known>=error||metadata!==1||!events[error].includes("known-upload-id"))throw Error("Known ID persistence ordering regressed");
 }else{
  if(missing<0||missing>=error||known>=0||metadata!==0)throw Error("Missing-ID response not retained first");
  if(events[missing].includes("example.invalid")||events[missing].includes("fake-credential")||events[missing].includes("private@example.invalid"))throw Error("Response evidence not redacted");
  if(kind==="oversized_missing_ID_response_bounded"&&events[missing].length>20000)throw Error("Missing-ID evidence not bounded");
 }
 cases.push({case:kind,passed:true,uploads,metadata_calls:metadata,raw_fetch_calls:raw});
}
return {state:"pass_three_mock_transport_failure_regressions",cases,mocked_only:true,actual_Drive_uploads:0,new_model_fits:0};
}
