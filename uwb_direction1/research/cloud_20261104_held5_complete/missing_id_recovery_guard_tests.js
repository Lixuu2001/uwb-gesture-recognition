async function test(recoverySource){
const recovery=eval("("+recoverySource+")"),cases=[];
for(const kind of ["prior_retry_blocks_another_upload","existing_exact_match_blocks_upload","failed_search_blocks_upload"]){
 let uploads=0,fetches=0,patches=0,searches=0;
 const api={
  exec_command:async()=>kind==="prior_retry_blocks_another_upload"?{exit_code:1,output:"prior retry marker"}:{exit_code:0,output:JSON.stringify({name:"part000",bytes:7,sha256:"mock"})},
  mcp__codex_apps__google_drive_search:async()=>{searches++;return kind==="existing_exact_match_blocks_upload"?{isError:false,structuredContent:{results:[{id:"existing-file"}]}}:{isError:true,structuredContent:{results:[]}};},
  mcp__codex_apps__google_drive_upload_file:async()=>{uploads++;throw Error("Unexpected upload");},
  apply_patch:async()=>{patches++;},
 };
 let rejected=false;try{await recovery("/workspace/mock-no-filesystem-writes",0,api,async()=>{fetches++;});}catch{rejected=true;}
 if(!rejected||uploads||fetches||patches)throw Error("Unsafe recovery "+kind);
 cases.push({case:kind,passed:true,uploads,raw_fetches:fetches,writes:patches,searches});
}
return {state:"pass_three_missing_ID_recovery_guards",mocked_only:true,actual_Drive_uploads:0,cases};
}
