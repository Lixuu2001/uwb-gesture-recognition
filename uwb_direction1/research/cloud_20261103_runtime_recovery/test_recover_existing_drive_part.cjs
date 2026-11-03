// Mocked connector/filesystem boundary tests; no real Drive fetch or training.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(__dirname+'/recover_existing_drive_part.js','utf8');
const recover=vm.runInNewContext('('+source+')',{Date,JSON,Number,String,Error});
const config={base:'/workspace/synthetic-recovery-fixture',index:0};
const chunk={name:'virtual.part000',bytes:100,sha256:'a'.repeat(64),file_id:'known-id'};
function fixture(change={}){
  const calls=[],saved=[];let commands=0;
  const meta={id:'known-id',shared:false,size:'100',parent_ids:['12AGMwT99yXCTYBQocQaX901Br-IelmOf'],permissions:[{role:'owner',type:'user'}],mime_type:'application/octet-stream',url:'https://drive.google.com/file/d/known-id/view',...change.meta};
  const api={
    exec_command:async x=>{calls.push('exec');commands++;return commands===1?{exit_code:0,output:JSON.stringify(chunk)}:change.badSHA?{exit_code:1,output:'failed'}:{exit_code:0,output:'verified\n'};},
    apply_patch:async p=>{saved.push(p);return {};},
    mcp__codex_apps__google_drive_get_file_metadata:async()=>{calls.push('metadata');return {structuredContent:meta};},
    mcp__codex_apps__google_drive_fetch:async options=>{calls.push('fetch');assert.equal(options.download_raw_file,true);assert.equal(options.include_base64,false);return change.raw??{structuredContent:{file_uri:{file_id:'sediment://file_synthetic',download_url:'https://secret.example/credential'}}};},
    download_file:async x=>{calls.push('materialize');assert.equal(x.file_id,'file_synthetic');return {path:'/workspace/synthetic-materialized'};}
  };
  return {api,calls,saved};
}
async function refuses(change,pattern,forbidden){
  const f=fixture(change);await assert.rejects(()=>recover(config,f.api),pattern);
  assert(!f.saved.some(x=>x.includes('/READBACK_000.json')));
  if(forbidden)assert(!f.calls.includes(forbidden));
  assert(!f.saved.join('').includes('secret.example'));return f;
}
(async()=>{
  let f=fixture();assert.equal((await recover(config,f.api)).verified,true);assert.deepEqual(f.calls,['exec','metadata','fetch','materialize','exec']);assert(f.saved.some(x=>x.includes('/READBACK_000.json')));assert(!f.saved.join('').includes('secret.example'));
  await refuses({meta:{shared:true}},/Metadata\/privacy/,'fetch');
  await refuses({meta:{size:'101'}},/Metadata\/privacy/,'fetch');
  f=await refuses({raw:{isError:true,content:[{type:'text',text:'MCP request timed out https://secret.example/token Bearer abc123'}]}},/Raw provider failure/,'materialize');assert(!f.saved.join('').includes('abc123'));
  await refuses({raw:{structuredContent:{}}},/Raw download reference missing/,'materialize');
  await refuses({badSHA:true},/Materialized bytes\/SHA/);
  f=fixture({raw:{file_uri:{file_id:'sediment://file_synthetic'}}});assert.equal((await recover(config,f.api)).verified,true);
  assert(!source.includes('upload_file('));console.log('PASS 7 mocked existing-ID recovery cases; no upload/ACL mutation/false readback');
})().catch(e=>{console.error(e);process.exitCode=1;});
