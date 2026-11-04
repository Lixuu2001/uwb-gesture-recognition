const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),cp=require('node:child_process'),assert=require('node:assert/strict');
let tools,load;const upload=eval('('+fs.readFileSync('/workspace/uwb-softvote-private-upload.js','utf8').trim()+')');
function fixture(options={}){
 const b=fs.mkdtempSync('/workspace/uploader-guard-'),part={name:'synthetic.part000',bytes:3,sha256:'synthetic_sha'};let calls=0,orderOK=false;
 load=k=>k==='Adam100ArchiveDir'?b:{chunks:[part]};
 tools={exec_command:async({cmd})=>{const r=cp.spawnSync(cmd,{shell:'/bin/bash',encoding:'utf8'});return{exit_code:r.status,output:r.stdout+r.stderr};},
 mcp__codex_apps__google_drive_upload_file:async()=>{calls++;return options.noID?{isError:true,content:[{type:'text',text:'synthetic unknown provider outcome'}]}:{structuredContent:{id:'synthetic_known_ID'},content:[{type:'text',text:'x'.repeat(options.large?30000:1)}]};},
 apply_patch:async patch=>{const rows=patch.split('\n'),file=rows.find(x=>x.startsWith('*** Add File: ')).slice(14),data=rows.filter(x=>x.startsWith('+')).map(x=>x.slice(1)).join('\n')+'\n';
  if(file.includes('/UPLOAD_RESPONSE_')&&!file.includes('_WARNING_')){orderOK=fs.existsSync(b+'/UPLOAD_000.json');assert(orderOK);if(options.failResponse)throw Error('synthetic response persistence failure');}
  fs.writeFileSync(file,data,{flag:'wx'});return{};},
 mcp__codex_apps__google_drive_get_file_metadata:async()=>({structuredContent:{shared:false,permissions:[{role:'owner',type:'user'}],size:'3',parent_ids:['12AGMwT99yXCTYBQocQaX901Br-IelmOf'],mime_type:'application/octet-stream',url:'https://drive.google.com/file/d/synthetic/view'}}),
 mcp__codex_apps__google_drive_fetch:async()=>{throw Error('synthetic stop after persistence/privacy stages; no real fetch');}};
 return{b,calls:()=>calls,orderOK:()=>orderOK};
}
(async()=>{
 let x=fixture({failResponse:true});await assert.rejects(()=>upload(0));assert.equal(JSON.parse(fs.readFileSync(x.b+'/UPLOAD_000.json')).file_id,'synthetic_known_ID');assert(x.orderOK());assert(fs.readdirSync(x.b).some(n=>n.startsWith('UPLOAD_RESPONSE_PERSIST_WARNING_')));await assert.rejects(()=>upload(0));assert.equal(x.calls(),1);fs.rmSync(x.b,{recursive:true});
 x=fixture({large:true});await assert.rejects(()=>upload(0));const n=fs.readdirSync(x.b).find(n=>n.startsWith('UPLOAD_RESPONSE_'));assert(JSON.parse(fs.readFileSync(x.b+'/'+n)).bounded_response);assert(fs.statSync(x.b+'/'+n).size<16000);assert(x.orderOK());fs.rmSync(x.b,{recursive:true});
 x=fixture({noID:true});await assert.rejects(()=>upload(0));assert(!fs.existsSync(x.b+'/UPLOAD_000.json'));assert(fs.readdirSync(x.b).some(n=>n.startsWith('ERROR_')));await assert.rejects(()=>upload(0));assert.equal(x.calls(),1);fs.rmSync(x.b,{recursive:true});
 process.stdout.write('PASS 3 mocked transport guards: durable known ID before optional response, bounded response, unknown-outcome retry blocked. No real upload/fetch/model update.\n');
})().catch(e=>{process.stderr.write(e.stack+'\n');process.exitCode=1;});
