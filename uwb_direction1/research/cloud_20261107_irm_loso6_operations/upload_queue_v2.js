// One callback must perform one complete upload/metadata/raw/materialize/SHA pipeline.
// Verify the destination's owner-only metadata first. This queue never retries failures.
// Inspect and persist the returned receipt; only complete=true permits archive verification.
async function transferQueue(total,uploadOne,progress=()=>{}) {
 if(!Number.isInteger(total)||total<1||total>512)throw Error("Invalid bounded chunk count");
 let next=0;const attempts=[];
 const counts=()=>({attempts_completed:attempts.length,verified_parts:attempts.filter(r=>r.status==="fulfilled").length,failed_parts:attempts.filter(r=>r.status==="rejected").length,total});
 async function worker(){for(;;){const i=next++;if(i>=total)return;
  try{const value=await uploadOne(i);if(value?.i!==i||value?.verified!==true)throw Error("Pipeline did not confirm raw verification");
   attempts.push({i,status:"fulfilled",value});
  }catch(error){attempts.push({i,status:"rejected",error:String(error).replace(/https?:\/\/\S+/g,"[redacted URL]")});progress(counts());return;}
  progress(counts());
 }}
 const workers=await Promise.allSettled([worker(),worker(),worker()]);
 const attempted=new Set(attempts.map(r=>r.i));const unattempted=Array.from({length:total},(_,i)=>i).filter(i=>!attempted.has(i));
 return {...counts(),attempts,unattempted,workers:workers.map(r=>r.status==="fulfilled"?{status:r.status}:{status:r.status,error:String(r.reason)}),complete:attempts.filter(r=>r.status==="fulfilled").length===total};
}

