// Storage administration only: each listed part gets one first attempt, max three complete pipelines.
// Failures are preserved and deferred; continuing other untouched parts never retries a failed index.
async function firstAttemptQueue(indices,uploadOne,progress=()=>{}) {
 if(!Array.isArray(indices)||!indices.length||indices.length>512||new Set(indices).size!==indices.length||indices.some(i=>!Number.isInteger(i)||i<0))throw Error('Invalid unique first-attempt indices');
 let next=0;const attempts=[];
 const counts=()=>({attempts_completed:attempts.length,verified_parts:attempts.filter(r=>r.status==='fulfilled').length,failed_parts:attempts.filter(r=>r.status==='rejected').length,total:indices.length});
 async function worker(){for(;;){const pos=next++;if(pos>=indices.length)return;const i=indices[pos];
  try{const value=await uploadOne(i);if(value?.i!==i||value?.verified!==true)throw Error('Raw pipeline verification absent');attempts.push({i,status:'fulfilled',value});}
  catch(error){attempts.push({i,status:'rejected',error:String(error).replace(/https?:\/\/\S+/g,'[redacted URL]')});}
  progress(counts());
 }}
 const workers=await Promise.allSettled([worker(),worker(),worker()]);const attempted=new Set(attempts.map(r=>r.i));
 return {...counts(),attempts,unattempted:indices.filter(i=>!attempted.has(i)),workers:workers.map(r=>r.status==='fulfilled'?{status:r.status}:{status:r.status,error:String(r.reason)}),complete:attempts.filter(r=>r.status==='fulfilled').length===indices.length,automatic_retries:0,each_listed_part_first_attempt_only:true};
}
