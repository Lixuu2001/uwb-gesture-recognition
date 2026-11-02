import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,time,json
from guards import ready,require_inference_complete,read,sha
from artifacts import write
B=Path(__file__).resolve().parent
def main():
 term=require_inference_complete();c=ready();assert not (B/'COMPLETE_AUDIT.json').exists();out=B/'run01';start=time.monotonic();hashes={}
 for n,d in read(B/'FROZEN_HASHES.json').items():assert sha(B/n)==d,n;hashes[str(B/n)]=d
 for n,d in read(out/'FILES.json').items():assert sha(out/n)==d,n;hashes[str(out/n)]=d
 from engine import np,torch,model,replay
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
 from data import load
 ids,users,ordered,maps=load(c);identity=read(out/'IDENTITY.json');assert identity==dict(protocol_sha256=sha(B/'PROTOCOL.json'),frozen_sha256=sha(B/'FROZEN_HASHES.json'),selected_checkpoint_sha256=sha(c['selected_checkpoint']),python=sys.version,torch=torch.__version__,numpy=np.__version__)
 parts=read(out/'split.json')
 for role,ix in ordered.items():np.testing.assert_array_equal(parts[role],ids[ix])
 raw=np.load(Path(c['data'])/'cache/X_clutterremoved.npy',mmap_mode='r')
 for i in range(0,len(ids),64):
  a=np.array(raw[ids[i:i+64]],copy=True);r=np.sqrt(np.mean(a*a,axis=(2,3),keepdims=True,dtype=np.float32));ref=(a/np.maximum(r,np.float32(1e-8))).transpose(0,2,1,3).reshape(len(a),90,567);np.testing.assert_allclose(ref,maps[i:i+64],rtol=2e-5,atol=2e-6);f=a.astype('float64');r64=np.sqrt(np.mean(f*f,axis=(2,3),keepdims=True));ref64=(f/np.maximum(r64,1e-8)).transpose(0,2,1,3).reshape(len(a),90,567);np.testing.assert_allclose(ref,ref64,rtol=2e-5,atol=2e-6)
 m=model(c['seed'],c['selected_checkpoint']);r=replay(c,out,m,ids,ordered,maps,identity,c['reference_predictions']);assert r['samples_replayed']==200
 for p,d in hashes.items():assert sha(p)==d,p
 r.update(state='pass',all6900source_raw_RMS_cache_FP64_verified=True,artifact_hashes=hashes,inference_termination=term,users7_8_labels_used=False,independent_validation=False,novel_contribution_validated=False,audit_wall_seconds=time.monotonic()-start,auditor_sha256=sha(__file__));write(B/'COMPLETE_AUDIT.json',r);print(json.dumps({k:v for k,v in r.items() if k not in ['artifact_hashes','inference_termination']}))
if __name__=='__main__':main()
