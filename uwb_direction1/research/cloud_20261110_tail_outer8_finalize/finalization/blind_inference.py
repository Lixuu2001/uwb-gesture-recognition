"""One measured test-input pass per saved arm; assert frozen scores, no truth."""
from common import *
def main():
 m=verify_manifest();resource_check()
 import core as e
 from role_data import RealData
 torch=e.torch;rows=[]
 for fold in m['folds']:
  h=fold['held'];data=RealData(h,production_release=ROOT/'PRODUCTION_RELEASE.json')
  for arm in ARMS:
   resource_check();a=fold['arms'][arm];wt=time.monotonic();ct=time.process_time()
   model=e.native.make_model('gap',20261029);saved=torch.load(a['model'],map_location='cpu',weights_only=True);model.load_state_dict(saved['model']);model.eval()
   load_cost=dict(wall=time.monotonic()-wt,CPU=time.process_time()-ct)
   with np.load(a['prediction'],allow_pickle=False) as z:ids=z['ids'].copy();expected=z['scores'].copy()
   forward_wall=forward_CPU=combined_wall=combined_CPU=0.;zs=[]
   with torch.inference_mode():
    for i in range(0,len(ids),64):
     resource_check();w=time.monotonic();c=time.process_time();x=torch.from_numpy(data.batch('test',ids[i:i+64]));fw=time.monotonic();fc=time.process_time();z=model(x).numpy();forward_wall+=time.monotonic()-fw;forward_CPU+=time.process_time()-fc;combined_wall+=time.monotonic()-w;combined_CPU+=time.process_time()-c;zs.append(z)
   np.testing.assert_array_equal(np.concatenate(zs),expected)
   assert not data.labels.reads and data._truth is None
   rows.append(dict(held=h,arm=arm,n=len(ids),batch=64,threads=1,model_load=load_cost,model_forward_wall_seconds=forward_wall,model_forward_CPU_seconds=forward_CPU,RMS_plus_forward_wall_seconds=combined_wall,RMS_plus_forward_CPU_seconds=combined_CPU,bitwise_saved_output_verified=True,gesture_truth_read=False))
 write(HERE/'INFERENCE_COST.json',dict(state='all24_saved_models_blind_inference_matches_frozen_scores',rows=rows,held_labels_read=False,passes_per_model=1,warmup_repetitions=0,device='CPU',timing_is_single_pass_not_a_latency_distribution=True,sourceBN_already_in_frozen_models=True))
 print('All24 frozen test outputs recomputed bitwise; zero gesture-label value access')
if __name__=='__main__':main()
