"""One prospectively bounded native Keras full ResNet CPU cost probe, no UWB."""
import json,time,resource,tempfile
from author_adapter import *
if __name__=='__main__':
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();rng=np.random.RandomState(7)
 with tempfile.TemporaryDirectory() as d:
  holder,model=make(d);x=rng.randn(64,90,567).astype('float32');y=np.eye(12,dtype='float32')[np.arange(64)%12];training=[];evaluation=[]
  for i in range(8):
   t=time.monotonic();r=step(model,x,y);assert np.isfinite(r['loss']);training.append(time.monotonic()-t)
  for i in range(4):
   t=time.monotonic();z=model.predict_on_batch(x);assert z.shape==(64,12) and np.isfinite(z).all();evaluation.append(time.monotonic()-t)
  estimates={str(ep):2*(ep*74*max(training)+(ep*16+16+19)*max(evaluation))+120 for ep in [1500,100]}
  chosen=next((ep for ep in [1500,100] if estimates[str(ep)]<=43200),None)
  result=dict(state='pass' if chosen else 'resource_gate_fail',chosen_epochs=chosen,full_author_architecture=True,parameters=model.count_params(),batch_size=64,source_author_epochs=1500,original_TensorFlow_reproduction=False,explicit_epoch_budget_adaptation=(chosen==100),training_batches_per_epoch=74,all4700records_each_epoch=True,train_last_batch=28,validation_batches=16,diagnostic_batches=19,training_seconds=training,eval_seconds=evaluation,estimated_seconds_by_epoch_budget=estimates,estimate_formula='2*(epochs*74*maxtrain+(epochs*16+16+19)*maxeval)+120',budget_seconds=43200,wall_seconds=time.monotonic()-start,max_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,real_UWB_values_labels_or_fit=False)
 (B/'COST_CHECK.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
