"""Controlled synthetic validation trajectory exercises actual producer/auditor stop path."""
from pathlib import Path
from unittest.mock import patch
import tempfile,json
from engine import np,torch
from run_pilot import train_arrays
from audit_complete import replay_arrays
from artifacts import write
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);g=np.random.default_rng(45);maps=(.1*g.normal(size=(9,90,567))).astype('float32');ids=np.arange(9,dtype='int64');ix=dict(train=np.arange(3),validation=np.arange(3,6),diagnostic=np.arange(6,9));ordered={k:ix[k] for k in ['validation','diagnostic']};c=dict(epochs=10,batch_size=32,seed=20261029);x=maps[:3].copy();vx=maps[3:6].copy();y=np.array([0,1,2]);vy=y.copy();identity=dict(scope='synthetic_controlled_validation_only')
 seq=[(1.,.9)]*2+[(1.1,.8)]*6
 with tempfile.TemporaryDirectory() as d:
  out=Path(d)
  with patch('engine.validate',side_effect=seq):r=train_arrays(c,out,identity,ids,ix,ordered,maps,x,y,vx,vy,lambda **kw:None)
  assert r['epochs']==4 and r['steps']==4 and r['selected_epoch']==1 and r['early_stop']
  with patch('engine.validate',side_effect=seq):a=replay_arrays(c,out,identity,ids,ix,ordered,maps,x,y,vx,vy)
  assert a['epochs_replayed']==4 and a['selected_epoch']==1
 write(Path(__file__).with_name('EARLYSTOP_PATH_CHECK.json'),dict(state='pass_controlled_synthetic_early_stop_producer_auditor',validation_observer_mocked_for_boundary_only=True,epochs=4,updates=4,selected_epoch=1,source_patience3_terminal_path_exact=True,real_UWB_values_labels_fit=False));print('Controlled synthetic4epoch producer/full auditor passed; no real validation values')
if __name__=='__main__':main()
