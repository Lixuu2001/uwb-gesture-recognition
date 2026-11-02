"""GPL-3.0-only. Prepared zero-information control; no UWB loading or runner.

Original full H-Inception author methods/binding are pinned and unedited.
Zero control adds one weightless operation after its 17 fixed responses/ReLU.
This is a causal ablation preparation, not standard InceptionTime or novelty.
"""
from pathlib import Path
import hashlib,sys
SOURCE=Path('/workspace/uwb-hinception-source-probe-20261031')
sys.path.insert(0,str(SOURCE))
import native_hinception as author
from native_hinception import keras,np,torch,capture,restore,equal
EXPECTED_BINDING_SHA='41b23d064abfbc67f16bcaaeca692a81935e8f29fbca497118dca82c2d42adcf'
@keras.saving.register_keras_serializable(package='HInceptionControls')
class ZeroFixedBranch(keras.layers.Layer):
 def __init__(self,**kwargs):
  kwargs.setdefault('trainable',False);super().__init__(**kwargs);assert self.trainable is False
 def call(self,inputs):return keras.ops.zeros_like(inputs)
 def compute_output_shape(self,input_shape):return input_shape
def check_binding():
 # Filled against the existing frozen native source; no edits to that helper.
 assert hashlib.sha256((SOURCE/'native_hinception.py').read_bytes()).hexdigest()==EXPECTED_BINDING_SHA
 author.check_source()
def make(output,seed,mode):
 if mode not in ['full','zero_fixed_responses']:raise ValueError('Unknown prepared intervention')
 check_binding()
 if mode=='full':
  holder,model,bindings=author.make(output,seed,567)
 else:
  keras.backend.clear_session();keras.utils.set_random_seed(seed);cls,bindings=author.bind_class(567);original=cls.hybrid_layer
  def blocked(self,*args,**kwargs):
   responses=original(self,*args,**kwargs)
   return ZeroFixedBranch(name='zero_fixed_responses')(responses)
  cls.hybrid_layer=blocked
  holder=cls(output_directory=str(output)+'/',length_TS=90,n_classes=12);model=holder.model
  model._symbolic_build(data_batch=(np.zeros((1,90,567),dtype='float32'),np.zeros((1,12),dtype='float32'),None));capture(model)
 assert model.count_params()==851434 and sum(int(np.prod(v.shape)) for v in model.trainable_variables)==601006 and len(model.trainable_variables)==50 and len(model.optimizer.variables)==102
 assert holder.depth==6 and holder.n_filters==32 and holder.bottleneck_size==32 and holder.epochs==1500 and holder.batch_size==64
 assert len([x for x in model.layers if isinstance(x,keras.layers.Conv1D) and not x.trainable])==17
 return holder,model,dict(bindings,mode=mode,added_weightless_zero_layer=mode!='full')
def full_state(model):
 return dict(native=capture(model),module_training_modes={n:bool(m.training) for n,m in model.named_modules()})
def restore_full(saved,model):
 modes=dict(model.named_modules())
 if set(modes)!=set(saved['module_training_modes']):raise ValueError('Different model graph/modes')
 restore(saved['native'],model)
 for n,v in saved['module_training_modes'].items():modes[n].training=v
