"""Full structural variant and independent math/dropout-RNG replay; no UWB."""
import json,tempfile,time
from native_mlstm import *

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic();m=make(7)
 conv=[l for l in m.layers if isinstance(l,keras.layers.Conv1D)];bn=[l for l in m.layers if isinstance(l,keras.layers.BatchNormalization)];dense=[l for l in m.layers if isinstance(l,keras.layers.Dense)];lstm=next(l for l in m.layers if isinstance(l,keras.layers.LSTM));drop=next(l for l in m.layers if isinstance(l,keras.layers.Dropout))
 assert [(l.filters,l.kernel_size,l.use_bias) for l in conv]==[(128,(8,),True),(256,(5,),True),(128,(3,),True)]
 assert len(bn)==3 and all(l.momentum==.99 and l.epsilon==.001 for l in bn);assert [l.units for l in dense]==[8,128,16,256,12];assert all(l.use_bias is False for l in dense[:-1]);assert lstm.units==8 and lstm.recurrent_activation.__name__=='sigmoid' and lstm.unit_forget_bias and drop.rate==.8
 assert m.input_shape==(None,567,90) and m.output_shape==(None,12);assert len(rng_variables(m))>=1
 g=np.random.RandomState(17);x=(.1*g.randn(2,567,90)).astype('float32');x[0,10,:]=0;x[1,:5,:]=0;y=np.eye(12,dtype='float32')[[0,1]]
 # Independent full567-step masked LSTM reference using the actual native gates.
 kernel,recurrent,bias=lstm.get_weights();h=np.zeros((2,8),dtype='float32');c=h.copy()
 sigmoid=lambda z:1/(1+np.exp(-z))
 for i in range(567):
  gates=x[:,i]@kernel+h@recurrent+bias;gi,gf,gc,go=np.split(gates,4,axis=1);nc=sigmoid(gf)*c+sigmoid(gi)*np.tanh(gc);nh=sigmoid(go)*np.tanh(nc);mask=np.any(x[:,i]!=0,axis=1)[:,None];h=np.where(mask,nh,h);c=np.where(mask,nc,c)
 got=keras.Model(m.inputs,lstm.output).predict_on_batch(x);np.testing.assert_allclose(got,h,rtol=2e-5,atol=2e-6)
 # Both original SE blocks versus independent NumPy GAP/Dense/ReLU/sigmoid.
 activations=[l for l in m.layers if isinstance(l,keras.layers.Activation)]
 multiplies=[l for l in m.layers if isinstance(l,keras.layers.Multiply)]
 assert len(activations)==3 and len(multiplies)==2
 for i,layer in enumerate(multiplies):
  a=keras.Model(m.inputs,activations[i].output).predict_on_batch(x);out=keras.Model(m.inputs,layer.output).predict_on_batch(x);w1=dense[2*i].get_weights()[0];w2=dense[2*i+1].get_weights()[0];gate=sigmoid(np.maximum(a.mean(axis=1)@w1,0)@w2);np.testing.assert_allclose(out,a*gate[:,None,:],rtol=2e-5,atol=2e-6)
 # Dropout consumes layer SeedGenerator state separately from global RNG.
 before=capture(m)
 for _ in range(3):m.train_on_batch(x,y)
 saved=capture(m);assert any(not np.array_equal(a['value'],b['value']) for a,b in zip(before['layer_rng'],saved['layer_rng']))
 for _ in range(2):m.train_on_batch(x,y)
 expected=capture(m)
 with tempfile.TemporaryDirectory() as d:
  torch.save(saved,Path(d)/'synthetic_state.pt');loaded=torch.load(Path(d)/'synthetic_state.pt',weights_only=False);m2=make(7);restore(loaded,m2);equal(loaded,capture(m2))
  for _ in range(2):m2.train_on_batch(x,y)
  equal(expected,capture(m2));p=m2.predict_on_batch(x);assert p.shape==(2,12) and np.isfinite(p).all();np.testing.assert_allclose(p.sum(1),1,rtol=1e-5,atol=1e-6)
  m2.save(Path(d)/'synthetic_native.keras');native=keras.models.load_model(Path(d)/'synthetic_native.keras');equal(m2.get_weights(),native.get_weights());equal([v.numpy() for v in m2.optimizer.variables],[v.numpy() for v in native.optimizer.variables]);native_rng_equal=all(np.array_equal(a.numpy(),b.numpy()) for a,b in zip(rng_variables(m2),rng_variables(native)))
 result=dict(state='pass_full_documented_MLSTM_FCN8_structure_and_native_synthetic_only',source_commit='85100ccfec7ee08106b3a25906a0c9c707e1b7ab',source_model='gesture_phase_model.py generate_model',source_file_blob='6d425bda53e84b0c9d6c620a016f29b903807ae6',full_conv128_256_128_k8_5_3_SE16_LSTM8_Dropout08_masking_preserved=True,total_parameters=m2.count_params(),trainable_parameters=sum(int(np.prod(v.shape)) for v in m2.trainable_variables),trainable_variable_tensors=len(m2.trainable_variables),native_Adam_variables=len(m2.optimizer.variables),layer_RNG_variables=len(rng_variables(m2)),native_recurrent_gate='Keras3 sigmoid; differs from legacy Keras2.1.2 default hard_sigmoid',independent_NumPy567step_masked_LSTM_reference=True,independent_NumPy_both_SE_blocks=True,restore_after3_next2_model_Adam_BN_metrics_global_and_layer_RNG_module_modes_exact=True,native_keras_weights_optimizer_reload_exact=True,native_keras_layer_RNG_reload_equal=native_rng_equal,native_keras_alone_claimed_as_training_resume=False,source_shape_attr_replacements=1,original_source_bytes_changed=False,original_TensorFlow_numeric_parity_asserted=False,paper_LSTM8_64_128_grid_reproduced=False,paper250epochs_or_sourcegesture1000epochs_run=False,formal_CPU_cost_or_real_UWB_configuration_selected=False,real_UWB_values_labels_fit=False,synthetic_batch_size=2,synthetic_seed=7,wall_seconds=time.monotonic()-start)
 (B/'SOURCE_PROBE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
