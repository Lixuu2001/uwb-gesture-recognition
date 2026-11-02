"""GPL-3.0-only compatibility wrapper, original author files local-only.

Full documented nonattention generate_model() from gesture_phase_model.py.
Author source recurrent gate is unspecified; this native Keras3 adapter uses
its current sigmoid default, explicitly not legacy Keras2.1.2 hard_sigmoid.
No claim of historical TensorFlow numerical parity or paper grid reproduction.
"""
import ast,copy,contextlib,hashlib,io,json,sys
from pathlib import Path
STATE=Path('/workspace/uwb-litemv-source-probe-20261031');sys.path.insert(0,str(STATE))
from native_litemv import np,torch,keras,capture as base_capture,restore as base_restore,equal
B=Path(__file__).resolve().parent

def check_source():
 for row in json.loads((B/'SOURCE_INDEX.json').read_bytes())['files']:
  if hashlib.sha256((B/'local_author_sources'/row['path']).read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Pinned local author source missing/changed: '+row['path'])

def bind():
 check_source();tree=ast.parse((B/'local_author_sources/gesture_phase_model.py').read_bytes());functions=[copy.deepcopy(x) for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ['generate_model','squeeze_excite_block']];assert len(functions)==2
 class Shape(ast.NodeTransformer):
  count=0
  def visit_Attribute(self,node):
   if node.attr=='_keras_shape':self.count+=1;node.attr='shape'
   return self.generic_visit(node)
 t=Shape();functions=[t.visit(x) for x in functions];assert t.count==1
 ns=dict(Model=keras.Model,MAX_NB_VARIABLES=567,MAX_TIMESTEPS=90,NB_CLASS=12)
 for name in ['Input','Dense','LSTM','multiply','concatenate','Activation','Masking','Reshape','Conv1D','BatchNormalization','GlobalAveragePooling1D','Permute','Dropout']:ns[name]=getattr(keras.layers,name)
 exec(compile(ast.fix_missing_locations(ast.Module(body=functions,type_ignores=[])),'local_author_MLSTM_generate_model_native_binding_shape_attr_only','exec'),ns)
 return ns['generate_model']

def rng_variables(model):
 return [v for v in model.variables if 'seed_generator_state' in v.path]

def capture(model):
 return dict(native=base_capture(model),layer_rng=[dict(path=v.path,value=v.numpy().copy()) for v in rng_variables(model)],module_training_modes={n:bool(v.training) for n,v in model.named_modules()})

def restore(saved,model):
 states=rng_variables(model)
 if [v.path for v in states]!=[x['path'] for x in saved['layer_rng']]:raise ValueError('Wrong layer RNG topology')
 modes=dict(model.named_modules())
 if set(modes)!=set(saved['module_training_modes']):raise ValueError('Wrong model module topology')
 base_restore(saved['native'],model)
 for v,x in zip(states,saved['layer_rng']):v.assign(x['value'])
 for n,v in saved['module_training_modes'].items():modes[n].training=v

def make(seed):
 if type(seed) is not int or not 0<=seed<2**32:raise ValueError('Explicit seed required')
 keras.backend.clear_session();keras.utils.set_random_seed(seed);assert keras.__version__=='3.11.3' and keras.backend.backend()=='torch'
 with contextlib.redirect_stdout(io.StringIO()):model=bind()()
 model.compile(optimizer=keras.optimizers.Adam(learning_rate=.001),loss='categorical_crossentropy',metrics=['accuracy'])
 model._symbolic_build(data_batch=(np.zeros((1,567,90),dtype='float32'),np.zeros((1,12),dtype='float32'),None));capture(model)
 return model
