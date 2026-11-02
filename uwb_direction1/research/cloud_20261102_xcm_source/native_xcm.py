"""Apache-2.0 XCM author AST bound to Keras 3.11.3 / Torch CPU.

Original vendor bytes remain unchanged. Only legacy imports are rebound;
the xcm function AST, all 128 filters and layer defaults are unchanged.
Current backend execution is not historical TensorFlow numerical parity.
"""
import os
os.environ['KERAS_BACKEND']='torch'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='4'
from pathlib import Path
import ast,copy,hashlib,json,random,sys
B=Path(__file__).resolve().parent
sys.path.insert(0,'/workspace/uwb-resnet-keras-screen-20261030/deps')
import numpy as np
import torch
import keras
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

def bind():
    idx=json.loads((B/'SOURCE_INDEX.json').read_bytes())
    for row in idx['files']:
        assert hashlib.sha256((B/'vendor'/row['path']).read_bytes()).hexdigest()==row['sha256']
    tree=ast.parse((B/'vendor/models/xcm.py').read_bytes())
    nodes=[copy.deepcopy(n) for n in tree.body if isinstance(n,ast.FunctionDef)]
    assert len(nodes)==1 and nodes[0].name=='xcm'
    ns={'Model':keras.Model}
    for key in ['Activation','BatchNormalization','concatenate','Dense','GlobalAveragePooling1D','Input','Reshape','Conv1D','Conv2D']:
        ns[key]=getattr(keras.layers,key)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'unchanged_author_xcm_AST_native_import_binding','exec'),ns)
    return ns['xcm']

def make(seed=20261029,shape=(90,567),classes=12):
    assert type(seed) is int and shape[0]>=5
    keras.backend.clear_session();keras.utils.set_random_seed(seed)
    assert keras.__version__=='3.11.3' and keras.backend.backend()=='torch'
    model=bind()(shape,classes,.2)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model

def capture(model):
    from keras.src.random.seed_generator import global_seed_generator
    # First construction of the Keras global generator consumes Python RNG.
    # Materialize it before taking ANY RNG snapshot, as in the validated bridge.
    kg=global_seed_generator().state.numpy().copy()
    return copy.deepcopy(dict(weights=model.get_weights(),optimizer=[v.numpy().copy() for v in model.optimizer.variables],metrics=[[v.numpy().copy() for v in metric.variables] for metric in model.metrics],torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),keras_global_rng=kg))

def restore(saved,model):
    from keras.src.random.seed_generator import global_seed_generator
    model.set_weights(saved['weights']);model.optimizer.build(model.trainable_variables)
    assert len(model.optimizer.variables)==len(saved['optimizer'])
    for v,x in zip(model.optimizer.variables,saved['optimizer']):v.assign(x)
    assert len(model.metrics)==len(saved['metrics'])
    for metric,xs in zip(model.metrics,saved['metrics']):
        assert len(metric.variables)==len(xs)
        for v,x in zip(metric.variables,xs):v.assign(x)
    global_seed_generator().state.assign(saved['keras_global_rng']);torch.set_rng_state(saved['torch_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])

def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'.'+str(i))
    else:assert a==b,(path,a,b)
