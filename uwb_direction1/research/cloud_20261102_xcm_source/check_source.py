"""Synthetic source-binding and continuation checks; no UWB arrays/labels."""
import contextlib,io,json,tempfile
from pathlib import Path
from native_xcm import np,torch,keras,make,capture,restore,equal,B

def independently_wired(shape=(10,3),classes=3):
    L=keras.layers;t,d=shape;k=int(.2*t);inp=L.Input((t,d,1))
    a=L.Conv2D(128,(k,1),padding='same')(inp)
    a=L.Activation('relu')(L.BatchNormalization()(a))
    a=L.Activation('relu')(L.Conv2D(1,(1,1))(a));a=L.Reshape((t,d))(a)
    b=L.Reshape((t,d))(inp);b=L.Conv1D(128,k,padding='same')(b)
    b=L.Activation('relu')(L.BatchNormalization()(b))
    b=L.Activation('relu')(L.Conv1D(1,1)(b))
    z=L.Conv1D(128,k,padding='same')(L.concatenate([a,b]))
    z=L.Activation('relu')(L.BatchNormalization()(z))
    return keras.Model(inp,L.Dense(classes,activation='softmax')(L.GlobalAveragePooling1D()(z)))

def main():
    rng=np.random.default_rng(37);x=rng.normal(size=(4,10,3,1)).astype('float32');y=np.eye(3,dtype='float32')[[0,1,2,0]]
    with contextlib.redirect_stdout(io.StringIO()):m=make(shape=(10,3),classes=3)
    independent=independently_wired();independent.set_weights(m.get_weights())
    # Same learned weights, independently wired published graph, asymmetric even padding.
    np.testing.assert_array_equal(m.predict_on_batch(x),independent.predict_on_batch(x))
    a=torch.tensor(x,requires_grad=True);b=torch.tensor(x,requires_grad=True)
    m(a,training=False)[:,1].sum().backward();independent(b,training=False)[:,1].sum().backward();assert torch.equal(a.grad,b.grad)
    assert np.all(np.isfinite(a.grad.detach().numpy()))
    m.train_on_batch(x,y);state=capture(m);m.reset_metrics();first=m.train_on_batch(x,y);after=capture(m)
    restore(state,m);m.reset_metrics();second=m.train_on_batch(x,y);equal(first,second);equal(after,capture(m))
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'full.keras';m.save(p);loaded=keras.models.load_model(p)
        np.testing.assert_array_equal(m.predict_on_batch(x),loaded.predict_on_batch(x))
        m.reset_metrics();loaded.reset_metrics();equal(m.train_on_batch(x,y),loaded.train_on_batch(x,y))
        equal(m.get_weights(),loaded.get_weights());equal([v.numpy() for v in m.optimizer.variables],[v.numpy() for v in loaded.optimizer.variables])
    with contextlib.redirect_stdout(io.StringIO()):full=make()
    assert tuple(full.input_shape)==(None,90,567,1) and tuple(full.output_shape)==(None,12)
    counts=[]
    for layer in full.layers:
        if isinstance(layer,(keras.layers.Conv1D,keras.layers.Conv2D)):
            counts.append(dict(name=layer.name,kernel_size=list(layer.kernel_size),filters=layer.filters))
    # Functional graphs order parallel branches topologically, not source order.
    byname={c['name']:c for c in counts}
    expected={'2D':(128,[18,1]),'2D_Reduced':(1,[1,1]),'1D':(128,[18]),'1D_Reduced':(1,[1]),'1D_Final':(128,[18])}
    assert set(byname)==set(expected)
    for key,(filters,kernel) in expected.items():
        assert byname[key]['filters']==filters and byname[key]['kernel_size']==kernel
    report=dict(state='passed_source_AST_binding_independent_toy_forward_input_gradients_native_update_restore_and_serialized_Adam_continuation',parameters=full.count_params(),trainable_parameters=sum(int(np.prod(v.shape)) for v in full.trainable_variables),input_shape=list(full.input_shape),output_shape=list(full.output_shape),convolutions=counts,filters=128,window_fraction=.2,kernel_length=18,epochs=100,batch_size=32,threads=torch.get_num_threads(),optimizer=full.optimizer.get_config(),toy_shape=list(x.shape),full_numerical_UWB_shape_forward=False,actual_UWB_values_labels_fit=False,historical_TensorFlow_numerical_parity=False)
    (B/'SOURCE_CHECK.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
