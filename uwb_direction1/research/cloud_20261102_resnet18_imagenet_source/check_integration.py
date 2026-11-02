"""Full public weights/architecture and independent driver on synthetic maps."""
from pathlib import Path
import json,pickle,copy,warnings
from native_resnet18 import np,torch,F,B,make,preprocess,extract,make_head,encoder_digest
from sklearn.exceptions import ConvergenceWarning
rng=np.random.default_rng(20261029);x=rng.normal(size=(2,3,90,189)).astype(np.float32);x[0,0]=1
z=preprocess(x).numpy();xx=x.astype(np.float64);lo=xx.min(axis=(-2,-1),keepdims=True);hi=xx.max(axis=(-2,-1),keepdims=True);u=(xx-lo)/np.maximum(hi-lo,1e-8)
yy=(np.arange(224)+.5)*90/224-.5;xxg=(np.arange(224)+.5)*189/224-.5;y0=np.floor(yy).astype(int);x0=np.floor(xxg).astype(int);wy=yy-y0;wx=xxg-x0;y1=np.clip(y0+1,0,89);x1=np.clip(x0+1,0,188);y0=np.clip(y0,0,89);x0=np.clip(x0,0,188)
ref=(u[:,:,y0[:,None],x0[None,:]]*(1-wy)[:,None]*(1-wx)[None,:]+u[:,:,y1[:,None],x0[None,:]]*wy[:,None]*(1-wx)[None,:]+u[:,:,y0[:,None],x1[None,:]]*(1-wy)[:,None]*wx[None,:]+u[:,:,y1[:,None],x1[None,:]]*wy[:,None]*wx[None,:]);ref=(ref-np.array([.485,.456,.406])[None,:,None,None])/np.array([.229,.224,.225])[None,:,None,None]
np.testing.assert_allclose(z,ref,rtol=2e-5,atol=2e-5)
m=make(True);image=preprocess(x);before=encoder_digest(m)
def conv(layer,t):return F.conv2d(t,layer.weight,layer.bias,layer.stride,layer.padding,layer.dilation,layer.groups)
with torch.inference_mode():
 native=m(image);t=F.max_pool2d(torch.relu(m.bn1(conv(m.conv1,image))),kernel_size=3,stride=2,padding=1)
 for stage in [m.layer1,m.layer2,m.layer3,m.layer4]:
  for block in stage:
   identity=t;t2=torch.relu(block.bn1(conv(block.conv1,t)));t2=block.bn2(conv(block.conv2,t2))
   if block.downsample is not None:identity=block.downsample[1](conv(block.downsample[0],identity))
   t=torch.relu(t2+identity)
 manual=m.fc(F.adaptive_avg_pool2d(t,(1,1)).flatten(1));torch.testing.assert_close(native,manual,rtol=0,atol=0)
assert encoder_digest(m)==before
m.fc=torch.nn.Identity();f,h=extract(m,x,2);f2,h2=extract(m,x,2);np.testing.assert_array_equal(f,f2);q=pickle.loads(pickle.dumps(m,protocol=5));assert encoder_digest(q)==encoder_digest(m);fr,_=extract(q,x,2);np.testing.assert_array_equal(f,fr)
features=rng.normal(size=(72,512));labels=np.arange(72)%12;head=make_head()
with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);head.fit(features,labels)
assert head.n_iter_<10000;other=pickle.loads(pickle.dumps(head,protocol=5));np.testing.assert_array_equal(other.coef_,head.coef_);np.testing.assert_array_equal(other.intercept_,head.intercept_);np.testing.assert_array_equal(other.predict(features),head.predict(features))
result=dict(state='passed_full_official_ResNet18_ImageNet_V1_frozen_encoder_linear_SVM_synthetic_integration',official_vision_commit='7a9db90e9aadea3e3e1fec35969601167e9efd4f',original_full_parameters=11689512,frozen_encoder_parameters=sum(p.numel() for p in m.parameters()),embedding_width=512,original_AST_classes_unmodified=True,all_official1000class_weights_loaded_strict=True,preprocessing_independent_FP64_reference_rtol=2e-5,preprocessing_independent_FP64_reference_atol=2e-5,independent_functional_residual_driver_logits_exact=True,encoder_all_parameters_frozen_BN_eval_unchanged=True,repeated_samebatch_encoder_features_exact=True,encoder_pickle_all_state_features_exact=True,linearSVC72case_12class_converged_and_restored_exact=True,C=1,seed=20261029,batch_size=16,torch_threads=4,original_radar_paper_finetuning_reproduction=False,no_UWB_values_labels_accessed=True,formal_cost_not_started=True,accuracy_or_novelty_claim=False)
(B/'INTEGRATION_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
