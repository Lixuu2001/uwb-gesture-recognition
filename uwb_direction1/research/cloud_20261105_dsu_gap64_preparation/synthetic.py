"""Independent author comparison and full native DSU checks; synthetic inputs only."""
import ast,copy,random,tempfile,time,json
from pathlib import Path
from common import B,read,ready
import engine as e
from uncertain_statistics import UncertainStatistics
def reference():
 tree=ast.parse(Path(e.c['author_class_reference']).read_text())
 node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DistributionUncertainty')
 namespace=dict(torch=e.torch,nn=e.torch.nn,np=e.np)
 exec(compile(ast.Module(body=[node],type_ignores=[]),e.c['author_class_reference'],'exec'),namespace)
 return namespace['DistributionUncertainty']
def main():
 ready();start=time.monotonic();rng=e.np.random.default_rng(650);x=rng.normal(size=(64,90,567)).astype('float32');y=e.np.arange(64)%12;xx=e.torch.from_numpy(x);target=e.torch.nn.functional.one_hot(e.torch.as_tensor(y),12).float()
 m,o,g=e.make();initial=copy.deepcopy(e.capture(m,o,g,0,0));assert e.semantic(initial['model'])==e.c['matched_initial_model_state_sha256']
 for u in m._dsu_units:u.probability=1.
 m.train();z=m(xx);ours=copy.deepcopy(e.capture(m,o,g,0,0));e.torch.nn.functional.binary_cross_entropy_with_logits(z,target).backward();grad={name:p.grad.clone() for name,p in m.named_parameters()}
 n=e.native.make_model('gap',e.c['seed']);no=e.native.StableLookahead(n.parameters(),lr=.001,weight_decay=.0001);ng=e.torch.Generator().manual_seed(e.c['seed']);units=[reference()(p=1.,eps=1e-6) for _ in range(2)];e.restore(initial,n,no,ng);n.train()
 f=xx.transpose(1,2)
 for i in range(2):f=units[i](n.feature_extractor.instance_encoder[i](f).unsqueeze(-1)).squeeze(-1)
 actual=n.head(f.mean(-1));e.torch.testing.assert_close(z,actual,rtol=0,atol=0);assert e.semantic(ours)==e.semantic(e.capture(n,no,ng,0,0))
 e.torch.nn.functional.binary_cross_entropy_with_logits(actual,target).backward()
 for name,p in n.named_parameters():e.torch.testing.assert_close(grad[name],p.grad,rtol=0,atol=0)
 # Evaluation equals the clean native model exactly and consumes no RNG.
 m.eval();n.eval();before=copy.deepcopy(e.capture(m,o,g,0,0));a=m(xx);b=n(xx);e.torch.testing.assert_close(a,b,rtol=0,atol=0);assert e.semantic(before)==e.semantic(e.capture(m,o,g,0,0))
 plain=e.native.make_model('gap',e.c['seed']);po=e.native.StableLookahead(plain.parameters(),lr=.001,weight_decay=.0001);pg=e.torch.Generator().manual_seed(e.c['seed']);e.restore(initial,plain,po,pg);plain.train();assert not e.torch.allclose(z,plain(xx),rtol=1e-6,atol=1e-7)
 # Small independent numerical Jacobian: gradients must flow through both levels
 # of statistics, unlike MixStyle's detached-statistics helper.
 a=e.torch.from_numpy(rng.normal(size=(3,2,4,1))).requires_grad_();unit=UncertainStatistics(1.);unit.train()
 def fixed_noise(v):e.np.random.seed(83);e.torch.manual_seed(83);return unit(v)
 assert e.torch.autograd.gradcheck(fixed_noise,(a,),eps=1e-6,atol=1e-5,rtol=1e-4)
 ref=reference()(p=1.);e.np.random.seed(19);e.torch.manual_seed(19);v=unit(a);post=copy.deepcopy(e.capture(m,o,g,0,0));e.np.random.seed(19);e.torch.manual_seed(19);w=ref(a);e.torch.testing.assert_close(v,w,rtol=0,atol=0);assert e.semantic(post)==e.semantic(e.capture(m,o,g,0,0))
 e.torch.testing.assert_close(e.torch.autograd.grad(v.square().sum(),a)[0],e.torch.autograd.grad(w.square().sum(),a)[0],rtol=0,atol=0)
 # Eval is a true identity; a skipped training gate consumes just one NumPy draw.
 unit.eval();t0=e.torch.get_rng_state().clone();n0=copy.deepcopy(e.np.random.get_state());assert unit(a) is a;e.np.testing.assert_array_equal(n0[1],e.np.random.get_state()[1]);assert n0[2:]==e.np.random.get_state()[2:] and e.torch.equal(t0,e.torch.get_rng_state())
 unit.train();unit.probability=0.;e.np.random.seed(47);assert unit(a) is a;got=e.np.random.get_state();e.np.random.seed(47);e.np.random.random();expected=e.np.random.get_state();e.np.testing.assert_array_equal(got[1],expected[1]);assert got[2:]==expected[2:] and e.torch.equal(t0,e.torch.get_rng_state())
 # Full native lazy optimizer and all RNG survive reconstruction and disk.
 m,o,g=e.make();branches=[]
 for j in range(5):m.train();branches.append(e.step(m,o,x,y,1)['dsu_applied'])
 saved=copy.deepcopy(e.capture(m,o,g,1,5));saved_hash=e.semantic(saved);m.train();left=e.step(m,o,x,y,1);after=copy.deepcopy(e.capture(m,o,g,1,6));assert o.count==6 and o.slow
 n,no,ng=e.make();e.restore(saved,n,no,ng);n.train();right=e.step(n,no,x,y,1);assert left==right and e.semantic(after)==e.semantic(e.capture(n,no,ng,1,6)) and saved_hash==e.semantic(saved)
 with tempfile.TemporaryDirectory() as td:
  path=Path(td)/'native.pt';e.torch.save(after,path);disk=e.torch.load(path,map_location='cpu',weights_only=True);m.train();left=e.step(m,o,x,y,1);last=copy.deepcopy(e.capture(m,o,g,1,7));n,no,ng=e.make();e.restore(disk,n,no,ng);n.train();right=e.step(n,no,x,y,1);assert left==right and e.semantic(last)==e.semantic(e.capture(n,no,ng,1,7))
 n.train();assert e.step(n,no,x[:28],y[:28],1)['forward_batch']==28
 unit=UncertainStatistics(1.);a=e.torch.ones(4,3,5,1,requires_grad=True);z=unit(a);z.sum().backward();assert e.torch.isfinite(z).all() and e.torch.isfinite(a.grad).all()
 q=dict(state='pass',parameters=617612,probability=.5,factor=1.,eps=1e-6,insertion=[0,1],seed=20261029,independent_component_matches_actual_author_logits_BN_gradients_and_allRNG_exact=True,statistics_and_uncertainty_gradients_numerical_Jacobian_pass=True,evaluation_clean_native_exact_no_RNG=True,skipped_gate_NumPy_only_verified=True,style_perturbation_reaches_logits=True,full_native_lazy_optimizer_and_allRNG_disk_restore_exact=True,hooks_reconstructed_from_frozen_code=True,tail28_and_constant_input_finite=True,probability_branches=branches,no_UWB_arrays_or_labels_opened=True,new_UWB_fit_started=False,prior_MixStyle_raw_diagnostic_labels_read=False,users7_8_labels_used=False,wall_seconds=time.monotonic()-start)
 (B/'SYNTHETIC_RESULT.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
if __name__=='__main__':main()
