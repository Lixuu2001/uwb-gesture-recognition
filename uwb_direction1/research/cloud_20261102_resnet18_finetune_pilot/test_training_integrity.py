"""Independent full native SGD equation and affine-geometry checks; synthetic only."""
from pathlib import Path
import unittest,json,sys
sys.path.insert(0,'/workspace/uwb-resnet18-finetune-source-20261102')
from native_finetune import np,torch,make,optimizer,augmented_pixels,preprocess
class Integrity(unittest.TestCase):
 def test_all_parameter_SGD_momentum_weight_decay_equations(self):
  m=make();o=optimizer(m);g=torch.Generator().manual_seed(20261029);rng=np.random.default_rng(20261029);previous={};changed=0
  for update in range(2):
   raw=rng.normal(size=(15,3,90,189)).astype(np.float32);y=torch.arange(15)%12;o.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(augmented_pixels(raw,g)),y);loss.backward();expected={}
   for n,p in m.named_parameters():
    self.assertIsNotNone(p.grad);grad=p.grad.detach().add(p.detach(),alpha=1e-4);buf=grad.clone() if n not in previous else previous[n].mul(.9).add(grad);expected[n]=p.detach().clone().add(buf,alpha=-3e-4);previous[n]=buf
   before={n:p.detach().clone() for n,p in m.named_parameters()};o.step()
   for n,p in m.named_parameters():torch.testing.assert_close(p,expected[n],rtol=0,atol=0);changed+=int(not torch.equal(before[n],p))
  self.assertEqual(int(m.bn1.num_batches_tracked),2);self.assertGreater(changed,100)
 def test_inverse_affine_full_pixels_zero_fill_independent_FP64(self):
  rng=np.random.default_rng(20261029);raw=rng.normal(size=(2,3,90,189)).astype(np.float32);g=torch.Generator().manual_seed(20261029);copy=torch.Generator().manual_seed(20261029);actual=augmented_pixels(raw,g).numpy().astype(np.float64);scales=(.8+.4*torch.rand(2,generator=copy)).numpy();shifts=torch.randint(-32,33,(2,2),generator=copy).numpy();mean=np.array([.485,.456,.406])[None,:,None,None];std=np.array([.229,.224,.225])[None,:,None,None]
  z=preprocess(raw);u=(z*torch.tensor([.229,.224,.225]).reshape(1,3,1,1)+torch.tensor([.485,.456,.406]).reshape(1,3,1,1)).clamp(0,1).numpy().astype(np.float64);ref=np.zeros_like(actual)
  for n in range(2):
   xx=(np.arange(224)-111.5-shifts[n,0])/float(scales[n])+111.5;yy=(np.arange(224)-111.5-shifts[n,1])/float(scales[n])+111.5;x0=np.floor(xx).astype(int);y0=np.floor(yy).astype(int);wx=xx-x0;wy=yy-y0
   for dy,dx,w in [(0,0,(1-wy)[:,None]*(1-wx)[None,:]),(1,0,wy[:,None]*(1-wx)[None,:]),(0,1,(1-wy)[:,None]*wx[None,:]),(1,1,wy[:,None]*wx[None,:])]:
    yi=y0+dy;xi=x0+dx;mask=((yi>=0)&(yi<224))[:,None]&((xi>=0)&(xi<224))[None,:];ref[n]+=u[n,:,np.clip(yi,0,223)[:,None],np.clip(xi,0,223)[None,:]].transpose(2,0,1)*w*mask
  ref=(ref-mean)/std;unit=2**-24;gamma32=32*unit/(1-32*unit);bound=gamma32*(2*224*1.25+2*32*1.25+16)/.224;self.assertLessEqual(float(np.max(np.abs(actual-ref))),bound)
  Path(__file__).with_name('AFFINE_REFERENCE_CHECK.json').write_text(json.dumps(dict(state='pass_synthetic_independent_full_inverse_affine_geometry',pixels=int(actual.size),maximum_error=float(np.max(np.abs(actual-ref))),analytic_gamma32_bound=bound,empirically_fitted_tolerance=False,UWB_labels_read=False),indent=2)+'\n')
if __name__=='__main__':unittest.main(verbosity=2)
