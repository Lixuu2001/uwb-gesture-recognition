"""UNEXECUTED: requires the restored, hashed Torch CPU environment and weights."""
import io,unittest
from native_adam100 import np,torch,make,optimizer,step,validate_roles
def assert_state(test,a,b):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        test.assertEqual(list(a),list(b))
        for k in a:assert_state(test,a[k],b[k])
    elif isinstance(a,(list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):assert_state(test,x,y)
    else:test.assertEqual(a,b)
class NativeChecks(unittest.TestCase):
    def test_full_B20_Adam_checkpoint_next_two_updates_exact(self):
        rng=np.random.default_rng(20261029)
        m=make();o=optimizer(m)
        raw=rng.normal(size=(20,3,90,189)).astype(np.float32)
        labels=np.arange(20)%12;step(m,o,raw,labels)
        memory=io.BytesIO()
        torch.save(dict(model=m.state_dict(),optimizer=o.state_dict(),
                        torch_rng=torch.get_rng_state()),memory)
        memory.seek(0);saved=torch.load(memory,map_location='cpu',weights_only=True)
        q=make();qo=optimizer(q)
        q.load_state_dict(saved['model'],strict=True)
        qo.load_state_dict(saved['optimizer'])
        for _ in range(2):
            raw=rng.normal(size=(20,3,90,189)).astype(np.float32)
            torch.set_rng_state(saved['torch_rng'])
            left=step(m,o,raw,labels)
            torch.set_rng_state(saved['torch_rng'])
            right=step(q,qo,raw,labels)
            self.assertEqual(left,right)
            assert_state(self,m.state_dict(),q.state_dict())
            assert_state(self,o.state_dict(),qo.state_dict())
    def test_unapproved_training_or_validation_user_metadata_blocks(self):
        train=np.repeat([3,4,5,6],[1200,1200,1200,1100]);val=np.full(1000,2)
        validate_roles(train,val)
        invalid=train.copy();invalid[0]=7
        with self.assertRaises(AssertionError):validate_roles(invalid,val)
        invalid=val.copy();invalid[0]=8
        with self.assertRaises(AssertionError):validate_roles(train,invalid)
if __name__=='__main__':unittest.main(verbosity=2)
