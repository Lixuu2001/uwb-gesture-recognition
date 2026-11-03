"""Synthetic full default batches, separate process per family; no UWB arrays."""
from pathlib import Path
import sys,tempfile,unittest,copy
import engine as e
ARM=sys.argv.pop(1)
e.setup(ARM)
class NativeState(unittest.TestCase):
    def test_full_batch_resume_rng_aliases_slow_sync_and_next_updates(self):
        m,o,g=e.make(ARM);rng=e.np.random.default_rng(20261029);batch=64 if ARM=='timemil' else 20
        shape=(batch,90,567) if ARM=='timemil' else (batch,3,90,189);labels=e.np.arange(batch)%12
        for epoch in range(1,6):
            e.order(ARM,4700,epoch,g);e.step(ARM,m,o,rng.normal(size=shape).astype('float32'),labels,epoch)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'checkpoint.pt';e.atomic_save(path,e.capture(m,o,g,5,5))
            right,ro,rg=e.make(ARM)
            e.restore(e.torch.load(path,map_location='cpu',weights_only=True),m,o,g)
            e.restore(e.torch.load(path,map_location='cpu',weights_only=True),right,ro,rg)
            if ARM=='timemil':
                self.assertEqual(m.wave1.data_ptr(),m.wave1_.data_ptr())
                self.assertEqual(right.wave1.data_ptr(),right.wave1_.data_ptr())
            for i,epoch in enumerate([6,10]):
                x=rng.normal(size=shape).astype('float32')
                rng_state=e.torch.get_rng_state();left_order=e.order(ARM,4700,epoch,g)
                left=e.step(ARM,m,o,x,labels,epoch);left_state=e.capture(m,o,g,epoch,6+i);left_sha=e.semantic(left_state)
                e.torch.set_rng_state(rng_state);right_order=e.order(ARM,4700,epoch,rg)
                e.np.testing.assert_array_equal(left_order,right_order)
                actual=e.step(ARM,right,ro,x,labels,epoch)
                self.assertEqual(left,actual);self.assertEqual(left_sha,e.semantic(e.capture(right,ro,rg,epoch,6+i)))
            if ARM=='timemil':self.assertEqual(o.count,7);self.assertTrue(o.slow)
    def test_generic_step_matches_unchanged_native_reference(self):
        import copy
        m,o,g=e.make(ARM);initial=copy.deepcopy(e.capture(m,o,g,0,0));rng=e.np.random.default_rng(9)
        shape=(64,90,567) if ARM=='timemil' else (20,3,90,189)
        x=rng.normal(size=shape).astype('float32');y=e.np.arange(shape[0])%12
        left=e.step(ARM,m,o,x,y,1);digest=e.semantic(e.capture(m,o,g,0,1))
        right,ro,rg=e.make(ARM);e.restore(initial,right,ro,rg)
        if ARM=='shuffle':actual=e.native.step(right,ro,x,y)
        else:
            ro.zero_grad();z=right(e.torch.from_numpy(x),warmup=True)
            loss=e.torch.nn.functional.binary_cross_entropy_with_logits(z,e.torch.nn.functional.one_hot(e.torch.as_tensor(y).long(),12).float())
            loss.backward();e.torch.nn.utils.clip_grad_norm_(right.parameters(),2.);ro.step();actual=float(loss.detach())
        self.assertEqual(left,actual);self.assertEqual(digest,e.semantic(e.capture(right,ro,rg,0,1)))
if __name__=='__main__':unittest.main(verbosity=2)
