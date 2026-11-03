"""Synthetic operational tests; no actual UWB fit, selection or held labels."""
import tempfile,unittest,json,io,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,'/workspace/uwb-cloud-current/uwb_direction1/research/cloud_20261103_radar_adam100_preparation')
import resume_engine as resume
from adam_engine import semantic

class Recovery(unittest.TestCase):
    def test_native_B20_Adam_order_and_global_rng_next_two_updates_exact(self):
        np=resume.np;torch=resume.torch
        rng=np.random.default_rng(20261029);model=resume.make();opt=resume.optimizer(model)
        batch=rng.normal(size=(20,3,90,189)).astype(np.float32);labels=np.arange(20)%12
        resume.step(model,opt,batch,labels)
        sampler=torch.Generator().manual_seed(20261029)
        torch.randperm(4700,generator=sampler)
        mem=io.BytesIO();torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),
            order_rng=sampler.get_state(),torch_rng=torch.get_rng_state(),epoch=0,updates=1),mem)
        mem.seek(0);saved=torch.load(mem,map_location='cpu',weights_only=True)
        batches=[rng.normal(size=(20,3,90,189)).astype(np.float32) for _ in range(2)]
        torch.set_rng_state(saved['torch_rng']);expected_order=torch.randperm(4700,generator=sampler)
        losses=[resume.step(model,opt,x,labels) for x in batches]
        expected=(semantic(model.state_dict()),semantic(opt.state_dict()),semantic(sampler.get_state()),semantic(torch.get_rng_state()))
        other,oo,sg=resume.restore(saved)
        torch.testing.assert_close(torch.randperm(4700,generator=sg),expected_order,rtol=0,atol=0)
        actual=[resume.step(other,oo,x,labels) for x in batches]
        self.assertEqual(losses,actual)
        self.assertEqual(expected,(semantic(other.state_dict()),semantic(oo.state_dict()),semantic(sg.get_state()),semantic(torch.get_rng_state())))

    def test_prefix_copy_keeps_original_suffix_and_only_committed_new_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'old/run01';old.mkdir(parents=True);out=root/'new';out.mkdir()
            lines=[json.dumps(dict(update=i+1,epoch=i//235+1))+'\n' for i in range(10604)]
            original=''.join(lines);(old/'steps.jsonl').write_text(original)
            resume.torch.save(dict(epoch=45,updates=10575),old/'final_full.pt')
            saved,suffix=resume.copy_prefix(dict(operational_recovery=dict(original_root=str(root/'old'))),out)
            self.assertEqual((saved['epoch'],saved['updates']),(45,10575));self.assertEqual(len(suffix),29)
            self.assertEqual((old/'steps.jsonl').read_text(),original)
            self.assertEqual(len((out/'steps.jsonl').read_text().splitlines()),10575)
    def test_incomplete_prefix_cannot_be_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);old=root/'old/run01';old.mkdir(parents=True);out=root/'new';out.mkdir()
            (old/'steps.jsonl').write_text('{}\n')
            with self.assertRaises(AssertionError):resume.copy_prefix(dict(operational_recovery=dict(original_root=str(root/'old'))),out)
    def test_changed_interrupted_suffix_cannot_be_silently_replaced(self):
        suffix=[dict(update=10576,loss=.1)]
        resume.check_suffix(suffix[0],suffix)
        with self.assertRaises(AssertionError):resume.check_suffix(dict(update=10576,loss=.2),suffix)
        with self.assertRaises(AssertionError):resume.check_suffix(dict(update=10575,loss=.1),suffix)

    def test_wrong_input_epoch_refused_before_any_update(self):
        np=resume.np;users=np.repeat(np.arange(1,7),[1200,1000,1200,1200,1200,1100])
        loc=dict(train=np.flatnonzero(users>=3),validation=np.flatnonzero(users==2))
        with patch.object(resume,'restore') as restore:
            with self.assertRaises(AssertionError):resume.continue_run(None,loc,None,None,users,dict(epoch=44,updates=10340),None,None)
            restore.assert_not_called()

if __name__=='__main__':unittest.main(verbosity=2)
