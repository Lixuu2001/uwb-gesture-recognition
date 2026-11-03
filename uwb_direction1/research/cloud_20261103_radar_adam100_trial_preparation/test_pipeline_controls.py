"""Full100 producer/auditor control fixtures with tiny mocked models/engine.

These certify I/O/callback/mismatch refusal only, never real network numerics,
RMS/pixels, private Drive metadata, registration or UWB accuracy.
"""
import hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import adam_engine as engine
import produce,audit_full,audit_pixels,trial_common as common

class PipelineControls(unittest.TestCase):
    def fixture(self,b):
        np=engine.np;torch=engine.torch
        ids=np.arange(6900);users=np.repeat(np.arange(1,7),[1200,1000,1200,1200,1200,1100])
        loc=dict(train=np.flatnonzero(users>=3),validation=np.flatnonzero(users==2),diagnostic=np.flatnonzero(users==1))
        inputs=(ids,users,loc,np.zeros((1,90,567),np.float32),np.arange(4700)%12,np.arange(1000)%10)
        def make():
            torch.manual_seed(20261029)
            model=torch.nn.Module();model.fc=torch.nn.Linear(3,12)
            model.register_buffer('counter',torch.tensor(0));return model
        def evaluate(model,maps,positions):
            a=np.zeros((len(positions),12),np.float32);a[:,int(model.counter)%12]=1;return a
        def run(maps,loc,y,v,users,on_step,on_epoch):
            model=make();g=torch.Generator().manual_seed(20261029);update=0
            for epoch in range(1,101):
                order=torch.randperm(4700,generator=g).numpy()
                for offset in range(0,4700,20):
                    update+=1;pos=order[offset:offset+20]
                    on_step(dict(epoch=epoch,offset=offset,update=update,loss=.25,
                        order_sha256=hashlib.sha256(pos.tobytes()).hexdigest()))
                model.counter.fill_(update)
                state=dict(model=model.state_dict(),optimizer={'virtual_steps':update},
                    order_rng=g.get_state(),torch_rng=torch.get_rng_state(),epoch=epoch,updates=update)
                record=dict(epoch=epoch,updates=update,validation_macro=0.,
                    selection='fixed_final_epoch100_not_best_validation',
                    diagnostic_labels_read=False,users7_8_labels_used=False)
                on_epoch(record,evaluate(model,maps,loc['validation']),state)
            return state
        policy=dict(source_manifest_sha256='virtual',data='not-used')
        common.write(b/'TRIAL_PROTOCOL.json',dict(state='virtual_fixture_only_no_activation'))
        patches=[patch.object(produce,'B',b),patch.object(audit_full,'B',b),
            patch.object(produce,'ready',return_value=policy),patch.object(audit_full,'ready',return_value=policy),
            patch.object(produce,'load_inputs',return_value=inputs),patch.object(audit_full,'load_inputs',return_value=inputs),
            patch.object(engine,'make',side_effect=make),patch.object(engine,'run',side_effect=run),
            patch.object(engine,'evaluate',side_effect=evaluate),
            patch.object(audit_pixels,'verify',return_value=dict(pixel_count=1038643200,
                all6900_raw_RMS_cache_FP64_verified=True,synthetic_mock_not_real_pixel_audit=True))]
        for p in patches:p.start()
        return patches
    def test_producer_full100_io_replays_and_mutated_step_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            b=Path(directory);patches=self.fixture(b)
            try:
                with patch('builtins.print'):produce.main()
                self.assertEqual(len((b/'run01/steps.jsonl').read_text().splitlines()),23500)
                self.assertEqual(len(list((b/'run01').glob('validation_epoch*.npz'))),100)
                (b/'supervisor01').mkdir()
                common.write(b/'supervisor01/TERMINATION.json',dict(exit_code=0,
                    wall_timeout=False,resource_stop_reason=None,cgroup_oom_kill_delta=0))
                with patch('builtins.print'):audit_full.main()
                self.assertEqual(common.read(b/'COMPLETE_AUDIT.json')['state'],'pass')
                self.assertFalse((b/'FINALIZE_ONCE.lock').exists())
                # Synthetic certificate is removed only inside this disposable fixture.
                (b/'COMPLETE_AUDIT.json').unlink()
                with (b/'run01/steps.jsonl').open('a') as f:f.write('{}\n')
                with self.assertRaises(AssertionError):audit_full.main()
            finally:
                for p in reversed(patches):p.stop()

if __name__=='__main__':unittest.main(verbosity=2)
