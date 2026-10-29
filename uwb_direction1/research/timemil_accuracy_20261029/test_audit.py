"""Synthetic tests only; never open project arrays, labels or checkpoints."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import audit_result as verifier

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))

def fixture(base):
    write(base/'run01/STATE.json',dict(state='complete',pid=None,completed_arms=list(verifier.ARMS)))
    write(base/'supervisor01/TERMINATION.json',dict(exit_code=0,wall_timeout=False,
          supervisor_wall_seconds=100.,child_cpu_seconds=100.))
    freeze = {}
    for arm in verifier.ARMS:
        folder = base/'run01'/arm
        folder.mkdir(parents=True)
        (folder/'predictions_unscored.npz').write_bytes(b'SENTINEL: not an array')
        (folder/'best.pt').write_bytes(b'SENTINEL: not a model')
        freeze[arm] = verifier.sha(folder/'predictions_unscored.npz')
        write(folder/'ARM_COMPLETE.json',dict(epochs=64,prediction_sha256=freeze[arm],checkpoint_sha256=verifier.sha(folder/'best.pt')))
    write(base/'run01/ALL_PREDICTIONS_FROZEN.json',freeze)

class AccessAndStatistics(unittest.TestCase):
    def test_absent_run_blocks_before_array_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with patch.object(verifier.np,'load',side_effect=AssertionError('array read')) as sentinel:
                with self.assertRaisesRegex(RuntimeError,'not started'):
                    verifier.audit(base,base/'out.json')
                sentinel.assert_not_called()

    def test_running_blocks_before_array_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write(base/'run01/STATE.json',dict(state='training',pid=123))
            with patch.object(verifier.np,'load',side_effect=AssertionError('array read')) as sentinel:
                with self.assertRaisesRegex(RuntimeError,'incomplete'):
                    verifier.audit(base,base/'out.json')
                sentinel.assert_not_called()

    def test_complete_guard_checks_bytes_without_decoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); fixture(base)
            with patch.object(verifier.np,'load',side_effect=AssertionError('array read')) as sentinel:
                verifier.ready(base)
                sentinel.assert_not_called()

    def test_tamper_and_missing_arm_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); fixture(base)
            path = base/'run01/gap/predictions_unscored.npz'
            path.write_bytes(b'TAMPERED')
            with self.assertRaises(AssertionError): verifier.ready(base)
            write(base/'run01/STATE.json',dict(state='complete',pid=None,completed_arms=['timemil']))
            with self.assertRaisesRegex(RuntimeError,'missing arm'): verifier.ready(base)

    def test_bad_supervisor_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); fixture(base)
            write(base/'supervisor01/TERMINATION.json',dict(exit_code=-9,wall_timeout=True))
            with self.assertRaisesRegex(RuntimeError,'supervisor failed'): verifier.ready(base)

    def test_validation_ties_ignore_diagnostic_scores(self):
        history = [dict(epoch=e,validation_macro=.8,validation_bce=b,diagnostic_macro=d)
                   for e,b,d in [(1,.5,1.),(2,.4,0.),(3,.4,1.)]]
        self.assertEqual(verifier.selection(history)['epoch'],2)
        for row in history: row['diagnostic_macro'] = 1-row['diagnostic_macro']
        self.assertEqual(verifier.selection(history)['epoch'],2)

    def test_gate_and_separate_baseline_eligibility(self):
        self.assertEqual(verifier.gates(dict(timemil=.97,gap=.95),.968),dict(gate_passed=True,gap_baseline_expansion_eligible=False))
        self.assertFalse(verifier.gates(dict(timemil=.95,gap=.94),.968)['gate_passed'])
        self.assertFalse(verifier.gates(dict(timemil=.97,gap=.969),.968)['gate_passed'])
        self.assertTrue(verifier.gates(dict(timemil=.97,gap=.99),.968)['gap_baseline_expansion_eligible'])

    def test_macro_uses_present_classes_and_not_record_weighting(self):
        self.assertAlmostEqual(verifier.metrics(np.array([0,0,0,2]),np.array([0,0,2,2])),5/6)

    def test_roles_reject_held_contamination_and_wrong_order(self):
        ids,users = np.arange(12)+100,np.repeat(np.arange(1,7),2)
        split = dict(train=ids[4:].tolist(),validation=ids[2:4].tolist(),diagnostic=ids[:2].tolist())
        verifier.roles(split,ids,users)
        split['train'][0] = split['diagnostic'][0]
        with self.assertRaises(AssertionError): verifier.roles(split,ids,users)

    def test_independent_preprocessing_view_order_and_zero_maps(self):
        x = np.zeros((2,3,90,189),dtype=np.float32)
        x[0,0] = 2; x[0,1] = -4; x[0,2] = 8
        result = verifier.independent_preprocess(x)
        np.testing.assert_array_equal(result[0],np.tile(np.repeat([1,-1,1],189),(90,1)))
        np.testing.assert_array_equal(result[1],0)

class OptimizerState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from adapter import StableLookahead
        torch.set_num_threads(1)
        cls.torch = torch
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([1.,2.]))
                self.unused = torch.nn.Parameter(torch.tensor([3.]))
        cls.model = Toy()
        opt = StableLookahead(cls.model.parameters())
        cls.first = None
        for i in range(7):
            opt.zero_grad()
            cls.model.weight.square().sum().backward()
            opt.step()
            if i == 0: cls.first = copy.deepcopy(opt.state_dict())
        cls.last = copy.deepcopy(opt.state_dict())

    def test_valid_first_and_post_sync_unused_state(self):
        verifier.check_optimizer(self.model,self.first,1,{'unused'})
        verifier.check_optimizer(self.model,self.last,7,{'unused'})

    def test_wrong_step_and_missing_slow_state_rejected(self):
        bad = copy.deepcopy(self.last); bad['count'] = 6
        with self.assertRaises(AssertionError): verifier.check_optimizer(self.model,bad,7,{'unused'})
        bad = copy.deepcopy(self.last); bad['slow'] = {}
        with self.assertRaises(AssertionError): verifier.check_optimizer(self.model,bad,7,{'unused'})

    def test_nonfinite_and_wrong_shape_moments_rejected(self):
        bad = copy.deepcopy(self.last); bad['fast']['state'][0]['exp_avg'][0] = float('nan')
        with self.assertRaises(AssertionError): verifier.check_optimizer(self.model,bad,7,{'unused'})
        bad = copy.deepcopy(self.last); bad['fast']['state'][0]['exp_avg_sq'] = self.torch.zeros(3)
        with self.assertRaises(AssertionError): verifier.check_optimizer(self.model,bad,7,{'unused'})

if __name__ == '__main__':
    unittest.main(verbosity=2)
