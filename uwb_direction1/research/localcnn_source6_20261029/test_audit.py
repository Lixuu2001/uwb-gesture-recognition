"""Synthetic guard/metric/role tests; never opens any real UWB input."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import audit_complete as verifier

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))

def fixture(base):
    write(base / 'run01/STATE.json', dict(state='complete', pid=None,
          completed_folds=list(range(1, 7)), all_new_diagnostic_predictions_frozen=True))
    write(base / 'supervisor01/TERMINATION.json', dict(exit_code=0, wall_timeout=False,
          supervisor_wall_seconds=100., child_cpu_seconds=200.))
    freeze = {}
    for held in range(2, 7):
        folder = base / f'run01/held{held}'
        folder.mkdir(parents=True)
        (folder / 'predictions_unscored.npz').write_bytes(b'SENTINEL: not an array')
        (folder / 'best.pt').write_bytes(b'SENTINEL: not a checkpoint')
        freeze[str(held)] = verifier.digest(folder / 'predictions_unscored.npz')
        write(folder / 'FOLD_COMPLETE.json', dict(held=held, epochs=24,
              prediction_sha256=freeze[str(held)], checkpoint_sha256=verifier.digest(folder/'best.pt')))
    write(base / 'run01/ALL_NEW_PREDICTIONS_FROZEN.json', freeze)

class GuardTests(unittest.TestCase):
    def test_running_guard_precedes_all_array_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            write(base / 'run01/STATE.json', dict(state='training', pid=123))
            with patch.object(verifier.np, 'load', side_effect=AssertionError('array accessed')) as sentinel:
                with self.assertRaisesRegex(RuntimeError, 'not_ready'):
                    verifier.audit(base, base / 'out.json')
                sentinel.assert_not_called()
            self.assertFalse((base/'out.json').exists())

    def test_complete_metadata_never_decodes_sentinel_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); fixture(base)
            with patch.object(verifier.np, 'load', side_effect=AssertionError('array accessed')) as sentinel:
                self.assertEqual(verifier.require_complete(base)['exit_code'], 0)
                sentinel.assert_not_called()

    def test_missing_prediction_freeze_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); fixture(base)
            state = verifier.read(base/'run01/STATE.json')
            state['all_new_diagnostic_predictions_frozen'] = False
            write(base/'run01/STATE.json', state)
            with self.assertRaisesRegex(RuntimeError, 'freeze'):
                verifier.require_complete(base)

    def test_tampered_prediction_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); fixture(base)
            (base/'run01/held6/predictions_unscored.npz').write_bytes(b'TAMPERED')
            with self.assertRaises(AssertionError):
                verifier.require_complete(base)

    def test_unsuccessful_supervisor_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); fixture(base)
            write(base/'supervisor01/TERMINATION.json', dict(exit_code=-9, wall_timeout=True))
            with self.assertRaisesRegex(RuntimeError, 'unsuccessful'):
                verifier.require_complete(base)

class StatisticsTests(unittest.TestCase):
    def test_missing_class_macro_and_unequal_counts(self):
        self.assertAlmostEqual(verifier.independent_macro(np.array([0,0,0,2]), np.array([0,0,2,2])), 5/6)

    def test_validation_ties_ignore_diagnostic_results(self):
        rows = [dict(epoch=e, validation_macro=.8, validation_loss=l, diagnostic_macro=d)
                for e,l,d in [(1,.5,1.),(2,.4,0.),(3,.4,1.)]]
        self.assertEqual(verifier.choose_epoch(rows)['epoch'], 2)
        for row in rows: row['diagnostic_macro'] = 1-row['diagnostic_macro']
        self.assertEqual(verifier.choose_epoch(rows)['epoch'], 2)

    def test_gate_all_three_conditions(self):
        self.assertTrue(verifier.gate([.03,.03,.03,.03,-.01,-.01]))
        self.assertFalse(verifier.gate([.001]*6))
        self.assertFalse(verifier.gate([.1,.1,.1,-.001,-.001,-.001]))
        self.assertFalse(verifier.gate([.1,.1,.1,.1,-.03,-.001]))

    def test_source_split_exact_order_and_contamination(self):
        ids = np.arange(12)+100
        users = np.repeat(np.arange(1,7),2)
        split = dict(train=ids[[0,1,6,7,8,9,10,11]].tolist(),
                     validation=ids[[4,5]].tolist(), diagnostic=ids[[2,3]].tolist())
        verifier.verify_split(split, ids, users, 2)
        split['train'][0] = split['diagnostic'][0]
        with self.assertRaises(AssertionError):
            verifier.verify_split(split, ids, users, 2)

if __name__ == '__main__':
    unittest.main(verbosity=2)
