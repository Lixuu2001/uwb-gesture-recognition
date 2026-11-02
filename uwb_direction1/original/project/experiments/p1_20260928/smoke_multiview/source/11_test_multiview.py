"""CPU contracts for observed-view training and reproducible recovery."""
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from train_baseline import T, extract_batch, source_parts


class MultiviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('multiview'),
                             'The observed-view implementation is missing')
        return importlib.import_module('multiview')

    def arrays(self):
        a = np.arange(7 * 3 * 90 * 189, dtype=np.float32).reshape(7, 3, 90, 189) / 1e5
        return a, -a, {'train': np.arange(5), 'val': np.array([5]), 'test': np.array([6])}

    def test_all_seven_direct_subsets_equal_independent_full_view_cache(self):
        m = self.module()
        self.assertEqual(m.SUBSETS, ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)))
        torch.manual_seed(17)
        model = m.SubsetModel().eval()
        x = torch.randn(1, 3, 90, 378)
        with torch.no_grad():
            full = model.encode(x)
            separate = torch.cat([model.encode(x[:, i:i+1]) for i in range(3)], 1)
            torch.testing.assert_close(full, separate, rtol=0, atol=1e-5)
            for subset in m.SUBSETS:
                direct = model(x[:, subset], subset)
                cached = model.fuse(full[:, subset], subset)
                self.assertEqual(tuple(direct.shape), (1, 12))
                torch.testing.assert_close(direct, cached, rtol=0, atol=1e-5)

    def test_extractor_preserves_right_penultimate_and_excludes_final_head(self):
        m = self.module()
        torch.manual_seed(21)
        right = T.models.Classifier([2], 'concat', 'mean').eval()
        model = m.SubsetModel(right).eval()
        x = torch.randn(1, 1, 90, 378)
        with torch.no_grad():
            expected, _ = extract_batch(right, x)
            actual = model.encode(x)[:, 0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(len(model.projection), 5)
        self.assertEqual((model.fuser[0].in_features, model.fuser[-1].out_features), (771, 12))
        self.assertFalse(any(name.startswith('head.') for name, _ in model.named_parameters()))

    def test_unobserved_nonfinite_raw_and_cached_features_cannot_change_prediction(self):
        m = self.module()
        a, p, parts = self.arrays()
        reader = m.SubsetData(a, p, parts)
        model = m.SubsetModel().eval()
        with torch.no_grad():
            raw = reader.load_ids([5], 'val', (2,), device='cpu')
            expected = model(raw, (2,))
            a[:, :2] = np.nan
            p[:, :2] = np.inf
            actual = model(reader.load_ids([5], 'val', (2,), device='cpu'), (2,))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            cached = torch.full((1, 3, 256), float('nan'))
            cached[:, 2:] = model.encode(raw)
            torch.testing.assert_close(model.fuse(cached[:, (2,)], (2,)), expected, rtol=0, atol=0)

    def test_gradients_reach_encoder_projection_and_fusion(self):
        m = self.module()
        model = m.SubsetModel().train()
        loss = nn.functional.cross_entropy(model(torch.randn(1, 2, 90, 378), (0, 2)), torch.tensor([3]))
        loss.backward()
        for parameter in (model.encoder.temporal.proj.weight, model.encoder.spatial.proj.weight,
                          model.projection[0].weight, model.projection[3].weight,
                          model.fuser[0].weight, model.fuser[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_input_shapes_and_physical_indices_are_guarded(self):
        m = self.module()
        model = m.SubsetModel().eval()
        for subset in ((), (0, 0), (-1,), (3,), (2, 0), (True,), (0.0,)):
            with self.subTest(subset=subset), self.assertRaises(ValueError):
                model(torch.zeros(1, max(1, len(subset)), 90, 378), subset)
        for shape in ((1, 90, 378), (1, 1, 90, 189), (1, 4, 90, 378), (0, 1, 90, 378)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model.encode(torch.zeros(shape))
        with self.assertRaises(ValueError):
            model(torch.zeros(1, 2, 90, 378), (2,))
        with self.assertRaises(ValueError):
            model.fuse(torch.zeros(1, 1, 255), (2,))
        with self.assertRaises(ValueError):
            model.fuse(torch.zeros(1, 3, 256), (2,))

    def test_reader_checks_roles_ids_and_partition_overlap(self):
        m = self.module()
        a, p, parts = self.arrays()
        reader = m.SubsetData(a, p, parts)
        loaded = reader.load_ids([5], 'val', (0, 2), device='cpu').numpy()
        np.testing.assert_array_equal(loaded, np.concatenate((a[[5]][:, (0, 2)], p[[5]][:, (0, 2)]), -1))
        for ids, role in (([0], 'val'), ([5], 'test'), ([], 'train'), ([0.], 'train'), ([0], 'bad')):
            with self.subTest(ids=ids, role=role), self.assertRaises(ValueError):
                reader.load_ids(ids, role, (0,), device='cpu')
        for bad in ({**parts, 'val': np.array([0])}, {**parts, 'train': np.array([0, 0])},
                    {**parts, 'test': np.array([7])}, {**parts, 'test': np.array([6.])}):
            with self.assertRaises(ValueError):
                m.SubsetData(a, p, bad)

    def test_eight_real_folds_match_9300_mask_and_six_whole_training_users(self):
        m = self.module()
        for held in range(1, 9):
            _, y, subjects, parts = source_parts(held)
            m.validate_parts(y, subjects, parts, held)
            self.assertEqual(sum(map(len, parts.values())), 9300)
            self.assertEqual(len(np.unique(subjects[parts['train']])), 6)
            self.assertEqual(set(subjects[parts['val']]), {held % 8 + 1})
            self.assertEqual(set(subjects[parts['test']]), {held})

    def test_sampler_subset_rng_is_independent_uniform_and_restorable(self):
        m = self.module()
        sampler = m.SubsetSampler(np.arange(31), 3)
        expected = np.random.default_rng(20263).integers(0, 7, size=32)
        for index in expected[:11]:
            sampler.next_epoch()
            self.assertEqual(sampler.next_subset(), m.SUBSETS[index])
        state = sampler.state_dict()
        next_order = sampler.next_epoch()
        next_subsets = [sampler.next_subset() for _ in range(21)]
        restored = m.SubsetSampler(np.arange(31), 3)
        restored.load_state_dict(state)
        np.testing.assert_array_equal(restored.next_epoch(), next_order)
        self.assertEqual([restored.next_subset() for _ in range(21)], next_subsets)
        self.assertEqual(next_subsets, [m.SUBSETS[i] for i in expected[11:]])

    def test_train_epoch_reads_one_subset_per_effective_batch_and_all_source_ids(self):
        m = self.module()
        a, p, parts = self.arrays()
        class Reader(m.SubsetData):
            def load_ids(self, ids, role, indices, device='cuda'):
                self.calls.append((np.asarray(ids).copy(), role, indices))
                return super().load_ids(ids, role, indices, device)
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(1, 12)
            def forward(self, raw, indices):
                return self.head(raw.square().mean((1, 2, 3)).unsqueeze(1))
        reader = Reader(a, p, parts)
        reader.calls = []
        model = Tiny()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        sampler = m.SubsetSampler(parts['train'], 1)
        config = {'batch': 4, 'microbatch': 2, 'smoke_steps': 0}
        row = m.train_epoch(model, optimizer, sampler, reader, np.arange(7), config)
        self.assertEqual((row['source_rows'], row['steps']), (5, 2))
        np.testing.assert_array_equal(np.sort(np.concatenate([x[0] for x in reader.calls])), parts['train'])
        self.assertTrue(all(x[1] == 'train' for x in reader.calls))
        self.assertEqual(reader.calls[0][2], reader.calls[1][2])

    def test_progress_restores_subset_rng_optimizer_and_best_after_interrupted_save(self):
        m = self.module()
        model = nn.Sequential(nn.Linear(2, 12), nn.Dropout(.1))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        sampler = m.SubsetSampler(np.arange(7), 1)
        torch.manual_seed(10)
        identity = {'fold': 1, 'source': 'test'}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            T.state.atomic_save(directory/'best.pt', {'state': model.state_dict(), 'epoch': 0, 'val_ce': 2.})
            selection = T.state.Selection(2.)
            m.save_epoch(directory, model, optimizer, sampler, identity, selection, [], 0)
            order, subset, draw = sampler.next_epoch(), sampler.next_subset(), torch.rand(3)
            T.state.atomic_save(directory/'best.pt', {'interrupted_next_best': True})
            got, _, epoch = m.restore_progress(directory, model, optimizer, sampler, identity)
            np.testing.assert_array_equal(sampler.next_epoch(), order)
            self.assertEqual(sampler.next_subset(), subset)
            torch.testing.assert_close(torch.rand(3), draw, rtol=0, atol=0)
            self.assertEqual((got.best_ce, epoch), (2., 0))
            self.assertIn('state', torch.load(directory/'best.pt', weights_only=False))
            previous = T.state.state_hash(model.state_dict())
            with self.assertRaises(ValueError):
                m.restore_progress(directory, model, optimizer, sampler, {'fold': 2})
            self.assertEqual(T.state.state_hash(model.state_dict()), previous)

    def test_offline_export_axis_order_roles_and_selected_validation_replay(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'export_predictions'), 'Frozen seven-subset export is missing')
        a, p, parts = self.arrays()
        reader = m.SubsetData(a, p, parts)
        class TinyCache(nn.Module):
            def __init__(self):
                super().__init__()
                self.probe = nn.Parameter(torch.zeros(1))
            def encode(self, raw):
                return raw.square().mean((2, 3)).unsqueeze(-1).expand(-1, -1, 256)
            def fuse(self, features, indices):
                signal = sum(features[:, k, 0] * (v + 1) for k, v in enumerate(indices))
                return signal[:, None] * torch.arange(12)[None, :]
        model = TinyCache()
        features, logits, ces = m.evaluate(model, reader, np.arange(7), parts['val'], 'val', batch=2)
        self.assertEqual(features.shape, (1, 3, 256))
        self.assertEqual(logits.shape, (1, 7, 12))
        expected = np.array([float(nn.functional.cross_entropy(torch.from_numpy(logits[:, i]), torch.tensor([5]))) for i in range(7)])
        np.testing.assert_array_equal(ces, expected)
        best = {'val_logits': torch.from_numpy(logits), 'val_ce': float(ces.mean())}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            m.export_predictions(directory, model, reader, np.arange(7), np.arange(7), parts, best, batch=2)
            with np.load(directory/'predictions.npz') as saved:
                np.testing.assert_array_equal(saved['ids'], np.arange(7))
                np.testing.assert_array_equal(saved['role'], [0, 0, 0, 0, 0, 1, 2])
                np.testing.assert_array_equal(saved['subset_masks'], m.SUBSET_MASKS)
                np.testing.assert_array_equal(saved['val_logits'], logits)
                self.assertEqual(saved['features'].shape, (7, 3, 256))
                self.assertEqual(saved['logits'].shape, (7, 7, 12))
            wrong = {**best, 'val_logits': torch.from_numpy(logits + 1.)}
            with self.assertRaises(ValueError):
                m.export_predictions(directory, model, reader, np.arange(7), np.arange(7), parts, wrong, batch=2)

    def test_initial_checkpoint_requires_new_same_fold_completed_baseline_hashes(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'verify_initial'), 'Baseline checkpoint verification is missing')
        parts = {'train': np.array([0]), 'val': np.array([1]), 'test': np.array([2])}
        source = {'code': {str(m.HERE/'train_baseline.py'): T.protocol.digest(m.HERE/'train_baseline.py')},
                  'data': {'data': 'frozen'}, 'plan_sha256': 'plan'}
        config = {'seed': 2026, 'smoke': False, 'warmup_epochs': 40, 'warmup_min': 12,
                  'warmup_patience': 8, 'warmup_batch': 128, 'warmup_microbatch': 32,
                  'warmup_lr': 1e-4, 'weight_decay': 1e-4}
        with tempfile.TemporaryDirectory() as temporary:
            baseline = Path(temporary)
            directory = baseline/'fold1'/'warmup'
            directory.mkdir(parents=True)
            files = {}
            for name in ('right.pt', 'best.pt', 'progress.pt', 'predictions.npz'):
                (directory/name).write_bytes(name.encode())
                files[name] = T.protocol.digest(directory/name)
            identity = {'source': source, 'config': config, 'fold': 1, 'arm': 'warmup',
                        'partitions': {k: T.protocol.array_hash(v) for k, v in parts.items()}}
            result = {'identity': identity, 'files': files, 'history': [{'seconds': 3.}], 'best_epoch': 0}
            T.protocol.dump(baseline/'manifest.json', {'source': source, 'config': config})
            T.protocol.dump(directory/'result.json', result)
            verified = m.verify_initial(baseline, 1, parts, source)
            self.assertEqual(verified['right_sha256'], files['right.pt'])
            self.assertEqual(verified['reused_training_seconds'], 3.)
            T.protocol.dump(directory/'result.json', {**result, 'identity': {**identity, 'fold': 2}})
            with self.assertRaises(ValueError):
                m.verify_initial(baseline, 1, parts, source)
            T.protocol.dump(directory/'result.json', result)
            (directory/'right.pt').write_bytes(b'tampered')
            with self.assertRaises(ValueError):
                m.verify_initial(baseline, 1, parts, source)

    def test_smoke_requires_separate_output_and_protocol_config_is_fixed(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'configuration'), 'Registered training configuration is missing')
        config = m.configuration(False)
        self.assertEqual((config['batch'], config['microbatch'], config['min_epochs'], config['epochs'], config['patience']), (128, 16, 12, 40, 8))
        self.assertEqual(config['criterion'], 'cross_entropy only')
        smoke = m.configuration(True)
        self.assertEqual((smoke['epochs'], smoke['smoke_steps']), (1, 3))
        with self.assertRaises(ValueError):
            m.validate_output_paths(m.OUTPUT/'multiview_s2026', m.OUTPUT/'baseline_s2026', True)
        with self.assertRaises(ValueError):
            m.validate_output_paths(m.OUTPUT/'baseline_s2026'/'child', m.OUTPUT/'baseline_s2026', False)
        m.validate_output_paths(m.OUTPUT/'smoke_multiview_test', m.OUTPUT/'baseline_s2026', True)

    def test_epoch_loop_keeps_initial_candidate_and_completed_resume_is_exact(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'fit_model'), 'Audited epoch loop is missing')
        a, p, parts = self.arrays()
        reader = m.SubsetData(a, p, parts)
        class TinyCache(nn.Module):
            def __init__(self):
                super().__init__()
                self.head = nn.Linear(1, 12)
            def encode(self, raw):
                return raw.square().mean((2, 3)).unsqueeze(-1).expand(-1, -1, 256)
            def fuse(self, features, indices):
                return self.head(features[:, :, :1].mean(1))
            def forward(self, raw, indices):
                return self.fuse(self.encode(raw), indices)
        model = TinyCache()
        config = {**m.configuration(True), 'batch': 4, 'microbatch': 2, 'eval_batch': 2}
        identity = {'fold': 1, 'config': config, 'source': 'cpu-contract'}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            selected, history, reason = m.fit_model(directory, model, reader, np.arange(7), parts, 1, identity, config)
            self.assertEqual([row['epoch'] for row in history], [0, 1])
            self.assertEqual(selected.best_epoch, int(np.argmin([row['val_ce'] for row in history])))
            self.assertEqual(history[1]['source_rows'], 5)
            self.assertEqual(reason, 'max_epochs')
            self.assertTrue((directory/'initial.pt').exists())
            best = torch.load(directory/'best.pt', weights_only=False)
            self.assertEqual(tuple(best['val_logits'].shape), (1, 7, 12))
            again, same_history, _ = m.fit_model(directory, model, reader, np.arange(7), parts, 1, identity, config)
            self.assertEqual(again.state_dict(), selected.state_dict())
            self.assertEqual(same_history, history)
            status = json.loads((directory/'status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['epoch'], 1)

    def test_cli_exposes_queue_options_and_rejects_unsafe_smoke_before_training(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'main') and hasattr(m, 'run_fold'), 'Sequential fold CLI is missing')
        command = [sys.executable, str(Path(m.__file__))]
        help_result = subprocess.run(command + ['--help'], capture_output=True, text=True, check=False)
        self.assertEqual(help_result.returncode, 0)
        for option in ('--out', '--baseline', '--folds', '--smoke'):
            self.assertIn(option, help_result.stdout)
        unsafe = subprocess.run(command + ['--smoke'], capture_output=True, text=True, check=False)
        self.assertEqual(unsafe.returncode, 2)
        self.assertIn('independent --out', unsafe.stderr)

    def test_frozen_source_snapshots_are_hash_checked_on_resume(self):
        m = self.module()
        self.assertTrue(hasattr(m, 'freeze_sources'), 'Snapshot integrity verification is missing')
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_file = directory/'source.py'
            source_file.write_text('frozen source', encoding='utf-8')
            source = {'code': {str(source_file): T.protocol.digest(source_file)}}
            out = directory/'out'
            m.freeze_sources(out, source, {'seed': 2026})
            m.freeze_sources(out, source, {'seed': 2026})
            (out/'source'/'00_source.py').write_text('tampered', encoding='utf-8')
            with self.assertRaises(ValueError):
                m.freeze_sources(out, source, {'seed': 2026})


if __name__ == '__main__':
    unittest.main()
