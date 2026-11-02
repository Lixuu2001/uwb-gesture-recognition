"""Synthetic checks for independent selection, split, provenance and gate audit."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

MODULE = Path(__file__).resolve().parents[1]/'audit.py'


def load_audit():
    assert MODULE.exists(), 'independent audit implementation is missing'
    spec = importlib.util.spec_from_file_location('_test_cross_meta_audit', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.audit = load_audit()

    def history(self, values, reason='max_epochs', minimum=2, patience=3):
        rows = [{'epoch': i, 'val_ce': x, 'train_loss': None if i == 0 else 1.,
                 'episode_hash': str(i), 'steps': 0 if i == 0 else 4, 'seconds': 1.}
                for i, x in enumerate(values)]
        return {'history': rows, 'best_epoch': int(np.argmin(values)), 'best_val_ce': min(values),
                'stop_reason': reason, 'max_epochs': len(values)-1, 'min_epochs': minimum, 'patience': patience}

    def test_first_minimum_and_epoch_zero_are_valid(self):
        self.audit.audit_history(self.history([1., 1.1, 1.2]))
        bad = self.history([2., 1., 1.])
        bad['best_epoch'] = 2
        with self.assertRaisesRegex(ValueError, 'first minimum'):
            self.audit.audit_history(bad)

    def test_premature_or_missed_early_stop_rejected(self):
        bad = self.history([1., 1.1], reason='early_stop', minimum=2)
        with self.assertRaises(ValueError):
            self.audit.audit_history(bad)
        bad = self.history([1., 1.1, 1.2, 1.3], minimum=1, patience=2)
        with self.assertRaisesRegex(ValueError, 'earlier'):
            self.audit.audit_history(bad)

    def test_independent_metrics_handle_missing_classes(self):
        logits = np.array([[4., 0., 0.], [0., 4., 0.], [0., 4., 0.]])
        metrics = self.audit.metrics(logits, np.array([0, 0, 1]))
        self.assertAlmostEqual(metrics['accuracy'], 2/3)
        self.assertAlmostEqual(metrics['macro_f1'], 2/3)
        self.assertAlmostEqual(metrics['macro_f1_fixed_classes'], 4/9)

    def test_gate_requires_all_32_and_user_equal_weight(self):
        scores = {arm: {str(h): {'accuracy': .5, 'macro_f1': .4} for h in range(1, 9)}
                  for arm in ['A0', 'A1', 'A2', 'A3']}
        for h in range(1, 9):
            scores['A3'][str(h)]['accuracy'] += .03 if h <= 6 else 0.
        result = self.audit.summarize_scores(scores, repetitions=1000)
        self.assertTrue(result['gate']['passed'])
        self.assertAlmostEqual(result['comparisons']['A3-A1']['accuracy']['mean_pp'], 2.25)
        del scores['A0']['8']
        self.assertIsNone(self.audit.summarize_scores(scores)['gate'])

    def test_episode_prefix_mismatch_is_detected(self):
        a = self.history([2., 1., .9])
        b = self.history([2., 1.])
        self.audit.audit_episode_prefix({'A0': a, 'A3': b})
        b['history'][1]['episode_hash'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'episode'):
            self.audit.audit_episode_prefix({'A0': a, 'A3': b})

    def test_episode_hash_recomputed_from_source_user_sampling(self):
        self.assertTrue(hasattr(self.audit, 'audit_sampling'), 'sampling replay audit is missing')
        labels = np.tile(np.repeat(np.arange(12), 2), 8)
        subjects = np.repeat(np.arange(1,9), 24)
        parts = self.audit.partitions(labels, subjects, 1)
        path = MODULE.with_name('protocol.py')
        spec = importlib.util.spec_from_file_location('_audit_test_source_sampler', path)
        protocol = importlib.util.module_from_spec(spec); spec.loader.exec_module(protocol)
        sampler = protocol.EpisodeSampler(parts['train'], labels, subjects, 7778, 2)
        result = self.history([2., 1., .9])
        result['identity'] = {'fold': 1, 'arm': 'A3', 'config':
                              {'seed': 2026, 'episode_seed': 7777, 'per_user': 2, 'smoke_steps': 2}}
        for row in result['history'][1:]:
            digest = hashlib.sha256()
            for _ in range(2):
                ep = sampler.next()
                digest.update(ep['support_ids'].tobytes()); digest.update(ep['query_ids'].tobytes())
            row.update(episode_hash=digest.hexdigest(), steps=2)
        self.audit.audit_sampling(result, labels, subjects, parts, sampler.state_dict())
        result['history'][1]['episode_hash'] = 'fabricated_common_hash'
        with self.assertRaisesRegex(ValueError, 'sampling trace'):
            self.audit.audit_sampling(result, labels, subjects, parts, sampler.state_dict())

    def fixture(self, root):
        labels = np.tile(np.repeat(np.arange(12), 2), 8)
        subjects = np.repeat(np.arange(1, 9), 24)
        data_dir = root/'data'; data_dir.mkdir()
        for name, data in [('y', labels), ('subj', subjects), ('off', np.zeros(1)), ('only', np.zeros(1))]:
            np.save(data_dir/(name+'.npy'), data)
        code = root/'source.py'; code.write_text('# original source\n')
        source = {'data': {str(p): self.audit.file_hash(p) for p in data_dir.glob('*.npy')},
                  'code': {str(code): self.audit.file_hash(code)}}
        parts = self.audit.partitions(labels, subjects, 1)
        part_hashes = {k: self.audit.array_hash(v) for k, v in parts.items()}
        matrix_dir = root/'matrices'; matrix_dir.mkdir()
        matrix = np.eye(12, dtype=np.float32)
        # Matrix is independently reconstructed from OOF true-class mean.
        oof = np.eye(12)[labels[parts['train']]]
        np.savez(matrix_dir/'fold1.npz', matrix=matrix, oof_train=oof,
                 **{k+'_ids': v for k, v in parts.items()})
        matrix_hash = self.audit.file_hash(matrix_dir/'fold1.npz')
        fits = [{'held_subject': int(u), 'fit_ids': parts['train'][subjects[parts['train']] != u].tolist(),
                 'predict_ids': parts['train'][subjects[parts['train']] == u].tolist(), 'warnings': []}
                for u in np.unique(subjects[parts['train']])]
        meta = {'identity': {'fold': 1, 'validation_subject': 2, 'training_subjects': [3,4,5,6,7,8],
                             'data': source['data'], 'partitions': part_hashes},
                'matrix_sha256': matrix_hash, 'oof_fits': fits, 'warnings': 0,
                'validation_used_for_fit': False, 'test_used_for_fit': False}
        (matrix_dir/'fold1.json').write_text(json.dumps(meta))
        state = {'right.encoder.weight': torch.tensor([1.]), 'right.encoder.view_emb': torch.tensor([.1]),
                 'right.head.weight': torch.tensor([2.]), 'fuser.weight': torch.tensor([3.])}
        identity = {'fold': 1, 'arm': 'warmup', 'source': source,
                    'config': {'warmup_epochs': 1, 'warmup_min': 1, 'warmup_patience': 3,
                               'epochs': 16, 'min_epochs': 6, 'patience': 5},
                    'matrix_sha256': matrix_hash, 'partitions': part_hashes,
                    'initial_state_sha256': self.audit.state_hash(state)}
        dest = root/'fold1'/'warmup'; dest.mkdir(parents=True)
        logits = np.eye(12)[labels[parts['test']]]*4
        val_logits = np.eye(12)[labels[parts['val']]]*4
        ce = self.audit.metrics(val_logits, labels[parts['val']])['ce']
        result = self.history([ce+.1, ce], minimum=1)
        result.update(identity=identity)
        np.savez(dest/'predictions.npz', logits=logits, y=labels[parts['test']], ids=parts['test'],
                 val_logits=val_logits, val_y=labels[parts['val']], **{k+'_ids': v for k,v in parts.items()})
        torch.save({'state': state, 'epoch': 1, 'val_ce': ce}, dest/'best.pt')
        torch.save({k[len('right.'):]: v for k,v in state.items() if k.startswith('right.')}, dest/'right.pt')
        info = {'identity': identity, 'history': result['history'], 'epoch': 1,
                'selection': {'best_ce': ce, 'best_epoch': 1, 'stale': 0}}
        progress = {'state': state, 'info': info, 'optimizer': {'state': {}, 'param_groups': []},
                    'sampler': {'step': 4}, 'python_rng': (), 'numpy_rng': (),
                    'torch_rng': torch.zeros(1, dtype=torch.uint8), 'cuda_rng': None}
        torch.save(progress, dest/'progress.pt')
        result['files'] = {name: self.audit.file_hash(dest/name) for name in ['best.pt','right.pt','progress.pt','predictions.npz']}
        (dest/'result.json').write_text(json.dumps(result))
        return dest, labels, subjects

    def continuation_fixture(self, root, arm):
        warmup, labels, subjects = self.fixture(root)
        initial = torch.load(warmup/'best.pt', map_location='cpu', weights_only=False)
        warmup_result = json.loads((warmup/'result.json').read_text())
        identity = copy.deepcopy(warmup_result['identity']); identity['arm'] = arm
        identity['initial_state_sha256'] = self.audit.state_hash(initial['state'])
        ce = initial['val_ce']
        result = self.history([ce]+[ce+i*.1 for i in range(1,7)], reason='patience', minimum=6, patience=5)
        result.update(identity=identity, max_epochs=16)
        dest = warmup.parent/arm; dest.mkdir()
        for filename in ['right.pt','predictions.npz']:
            shutil.copyfile(warmup/filename, dest/filename)
        best = {'state': initial['state'], 'epoch': 0, 'val_ce': ce}
        torch.save(best, dest/'best.pt')
        last = {k: v.clone() for k,v in initial['state'].items()}
        last['right.encoder.weight'] += .01
        last['right.head.weight'] += .01
        if arm in ('A2','A3'): last['fuser.weight'] += .01
        info = {'identity': identity, 'history': result['history'], 'epoch': 6,
                'selection': {'best_ce': ce, 'best_epoch': 0, 'stale': 6},
                'best_file_sha256': self.audit.file_hash(dest/'best.pt'), 'best_checkpoint': best}
        progress = {'state': last, 'info': info, 'optimizer': {'state': {}, 'param_groups': []},
                    'sampler': {'step': 24}, 'python_rng': (), 'numpy_rng': (),
                    'torch_rng': torch.zeros(1, dtype=torch.uint8), 'cuda_rng': None}
        torch.save(progress, dest/'progress.pt')
        result['files'] = {name: self.audit.file_hash(dest/name) for name in ['best.pt','right.pt','progress.pt','predictions.npz']}
        (dest/'result.json').write_text(json.dumps(result))
        return dest

    def test_selected_epoch_zero_still_proves_last_encoder_and_head_updates(self):
        for arm in ['A0','A1','A2','A3']:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); self.continuation_fixture(root, arm)
                row = self.audit.audit_run(root, 1, arm)
                self.assertEqual(row['best_epoch'], 0)
                self.assertIn('last_parameter_updates', row)
                updates = row['last_parameter_updates']
                self.assertEqual(updates['encoder'], {'total': 2, 'changed': 1})
                self.assertEqual(updates['head'], {'total': 1, 'changed': 1})
                self.assertEqual(updates['aux'], {'total': 1, 'changed': int(arm in ('A2','A3'))})

    def test_unchanged_last_model_is_rejected_even_with_valid_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); dest = self.continuation_fixture(root, 'A3')
            progress = torch.load(dest/'progress.pt', map_location='cpu', weights_only=False)
            progress['state'] = progress['info']['best_checkpoint']['state']
            torch.save(progress, dest/'progress.pt')
            result = json.loads((dest/'result.json').read_text())
            result['files']['progress.pt'] = self.audit.file_hash(dest/'progress.pt')
            (dest/'result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'last encoder/head did not update'):
                self.audit.audit_run(root, 1, 'A3')

    def test_auxiliary_updates_must_follow_arm_definition(self):
        for arm in ['A0','A1','A2','A3']:
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); dest = self.continuation_fixture(root, arm)
                progress = torch.load(dest/'progress.pt', map_location='cpu', weights_only=False)
                initial = progress['info']['best_checkpoint']['state']['fuser.weight']
                progress['state']['fuser.weight'] = initial.clone()
                if arm in ('A0','A1'): progress['state']['fuser.weight'] += .01
                torch.save(progress, dest/'progress.pt')
                result = json.loads((dest/'result.json').read_text())
                result['files']['progress.pt'] = self.audit.file_hash(dest/'progress.pt')
                (dest/'result.json').write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, 'last auxiliary update pattern'):
                    self.audit.audit_run(root, 1, arm)

    def test_fixture_audits_then_rejects_modified_prediction_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dest, _, _ = self.fixture(root)
            report = self.audit.audit_run(root, 1, 'warmup')
            self.assertEqual(report['metrics']['accuracy'], 1.)
            with (dest/'predictions.npz').open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'hash'):
                self.audit.audit_run(root, 1, 'warmup')

    def test_result_stopping_fields_must_match_frozen_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dest, _, _ = self.fixture(root)
            original = json.loads((dest/'result.json').read_text())
            for field in ['patience', 'min_epochs', 'max_epochs']:
                with self.subTest(field=field):
                    result = copy.deepcopy(original)
                    result[field] += 1
                    (dest/'result.json').write_text(json.dumps(result))
                    with self.assertRaisesRegex(ValueError, 'stopping configuration differs'):
                        self.audit.audit_run(root, 1, 'warmup')

    def test_continuations_use_branch_budget_not_warmup_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dest, _, _ = self.fixture(root)
            original = json.loads((dest/'result.json').read_text())
            for arm in ['A0', 'A1', 'A2', 'A3']:
                with self.subTest(arm=arm):
                    branch = dest.parent/arm; branch.mkdir()
                    result = copy.deepcopy(original)
                    result['identity']['arm'] = arm
                    # The copied stopping values fit warmup, but do not fit
                    # the declared continuation budget of 16/6/5.
                    (branch/'result.json').write_text(json.dumps(result))
                    with self.assertRaisesRegex(ValueError, 'stopping configuration differs'):
                        self.audit.audit_run(root, 1, arm)

    def test_matrix_oof_cannot_include_validation_user(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, _ = self.fixture(root)
            path = root/'matrices'/'fold1.json'
            meta = json.loads(path.read_text())
            meta['oof_fits'][0]['fit_ids'].append(24)
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'OOF'):
                self.audit.audit_run(root, 1, 'warmup')

    def test_rehashed_wrong_deployment_state_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dest, _, _ = self.fixture(root)
            torch.save({'weight': torch.tensor([9.])}, dest/'right.pt')
            result = json.loads((dest/'result.json').read_text())
            result['files']['right.pt'] = self.audit.file_hash(dest/'right.pt')
            (dest/'result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'standalone Right'):
                self.audit.audit_run(root, 1, 'warmup')

    def test_rehashed_wrong_test_ids_are_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dest, _, _ = self.fixture(root)
            with np.load(dest/'predictions.npz') as archive:
                values = {k: archive[k] for k in archive.files}
            values['ids'] = values['ids'][::-1]
            np.savez(dest/'predictions.npz', **values)
            result = json.loads((dest/'result.json').read_text())
            result['files']['predictions.npz'] = self.audit.file_hash(dest/'predictions.npz')
            (dest/'result.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'test IDs/order'):
                self.audit.audit_run(root, 1, 'warmup')

    def load_renderer(self):
        path = MODULE.with_name('render_report.py')
        self.assertTrue(path.exists(), 'independent report renderer is missing')
        spec = importlib.util.spec_from_file_location('_test_cross_meta_report', path)
        renderer = importlib.util.module_from_spec(spec); spec.loader.exec_module(renderer)
        return renderer

    def report_fixture(self):
        scores = {a: {str(h): {'accuracy': value, 'macro_f1': value-.02} for h in range(1,9)}
                  for a, value in [('A0',.5),('A1',.51),('A2',.515),('A3',.54)]}
        runs = [{'fold': h, 'arm': a, 'metrics': scores[a][str(h)], 'best_epoch': 1,
                 'last_epoch': 2, 'training_seconds': 120., 'replay': {'passed': True},
                 'last_update_verified': True,
                 'last_parameter_updates': {'encoder': {'total': 103, 'changed': 102},
                                           'head': {'total': 6, 'changed': 6},
                                           'aux': {'total': 10, 'changed': 10 if a in ('A2','A3') else 0}}}
                for h in range(1,9) for a in ['A0','A1','A2','A3']]
        return {'status': 'verified', 'errors': [], 'missing': [], 'replay_requested': True,
                'runs': runs, 'summary': self.audit.summarize_scores(scores, repetitions=1000),
                'config': {'seed': 2026, 'per_user': 4, 'alpha': 1e-4, 'epochs': 16,
                           'min_epochs': 6, 'patience': 5, 'warmup_epochs': 40}}

    def test_report_complete_contract_and_protocol_boundaries(self):
        renderer = self.load_renderer()
        report = renderer.render_report(self.report_fixture())
        for phrase in ['已完成', 'A3−A1', '交互', 'bootstrap', '74.773', '六位', '二阶', '固定配置']:
            self.assertIn(phrase, report)
        self.assertIn('54.000', report)

    def test_report_cannot_claim_complete_without_all_replays(self):
        renderer = self.load_renderer()
        fixture = self.report_fixture()
        fixture['runs'][-1]['replay']['passed'] = False
        report = renderer.render_report(fixture)
        self.assertIn('尚未完成', report)
        self.assertNotIn('效果结果已完成', report)
        self.assertNotIn('54.000', report)
        fixture = self.report_fixture(); fixture['summary']['gate'] = None
        self.assertIn('尚未完成', renderer.render_report(fixture))

    def test_report_rejects_duplicate_runs_even_if_count_is_32(self):
        renderer = self.load_renderer()
        fixture = self.report_fixture()
        fixture['runs'][-1] = copy.deepcopy(fixture['runs'][0])
        self.assertIn('尚未完成', renderer.render_report(fixture))

    def test_negative_report_limits_claim_to_fixed_configuration(self):
        renderer = self.load_renderer()
        fixture = self.report_fixture()
        fixture['summary']['gate']['passed'] = False
        report = renderer.render_report(fixture)
        self.assertIn('未达到', report)
        self.assertIn('负结果仅约束', report)
        self.assertNotIn('该结果支持追加独立初始化', report)

    def test_report_explains_epoch_zero_using_verified_last_updates(self):
        renderer = self.load_renderer()
        fixture = self.report_fixture()
        for row in fixture['runs']: row['best_epoch'] = 0
        text = renderer.render_report(fixture)
        self.assertIn('选中第 0 轮', text)
        self.assertIn('不是训练没有执行', text)
        self.assertIn('32/32', text)
        self.assertIn('102/103', text)

    def test_report_requires_last_update_evidence(self):
        renderer = self.load_renderer()
        fixture = self.report_fixture()
        del fixture['runs'][0]['last_parameter_updates']
        self.assertIn('尚未完成', renderer.render_report(fixture))


if __name__ == '__main__':
    unittest.main()
