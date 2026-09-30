"""CPU tests for report rules that could change scientific conclusions."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
try:
    import report as R
except ImportError:
    R = None


class ReportRules(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(R, 'The report validation module must be implemented')

    def test_support_repeats_are_averaged_before_user_bootstrap(self):
        candidate = np.array([[1., 0., 0.], [1., 1., 1.]])
        baseline = np.zeros_like(candidate)
        row = R.paired_user_effect(candidate, baseline)
        self.assertEqual(row['n_users'], 2)
        np.testing.assert_allclose(row['per_user_difference_pp'], [100/3, 100])
        self.assertAlmostEqual(row['mean_difference_pp'], 200/3)
        self.assertEqual(row['bootstrap_unit'], 'user')

    def test_fixed12_and_present_class_f1_are_distinct(self):
        y, pred = np.array([0, 0, 1, 1]), np.array([0, 0, 1, 2])
        self.assertAlmostEqual(R.macro_f1(pred, y, [0, 1]), 5/6)
        self.assertAlmostEqual(R.macro_f1(pred, y, np.arange(12)), 5/36)

    def test_uncalibrated_f1_filters_queries_before_scoring(self):
        row = R.classification_metrics(np.array([0, 0, 0, 1]),
                                       np.array([0, 0, 1, 1]), [0])
        self.assertAlmostEqual(row['uncalibrated'], .5)
        self.assertAlmostEqual(row['f1_uncalibrated'], 2/3)

    def test_no_rounding_can_rescue_an_over_budget_strategy(self):
        self.assertFalse(R.within_budget(2.0046155303, 2))
        self.assertFalse(R.within_budget(2.00001, 2))
        self.assertTrue(R.within_budget(2, 2))

    def test_rejects_duplicate_or_wrong_raw_identity(self):
        y, s = np.array([0, 1, 2]), np.array([1, 1, 1])
        with self.assertRaisesRegex(ValueError, 'unique'):
            R.validate_ids(np.array([0, 0]), [0, 0], y, s, 1)
        with self.assertRaisesRegex(ValueError, 'label'):
            R.validate_ids(np.array([0, 1]), [0, 2], y, s, 1)
        with self.assertRaisesRegex(ValueError, 'user'):
            R.validate_ids(np.array([0, 1]), [0, 1], y, s, 2)

    def test_cross_route_uses_requested_order_and_rejects_missing_or_wrong_labels(self):
        data = {'ids': np.array([9, 5, 7]), 'y': np.array([2, 0, 1]),
                'subjects': np.array([1, 1, 1]), 'role': np.array([2, 2, 2]),
                'logits': np.arange(3*7*12).reshape(3, 7, 12)}
        aligned = R.align_query(data, np.array([5, 9]), np.array([0, 2]), 1)
        np.testing.assert_array_equal(aligned, data['logits'][[1, 0]])
        with self.assertRaisesRegex(ValueError, 'missing'):
            R.align_query(data, [5, 8], [0, 2], 1)
        with self.assertRaisesRegex(ValueError, 'label'):
            R.align_query(data, [5, 9], [0, 1], 1)

    def test_result_comparison_does_not_accept_changed_or_nonfinite_values(self):
        R.reconcile({'a': [1., 2.]}, {'a': [1., 2.]}, 'sample')
        with self.assertRaisesRegex(ValueError, 'sample.a'):
            R.reconcile({'a': 2.01}, {'a': 2.}, 'sample')
        with self.assertRaises(ValueError):
            R.reconcile({'a': float('nan')}, {'a': 2.}, 'sample')

    def test_smoke_or_missing_eight_fold_inputs_block_formal_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp), Path(tmp)/'deliverable'
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                R.generate(root, out)
            self.assertFalse(out.exists())
        with self.assertRaisesRegex(ValueError, 'smoke'):
            R.require_eight_folds([{'fold': f, 'identity': {'smoke': f == 2}}
                                  for f in range(1, 9)])
        with self.assertRaisesRegex(ValueError, 'eight'):
            R.require_eight_folds([{'fold': f, 'identity': {'smoke': False}}
                                  for f in [1, 2, 3, 4, 5, 6, 7, 7]])

    def test_hash_validation_rejects_tampering_and_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'predictions.npz'
            path.write_bytes(b'original')
            inventory = R.HashInventory()
            sha = inventory.verify(path)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'hash'):
                R.verify_artifacts(root, {'predictions.npz': sha}, inventory)
            with self.assertRaisesRegex(ValueError, 'escapes'):
                R.verify_artifacts(root, {'../other': sha}, inventory)

    def test_idle_absent_is_none_and_joint_rescue_is_not_single_extra_rescue(self):
        y = np.array([0, 1, 2])
        preds = np.array([[1, 1, 1, 1, 1, 1, 0],
                          [1, 1, 1, 2, 2, 2, 2],
                          [0, 0, 0, 2, 0, 0, 2]])
        row = R.rescue_decomposition(preds, y, 0)
        self.assertEqual(row['joint_only_rescue'], 1)
        self.assertEqual(row['single_wrong_full_correct'], 2)
        self.assertEqual(row['single_correct_full_wrong'], 1)
        group = R.group_metrics(preds[:, 6], np.full(3, 3), y)
        self.assertIsNone(group['idle'])
        self.assertEqual(group['gesture']['n'], 3)

    def test_oracle_subgroups_keep_full_fold_label_informed_allocation(self):
        self.assertTrue(hasattr(R, 'oracle_group_metrics'), 'Missing full-fold oracle subgroup audit')
        labels = np.array([0, 0, 1, 11])
        logits = np.zeros((4, 7, 12))
        logits[:, :, 2] = 5
        logits[0, 0, 0] = 10
        logits[1, 3, 0] = 10
        logits[2, 6, 1] = 10
        logits[3, 4, 11] = 10
        groups, overall = R.oracle_group_metrics(logits, labels, first=0, budget=1.25)
        self.assertAlmostEqual(overall['macro_accuracy'], .5)
        self.assertAlmostEqual(overall['mean_views'], 1.25)
        self.assertAlmostEqual(groups['gesture']['macro_accuracy'], .25)
        self.assertAlmostEqual(groups['gesture']['mean_views'], 1.)
        self.assertAlmostEqual(groups['idle']['macro_accuracy'], 1.)
        self.assertAlmostEqual(groups['idle']['mean_views'], 2.)

    def test_latency_summary_comes_from_same_clips_and_rejects_duplicate_or_replay_failure(self):
        rows = [dict(fold=1, clip_id=i, label=i, method='fixed1', repeat=0,
                     elapsed_ms=float(i+1), actual_views=1, subset_index=0,
                     raw_accesses=[0], encoded_views=[0], path=[0, -1, -1],
                     replay={'bitwise_equal': True}, final_logits=[1.]+[0.]*11,
                     prediction=0, offline_prediction_differs=False, offline_path_differs=False)
                for i in range(3)]
        summary = R.validate_latency_rows(rows, {1: [0, 1, 2]}, ['fixed1'],
                                          np.array([0, 1, 2]), np.array([1, 1, 1]), 1)
        self.assertEqual(summary['fixed1']['mean_ms'], 2.)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            R.validate_latency_rows(rows+[rows[0]], {1: [0, 1, 2]}, ['fixed1'],
                                    np.array([0, 1, 2]), np.array([1, 1, 1]), 1)
        rows[0]['replay']['bitwise_equal'] = False
        with self.assertRaisesRegex(ValueError, 'replay'):
            R.validate_latency_rows(rows, {1: [0, 1, 2]}, ['fixed1'],
                                    np.array([0, 1, 2]), np.array([1, 1, 1]), 1)

    def test_budget_fixed_selection_uses_validation_even_when_target_prefers_other_view(self):
        self.assertTrue(hasattr(R, 'compute_budget'), 'Missing raw-array budget recomputation')
        y = np.tile([0, 1], 6)
        logits = np.zeros((12, 7, 12))
        for row in range(12):
            for subset in range(7):
                correct = (row < 6 and subset == 0) or (row >= 6 and subset == 2)
                logits[row, subset, y[row] if correct else 1-y[row]] = 5
        data = dict(ids=np.arange(12), y=y, role=np.r_[np.ones(6), np.full(6, 2)], logits=logits)
        row, selection, arrays = R.compute_budget(data, 1)
        self.assertEqual(row['fixed_validation_selected']['1'], 0)
        self.assertEqual(row['fixed_accuracy'][0], 0.)
        self.assertEqual(row['fixed_accuracy'][2], 1.)
        self.assertFalse(selection['uses_target_labels'])

    def test_a_gate_preserves_per_user_harm_even_when_average_gain_exceeds_two_pp(self):
        self.assertTrue(hasattr(R, 'summarize_a'), 'Missing user-level A summary')
        rows = []
        for fold in range(1, 9):
            methods = {}
            for method in R.A_METHODS:
                value = .69 if method == 'lowrank' and fold == 8 else (.77 if method == 'lowrank' else .72)
                metrics = [dict(all=value, calibrated=value, uncalibrated=value,
                                uncalibrated_center_mse=1., f1_all_present=value,
                                f1_all_fixed12=value, f1_uncalibrated=value) for _ in range(20)]
                methods[method] = dict(metrics=metrics, support_budget=36 if method == 'allclass_lowrank' else 9)
            rows.append(dict(fold=fold, identity={'smoke': False}, methods=methods))
        row = R.summarize_a(rows)
        self.assertGreater(row['lowrank_vs_simple']['mean_difference_pp'], 2.)
        self.assertEqual(row['lowrank_vs_simple']['positive_users'], 7)
        self.assertFalse(row['gate']['no_user_worse_than_minus2pp_vs_simple'])
        self.assertEqual(row['decision'], 'STOP_LOW_RANK_MECHANISM')

    def test_policy_cannot_pass_by_beating_only_the_favorable_comparator(self):
        self.assertTrue(hasattr(R, 'summarize_policy'), 'Missing fixed-comparator gate')
        rows = []
        for fold in range(1, 9):
            pairs = {'extra_trees': (.91, 2.007), 'class_gain': (.875, 1.786),
                     'frozen_budget2_cascade': (.92, 2.004),
                     'fixed_mixture_within_candidate_cost': (.885, 2.007), 'three_views': (.935, 3.)}
            rows.append({'fold': fold, 'n_test': 100, 'identity': {'smoke': False},
                         'metrics': {k: dict(macro_accuracy=a, mean_views=c) for k, (a, c) in pairs.items()}})
        result = R.summarize_policy(rows)
        self.assertEqual(result['decision'], 'STOP_VALUE_POLICY_MECHANISM')
        self.assertTrue(result['gate']['gain_at_least_2pp_vs_fixed_mixture_within_candidate_cost'])
        self.assertFalse(result['gate']['gain_at_least_2pp_vs_frozen_budget2_cascade'])
        self.assertFalse(result['gate']['population_cost_no_greater_than_class_gain'])
        self.assertFalse(result['gate']['mean_actual_views_at_most_2'])

    def test_formal_latency_accepts_disclosed_desktop_context_but_not_cpu_or_smoke(self):
        self.assertTrue(hasattr(R, 'validate_latency_mode'), 'Missing formal latency environment check')
        summary = {'state': 'complete', 'folds': list(range(1, 9)),
                   'classification': 'desktop_context_gpu_latency',
                   'environment': {'device': 'cuda', 'batch': 1, 'attention': 'MATH',
                                   'autocast': 'bfloat16', 'tf32': False, 'gpu_exclusive': False},
                   'configuration': {'allow_smoke': False, 'warmup': 10, 'repeats': 1},
                   'batch_one_replay': 'bitwise_equal'}
        R.validate_latency_mode(summary, has_environment_note=True)
        with self.assertRaisesRegex(ValueError, 'environment note'):
            R.validate_latency_mode(summary, has_environment_note=False)
        summary['environment']['device'] = 'cpu'
        with self.assertRaisesRegex(ValueError, 'CUDA'):
            R.validate_latency_mode(summary, has_environment_note=True)

    def test_independent_policy_audit_is_bound_to_verified_files(self):
        self.assertTrue(hasattr(R, 'audit_independent_policy'), 'Missing independent policy evidence audit')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'p1_20260928'
            directory = root/'value_policy_s2026'
            directory.mkdir(parents=True)
            source = directory/'independent_audit.py'
            evidence = directory/'fold1'/'result.json'
            evidence.parent.mkdir()
            old_data = root.parent/'p1_20260905'/'prepared'/'y.npy'
            old_data.parent.mkdir(parents=True)
            old_data.write_bytes(b'frozen labels')
            source.write_text('audit source', encoding='utf-8')
            evidence.write_text('{}', encoding='utf-8')
            digest = R.A.digest
            saved = {'status': 'PASS', 'failures': [], 'checks': {'replay': 2},
                     'checks_total': 2, 'counts': {'outer_folds': 8,
                     'extra_trees_models_independently_refit': 24,
                     'individual_trees_independently_refit_and_compared': 4800},
                     'script_sha256': digest(source),
                     'files': {str(evidence): {'sha256': digest(evidence)},
                               str(old_data): {'sha256': digest(old_data)}}}
            (directory/'INDEPENDENT_AUDIT.json').write_text(json.dumps(saved), encoding='utf-8')
            record = R.audit_independent_policy(root, R.HashInventory())
            self.assertEqual(record['checks'], 2)
            self.assertEqual(record['status'], 'PASS')
            evidence.write_text('{"changed": true}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash'):
                R.audit_independent_policy(root, R.HashInventory())

    def test_independent_latency_audit_requires_complete_evidence_counts(self):
        self.assertTrue(hasattr(R, 'audit_independent_latency'), 'Missing independent latency audit citation')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root/'latency_audit_s2026'
            directory.mkdir()
            (directory/'independent_audit.py').write_text('audit source', encoding='utf-8')
            saved = {'status': 'PASS', 'issues': [], 'checks': {'replay': 2}, 'total_checks': 2,
                     'evidence': {'frozen_files_sha256': 19, 'clips': 279,
                                  'observations': 1953, 'methods': 7, 'folds': 8,
                                  'preprocess_view_bitwise_checks': 837, 'environment_checks': 17}}
            path = directory/'INDEPENDENT_AUDIT.json'
            path.write_text(json.dumps(saved), encoding='utf-8')
            self.assertEqual(R.audit_independent_latency(root, R.HashInventory())['checks'], 2)
            saved['evidence']['clips'] = 278
            path.write_text(json.dumps(saved), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'latency audit'):
                R.audit_independent_latency(root, R.HashInventory())

    def test_report_figure_link_uses_final_absolute_directory(self):
        self.assertTrue(hasattr(R, 'report_figure_link'), 'Missing final-output figure link')
        with tempfile.TemporaryDirectory() as tmp:
            final = Path(tmp)/'report_s2026_final'
            link = R.report_figure_link(final, 'A_user_effects.png')
            self.assertEqual(link, f'![A逐用户配对效应]({final.resolve().as_posix()}/A_user_effects.png)')


if __name__ == '__main__':
    unittest.main()
