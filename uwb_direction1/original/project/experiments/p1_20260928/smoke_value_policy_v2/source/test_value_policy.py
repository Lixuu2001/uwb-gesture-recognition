"""Leakage, routing, selection and provenance tests for the conditional pilot."""
import importlib
import copy
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def implementation():
    assert Path(__file__).with_name('value_policy.py').exists(), 'Value policy implementation is missing'
    return importlib.import_module('value_policy')


class ConstantGain:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def predict(self, features):
        return np.repeat(self.values.reshape(1, -1), len(features), axis=0)


def constant_policy(vp, first_gains, pair_gains=(0., 0.), first=0):
    pairs = [i for i in range(3, 6) if first in vp.SUBSETS[i]]
    return vp.GainPolicy('constant_test', first, {
        first: ConstantGain(first_gains),
        pairs[0]: ConstantGain([pair_gains[0]]),
        pairs[1]: ConstantGain([pair_gains[1]])})


class CausalPolicyTest(unittest.TestCase):
    def test_state_features_are_exactly_current_probabilities_entropy_and_margin(self):
        vp = implementation()
        logits = np.zeros((2, 12))
        logits[1, 0] = np.log(11.)
        feature = vp.state_features(logits)
        self.assertEqual(feature.shape, (2, 14))
        np.testing.assert_allclose(feature[0, :12], np.ones(12)/12)
        self.assertAlmostEqual(feature[0, 12], 1.)
        self.assertAlmostEqual(feature[0, 13], 0.)
        self.assertAlmostEqual(feature[1, 0], .5)
        self.assertAlmostEqual(feature[1, 13], .5-1/22)
        np.testing.assert_allclose(vp.state_features(logits+1000), feature, atol=1e-14)

    def test_internal_split_is_disjoint_classwise_deterministic_and_order_invariant(self):
        vp = implementation()
        ids = np.arange(100, 130)
        labels = np.repeat([0, 3, 11], 10)
        split = vp.internal_split(ids, labels, 4)
        self.assertEqual(len(split['fit_indices']), 18)
        self.assertEqual(len(split['tune_indices']), 12)
        self.assertFalse(set(split['fit_indices']) & set(split['tune_indices']))
        self.assertEqual(set(split['fit_indices']) | set(split['tune_indices']), set(range(30)))
        for c in np.unique(labels):
            self.assertEqual(sum(labels[split['fit_indices']] == c), 6)
        order = np.random.default_rng(32).permutation(30)
        reordered = vp.internal_split(ids[order], labels[order], 4)
        np.testing.assert_array_equal(np.sort(ids[split['fit_indices']]),
                                      np.sort(ids[order][reordered['fit_indices']]))

    def test_gain_table_shrinks_to_global_with_twenty_equivalent_examples(self):
        vp = implementation()
        feature = vp.state_features(np.eye(12)[[0, 0, 1]]*5)
        gain = np.array([[1., 0., -1.], [1., 1., 0.], [-1., 0., 1.]])
        model = vp.ClassGainRegressor().fit(feature, gain)
        global_mean = gain.mean(0)
        wanted = (gain[:2].sum(0)+20*global_mean)/22
        np.testing.assert_allclose(model.predict(feature[:1])[0], wanted)
        unseen = vp.state_features(np.eye(12)[[9]]*5)
        np.testing.assert_allclose(model.predict(unseen)[0], global_mean)

    def test_stop_reads_no_unrequested_view_even_if_poisoned(self):
        vp = implementation()
        logits = np.full((6, 7, 12), np.nan)
        logits[:, 0] = np.eye(12)[[0, 1, 2, 3, 4, 5]]*2
        policy = constant_policy(vp, [-1., 0., -.1])
        calls = []
        def reader(rows, subset):
            calls.append((rows.copy(), subset))
            if subset != 0:
                raise AssertionError('An unrequested subset was read')
            return logits[rows, subset]
        result = vp.route_policy(reader, policy, 0., n_rows=6)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(result['prediction'], np.arange(6))
        np.testing.assert_array_equal(result['subset_indices'], np.zeros(6))
        np.testing.assert_array_equal(result['costs'], np.ones(6))
        np.testing.assert_array_equal(result['path'][:, 1:], -np.ones((6, 2)))

    def test_pair_then_last_records_each_new_view_once(self):
        vp = implementation()
        logits = np.zeros((4, 7, 12))
        logits[:, 0, 1] = 2
        logits[:, 3, 2] = 2
        logits[:, 6, 3] = 2
        calls = []
        def reader(rows, subset):
            calls.append(subset)
            return logits[rows, subset]
        result = vp.route_policy(reader, constant_policy(vp, [.5, .2, .1], [.4, -.2]),
                                 .1, n_rows=4)
        self.assertEqual(calls, [0, 3, 6])
        np.testing.assert_array_equal(result['path'], np.tile([0, 3, 6], (4, 1)))
        np.testing.assert_array_equal(result['added_views'], np.ones((4, 3)))
        np.testing.assert_array_equal(result['requested_mask'], np.ones((4, 3)))
        np.testing.assert_array_equal(result['prediction'], np.full(4, 3))
        np.testing.assert_array_equal(result['costs'], np.full(4, 3))

    def test_direct_add_both_does_not_read_intermediate_pairs(self):
        vp = implementation()
        logits = np.full((3, 7, 12), np.nan)
        logits[:, 2] = 0
        logits[:, 6] = np.eye(12)[[1, 2, 3]]*3
        result = vp.route_policy(logits, constant_policy(vp, [.1, .1, 1.], first=2), .1)
        np.testing.assert_array_equal(result['path'], np.tile([2, 6, -1], (3, 1)))
        np.testing.assert_array_equal(result['added_views'], np.tile([1, 2, 0], (3, 1)))
        np.testing.assert_array_equal(result['costs'], result['added_views'].sum(1))
        np.testing.assert_array_equal(result['prediction'], [1, 2, 3])

    def test_action_ties_prefer_stop_then_cheaper_then_lower_subset(self):
        vp = implementation()
        logits = np.zeros((1, 7, 12))
        stopped = vp.route_policy(logits, constant_policy(vp, [.1, .1, .2]), .1)
        self.assertEqual(stopped['subset_indices'][0], 0)
        paired = vp.route_policy(logits, constant_policy(vp, [.3, .3, .4]), .1)
        self.assertEqual(paired['subset_indices'][0], 3)

    def test_fitting_gain_targets_accounts_for_harmful_and_joint_additions(self):
        vp = implementation()
        logits = np.zeros((8, 7, 12))
        labels = np.zeros(8, dtype=int)
        logits[:, :, 1] = 2
        logits[:4, 0, 0] = 3  # first correct on half
        logits[4:, 3, 0] = 3  # pair replaces four correct with four different ones
        logits[:, 6, 0] = 3   # both views recover all errors
        policy = vp.fit_policy(logits, labels, first=0, method='class_gain')
        # An unseen predicted class gets exactly the global mean gain.
        gain = policy.predict_gain(0, np.eye(12)[[9]]*3)[0]
        np.testing.assert_allclose(gain, [0., -.5, .5])

    def test_extra_trees_configuration_and_predictions_are_reproducible(self):
        vp = implementation()
        rng = np.random.default_rng(71)
        logits = rng.normal(size=(60, 7, 12))
        labels = rng.integers(0, 12, 60)
        a = vp.fit_policy(logits, labels, 1, 'extra_trees')
        b = vp.fit_policy(logits, labels, 1, 'extra_trees')
        params = a.models[1].get_params()
        for key, value in {'n_estimators': 200, 'max_depth': 4, 'min_samples_leaf': 20,
                           'random_state': 2026, 'n_jobs': 1}.items():
            self.assertEqual(params[key], value)
        np.testing.assert_array_equal(a.predict_gain(1, logits[:, 1]),
                                      b.predict_gain(1, logits[:, 1]))

    def test_price_ties_choose_less_cost_then_larger_price(self):
        vp = implementation()
        logits = np.zeros((10, 7, 12))
        labels = np.zeros(10, dtype=int)
        selection, grid = vp.select_price(constant_policy(vp, [.4, .4, .4]), logits, labels)
        self.assertEqual(selection['price'], 2.)
        self.assertEqual(selection['mean_views'], 1.)
        self.assertEqual(len(grid), 13)
        self.assertTrue(all(row['mean_views'] >= 1 for row in grid))


class ProtocolTest(unittest.TestCase):
    def api(self, name):
        vp = implementation()
        self.assertTrue(hasattr(vp, name), 'Missing protocol API: '+name)
        return vp

    def synthetic_validation(self):
        rng = np.random.default_rng(384)
        labels = np.repeat(np.arange(3), 20)
        logits = rng.normal(size=(60, 7, 12))
        logits[np.arange(60)[:, None], np.arange(7)[None, :], labels[:, None]] += 1
        return {'ids': np.arange(100, 160), 'y': labels, 'logits': logits}

    def frozen_selection(self):
        return {'curves': {'2.0': {'params': {'first': 0, 'pair': 3,
                    'mode': 'confidence', 'threshold1': .8, 'threshold2': .8}}},
                'validation_fixed_accuracy': [.6, .5, .4, .7, .6, .65, .8],
                'uses_target_labels': False}

    def completed_fixture(self, vp, directory, fold=1, smoke=True):
        validation = self.synthetic_validation()
        target = {'ids': np.arange(200, 215), 'y': validation['y'][::4].copy(),
                  'logits': validation['logits'][::4].copy()}
        identity = {'fold': fold, 'smoke': smoke, 'source': {'synthetic_test': 'fixed'}}
        result = vp.run_arrays(directory, fold, identity, validation,
                               lambda: target, self.frozen_selection())
        return result, vp.subset_accuracies(target['logits'], target['y'])

    def test_selection_is_saved_before_target_load_and_is_invariant_to_target_labels(self):
        vp = self.api('run_arrays')
        validation = self.synthetic_validation()
        selections, routes, mixtures = [], [], []
        with tempfile.TemporaryDirectory() as temp:
            for shift in (0, 1):
                directory = Path(temp)/str(shift)
                def target_loader():
                    self.assertTrue((directory/'selection.json').is_file())
                    self.assertTrue((directory/'models.pkl').is_file())
                    return {'ids': np.arange(200, 215),
                            'y': (validation['y'][::4]+shift) % 3,
                            'logits': validation['logits'][::4].copy()}
                result = vp.run_arrays(directory, 1, {'smoke': True}, validation,
                                       target_loader, self.frozen_selection())
                selections.append(json.loads((directory/'selection.json').read_text()))
                with np.load(directory/'predictions.npz') as artifact:
                    routes.append(artifact['extra_trees_path'].copy())
                    mixtures.append(artifact['fixed_mixture_weights'].copy())
                self.assertEqual(result['n_test'], 15)
                self.assertEqual(set(result['files']), {'selection.json', 'models.pkl',
                                                       'predictions.npz', 'fit_log.json'})
                self.assertFalse(result['can_claim_speedup'])
            self.assertEqual(selections[0], selections[1])
            np.testing.assert_array_equal(routes[0], routes[1])
            np.testing.assert_array_equal(mixtures[0], mixtures[1])

    def test_resume_rejects_tampered_source_identity_and_artifacts(self):
        vp = self.api('checked_existing')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            saved, fixed = self.completed_fixture(vp, directory)
            identity = saved['identity']
            with self.assertRaisesRegex(ValueError, 'identity'):
                vp.checked_existing(directory, {'source': {'value_policy.py': 'new-hash'}, 'smoke': True})
            (directory/'models.pkl').write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'artifact'):
                vp.checked_existing(directory, identity)

    def test_completed_result_only_corruption_is_rejected_without_refit(self):
        vp = self.api('checked_existing')
        self.assertIn('budget_fixed_accuracy', inspect.signature(vp.checked_existing).parameters,
                      'Completed result reconciliation lacks a verified budget reference')
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            original, fixed = self.completed_fixture(vp, directory)
            self.assertEqual(vp.checked_existing(directory, original['identity'], fixed), original)
            mutations = [
                (('fold',), 2), (('n_test',), 999),
                (('selected_prices', 'extra_trees', 'price'), .123),
                (('protocol', 'main_candidate'), 'changed'),
                (('can_claim_speedup',), True), (('runtime_validation',), 'already measured'),
                (('metrics', 'extra_trees', 'macro_accuracy'), .999),
                (('metrics', 'class_gain', 'mean_views'), 2.999),
                (('metrics', 'frozen_budget2_cascade', 'view_count_histogram'), [0, 0, 15]),
                (('metrics', 'three_views', 'macro_accuracy'), .999),
                (('metrics', 'fixed_mixture_within_candidate_cost', 'macro_accuracy'), .999),
                (('metrics', 'fixed_mixture_within_candidate_cost', 'weights'), [0., 0., 0., 0., 0., 0., 1.]),
                (('metrics', 'fixed_mixture_within_candidate_cost', 'mean_views'), 2.999),
                (('metrics', 'fixed_mixture_within_candidate_cost', 'cost_ceiling'), 2.999),
                (('fixed_subset_accuracy',), [.999]*7)]
            for path, value in mutations:
                changed = copy.deepcopy(original)
                parent = changed
                for key in path[:-1]:
                    parent = parent[key]
                if parent[path[-1]] == value:
                    continue
                parent[path[-1]] = value
                vp.dump(directory/'result.json', changed)
                with self.subTest(field='.'.join(path)), self.assertRaisesRegex(ValueError, 'result'):
                    vp.checked_existing(directory, original['identity'], fixed)
            vp.dump(directory/'result.json', original)
            self.assertEqual(vp.checked_existing(directory, original['identity'], fixed), original)

    def test_summary_rejects_result_only_corruption_on_disk_and_in_memory(self):
        vp = self.api('summarize')
        self.assertIn('budget_fixed_accuracy', inspect.signature(vp.summarize).parameters,
                      'Scientific summary does not revalidate verified fold artifacts')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, references = [], {}
            for fold in range(1, 9):
                row, fixed = self.completed_fixture(vp, root/f'fold{fold}', fold, smoke=False)
                rows.append(row)
                references[fold] = fixed
            original_summary = vp.summarize(rows, root, references)
            self.assertFalse(original_summary['can_claim_speedup'])
            changed = copy.deepcopy(rows)
            for row in changed:
                row['metrics']['extra_trees'].update(macro_accuracy=1., mean_views=1.)
                row['can_claim_speedup'] = True
            with self.assertRaisesRegex(ValueError, 'result|Summary'):
                vp.summarize(changed, root, references)
            vp.dump(root/'fold1'/'result.json', changed[0])
            with self.assertRaisesRegex(ValueError, 'result|Summary'):
                vp.summarize(rows, root, references)

    def test_formal_no_oracle_guard_rejects_before_missing_model_inputs_are_opened(self):
        vp = self.api('preflight')
        with tempfile.TemporaryDirectory() as temp:
            budget = Path(temp)/'budget'
            vp.dump(budget/'summary.json', {'state': 'complete',
                    'opportunity_decision': 'STOP_NO_ORACLE_ROOM'})
            with self.assertRaisesRegex(ValueError, 'opportunity'):
                vp.preflight(Path(temp)/'nonexistent_models', budget, [1], False)

    def test_opportunity_guard_recomputes_both_conditions_and_rejects_smoke(self):
        vp = self.api('validate_opportunity')
        rows = [{'fold': f, 'identity': {'smoke': False},
                 'oracle_at_2': {'macro_accuracy': .85},
                 'fixed_mixture_at_2': {'accuracy': .8},
                 'cascade_curves': {'2.0': {'test_accuracy': .78,
                     'oracle_at_actual_cost': {'macro_accuracy': .84}}}}
                for f in range(1, 9)]
        summary = {'state': 'complete', 'opportunity_decision': 'POLICY_PILOT_WARRANTED',
                   'oracle_minus_mixture_at_2_pp': 5., 'oracle_minus_cascade_at_same_cost_pp': 6.}
        vp.validate_opportunity(summary, rows)
        altered = dict(summary, oracle_minus_mixture_at_2_pp=50.)
        with self.assertRaisesRegex(ValueError, 'summary'):
            vp.validate_opportunity(altered, rows)
        rows[0]['identity']['smoke'] = True
        with self.assertRaisesRegex(ValueError, 'smoke|Smoke'):
            vp.validate_opportunity(summary, rows)

    def test_output_guard_protects_inputs_and_formal_output_from_smoke(self):
        vp = self.api('validate_output_paths')
        models = Path('C:/synthetic/models')
        budget = Path('C:/synthetic/budget')
        with self.assertRaisesRegex(ValueError, 'input'):
            vp.validate_output_paths(models, budget, models/'nested', False)
        with self.assertRaisesRegex(ValueError, 'Smoke'):
            vp.validate_output_paths(models, budget, vp.ROOT/'value_policy_s2026', True)
        vp.validate_output_paths(models, budget, Path('C:/synthetic/separate'), True)

    def test_source_record_freezes_the_comparator_clarification(self):
        vp = implementation()
        source = vp.source_record()
        self.assertIn('POLICY_COMPARATOR_CLARIFICATION.md',
                      {Path(path).name for path in source['files']})

    def test_budget_result_tampering_after_preflight_is_rejected(self):
        vp = implementation()
        with tempfile.TemporaryDirectory() as temp:
            models, budget, out = (Path(temp)/name for name in ('models', 'budget', 'out'))
            (models/'fold1').mkdir(parents=True)
            predictions = models/'fold1'/'predictions.npz'
            predictions.write_bytes(b'not needed: reject before loading')
            vp.dump(budget/'fold1'/'result.json', {'original': True})
            vp.dump(budget/'fold1'/'selection.json', {})
            original = vp.digest(budget/'fold1'/'result.json')
            row = {'identity': {'predictions_sha256': vp.digest(predictions)},
                   'files': {'selection.json': vp.digest(budget/'fold1'/'selection.json')}}
            context = {'models': str(models.resolve()), 'budget': str(budget.resolve()),
                       'source': vp.source_record(), 'smoke': True, 'summary_sha256': None,
                       'rows': {1: row}, 'selections': {1: {}},
                       'budget_result_sha256': {'1': original}}
            vp.dump(budget/'fold1'/'result.json', {'original': False})
            with self.assertRaisesRegex(ValueError, 'Budget result changed'):
                vp.run_fold(models, budget, out, 1, context)

    def test_budget_gate_numbers_are_recomputed_not_trusted_from_editable_json(self):
        vp = self.api('validate_budget_numbers')
        from budget_analysis import tune_cascades, subset_accuracies, route_cascade, fixed_mixture, oracle_bound
        validation = self.synthetic_validation()
        z, y = validation['logits'], validation['y']
        curves, val_accuracy = tune_cascades(z, y)
        target_logits, target_labels = z[::4], y[::4]
        route = route_cascade(target_logits, curves['2.0']['params'])
        cost = float(route['costs'].mean())
        fixed = subset_accuracies(target_logits, target_labels)
        first = curves['2.0']['params']['first']
        mixture = fixed_mixture(val_accuracy, 2.)
        row = {'fixed_accuracy': fixed.tolist(),
               'fixed_mixture_at_2': {'accuracy': float(mixture @ fixed),
                                      'cost': float(mixture @ vp.COSTS), 'weights': mixture.tolist()},
               'oracle_at_2': oracle_bound(target_logits, target_labels, first, 2.),
               'cascade_curves': {'2.0': {'test_cost': cost,
                    'test_accuracy': vp.macro_accuracy(route['prediction'], target_labels),
                    'oracle_at_actual_cost': oracle_bound(target_logits, target_labels, first, cost)}}}
        selection = {'curves': curves, 'validation_fixed_accuracy': val_accuracy.tolist()}
        data = {'logits': np.concatenate([z, target_logits]), 'y': np.r_[y, target_labels],
                'role': np.r_[np.ones(len(y)), np.full(len(target_labels), 2)]}
        vp.validate_budget_numbers(data, row, selection)
        row['fixed_mixture_at_2']['accuracy'] += .1
        with self.assertRaisesRegex(ValueError, 'statistics'):
            vp.validate_budget_numbers(data, row, selection)

    def gate_rows(self):
        return [{'fold': f, 'identity': {'smoke': False, 'source': {'code': 'fixed'}},
                 'n_test': 100,
                 'metrics': {
                     'extra_trees': {'macro_accuracy': .80, 'mean_views': 1.9},
                     'class_gain': {'macro_accuracy': .75, 'mean_views': 2.0},
                     'frozen_budget2_cascade': {'macro_accuracy': .74, 'mean_views': 2.1},
                     'fixed_mixture_within_candidate_cost': {'macro_accuracy': .73, 'mean_views': 1.8},
                     'three_views': {'macro_accuracy': .805, 'mean_views': 3.}}}
                for f in range(1, 9)]

    def test_summary_requires_every_comparator_and_cost_fairness_without_target_winner(self):
        vp = self.api('_aggregate_metrics')
        rows = self.gate_rows()
        with tempfile.TemporaryDirectory() as temp:
            summary = vp._aggregate_metrics(rows)
            self.assertTrue(all(summary['gate'].values()))
            self.assertEqual(summary['decision'], 'STATISTICAL_GATE_PASSED_RUNTIME_PENDING')
            self.assertFalse(summary['can_claim_speedup'])
            self.assertEqual(len(summary['comparisons']), 3)
            for row in rows:
                row['metrics']['class_gain']['macro_accuracy'] = .80
            failed = vp._aggregate_metrics(rows)
            self.assertFalse(failed['gate']['gain_at_least_2pp_vs_class_gain'])
            self.assertEqual(failed['decision'], 'STOP_VALUE_POLICY_MECHANISM')
            for row in rows:
                row['metrics']['class_gain']['macro_accuracy'] = .75
                row['metrics']['frozen_budget2_cascade']['mean_views'] = 1.8
            expensive = vp._aggregate_metrics(rows)
            self.assertFalse(expensive['gate']['population_cost_no_greater_than_frozen_budget2_cascade'])
            self.assertEqual(expensive['fairness']['frozen_budget2_cascade']['per_user_cost_no_greater'], [False]*8)

    def test_summary_rejects_partial_and_smoke_scientific_results(self):
        vp = self.api('summarize')
        rows = self.gate_rows()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'eight|Eight'):
                vp.summarize(rows[:7], Path(temp))
            rows[0]['identity']['smoke'] = True
            with self.assertRaisesRegex(ValueError, 'smoke|Smoke'):
                vp.summarize(rows, Path(temp))


if __name__ == '__main__':
    unittest.main()
