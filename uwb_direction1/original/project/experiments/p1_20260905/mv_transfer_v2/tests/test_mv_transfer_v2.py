"""Boundary tests for the multi-view supervised, single-radar deployed study.

Each test is written to fail loudly on the specific mistake it guards against:
leaked views, silently shrunk relation groups, cancelled phase energy, an
auxiliary head that changes the deployed logits.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import protocol as P                                        # noqa: E402
import relation as REL                                      # noqa: E402
import sampling                                             # noqa: E402
import motion_targets as MT                                 # noqa: E402
import teacher_targets                                      # noqa: E402
from model_adapter import (build_student, classification_features,   # noqa: E402
                           student_aux_features, teacher_compact)
from objectives import classification_loss                   # noqa: E402
from process import process_loss                             # noqa: E402

CUDA = torch.cuda.is_available()


class ProtocolTests(unittest.TestCase):
    def test_every_quality_row_is_tested_exactly_once(self):
        manifest = P.verify_manifest(write=False)
        self.assertEqual(manifest['quality_rows'], 9300)
        self.assertEqual(sum(f['n_test'] for f in manifest['folds'].values()), 9300)

    def test_incomplete_class_folds_are_recorded_not_hidden(self):
        manifest = P.verify_manifest(write=False)
        self.assertEqual(len(manifest['folds'][2]['test_classes']), 10)
        self.assertEqual(len(manifest['folds'][6]['test_classes']), 11)
        self.assertEqual(len(manifest['folds'][1]['test_classes']), 12)

    def test_inner_splits_never_share_a_user(self):
        y, subj = P.labels()
        tr, va, te = P.fold_split(1, y, subj)
        seen = []
        for user, fit, probe in P.inner_splits(tr, subj):
            self.assertNotIn(user, set(subj[fit].tolist()))
            self.assertEqual(set(subj[probe].tolist()), {user})
            self.assertEqual(len(fit) + len(probe), len(tr))
            self.assertFalse(set(probe) & set(va) or set(probe) & set(te))
            seen.extend(probe.tolist())
        np.testing.assert_array_equal(np.sort(seen), tr)

    def test_base_objective_matches_the_existing_global_control(self):
        from label_uniform import label_loss
        matrix_np, mass, _ = P.global_matrix(1)
        matrix = torch.tensor(matrix_np)
        logits = torch.randn(16, 12)
        labels = torch.randint(0, 12, (16,))
        expected = torch.nn.functional.cross_entropy(logits, labels) + .5 * label_loss(logits, labels, matrix)
        torch.testing.assert_close(classification_loss(logits, labels, matrix), expected)
        self.assertAlmostEqual(mass, float((1 + matrix_np[0, 0]) / 2), places=6)

    def test_provenance_refuses_to_default_a_missing_source(self):
        with self.assertRaises(KeyError):
            P.provenance(student_input_views=['Right'])


@unittest.skipUnless(CUDA, 'needs the training GPU')
class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.model, self.heads, self.fingerprint = build_student(2026, relation=True, process=True)
        self.model.eval()
        self.x = torch.randn(4, 1, 90, 378, device='cuda')

    def test_adapter_reproduces_the_deployed_logits_exactly(self):
        with torch.no_grad():
            reference, _ = self.model(self.x)
            logits, merged, temporal, spatial = classification_features(self.model, self.x)
        self.assertEqual(float((reference - logits).abs().max()), 0.0)
        self.assertEqual(tuple(merged.shape), (4, 14976))
        self.assertEqual(tuple(temporal.shape), (4, 1, 90, 32))
        self.assertEqual(tuple(spatial.shape), (4, 1, 378, 32))

    def test_378_is_feature_concatenation_not_extra_range(self):
        self.assertEqual(self.model.encoder.M, 2 * P.RANGE_BINS)
        self.assertEqual(P.RANGE_BINS, 189)

    def test_backbone_initialisation_is_identical_across_arms(self):
        plain, _, plain_fp = build_student(2026)
        both, _, both_fp = build_student(2026, relation=True, process=True)
        self.assertEqual(plain_fp, both_fp)
        self.assertEqual(plain_fp, self.fingerprint)
        other, _, other_fp = build_student(2027)
        self.assertNotEqual(other_fp, plain_fp)

    def test_student_never_accepts_the_hidden_views(self):
        triplet = torch.randn(2, 3, 90, 378, device='cuda')
        with self.assertRaises(Exception):
            with torch.no_grad():
                _, _, temporal, spatial = classification_features(self.model, triplet)
                student_aux_features(temporal, spatial)

    def test_teacher_compact_matches_the_reconstruction_definition(self):
        temporal = torch.randn(2, 3, 90, 32, device='cuda')
        spatial = torch.randn(2, 3, 189, 32, device='cuda')
        code = teacher_compact(temporal, spatial)
        self.assertEqual(tuple(code.shape), (2, 3, 992))
        expected = torch.cat([temporal.reshape(2, 3, 10, 9, 32).mean(3).flatten(2),
                              spatial.reshape(2, 3, 21, 9, 32).mean(3).flatten(2)], -1)
        torch.testing.assert_close(code, expected)
        with self.assertRaises(ValueError):
            teacher_compact(temporal, torch.randn(2, 3, 378, 32, device='cuda'))

    def test_process_head_removal_leaves_the_logits_untouched(self):
        with torch.no_grad():
            before, _ = self.model(self.x)
        for parameter in self.heads.parameters():
            parameter.data.add_(1.0)
        with torch.no_grad():
            after, _ = self.model(self.x)
        self.assertEqual(float((before - after).abs().max()), 0.0)


class SamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.y, cls.subj = P.labels()
        cls.tr, cls.va, cls.te = P.fold_split(1, cls.y, cls.subj)
        cls.groups = sampling.relation_epochs(cls.y, cls.subj, cls.tr, 1, P.SAMPLER_SEED + 1)

    def test_group_structure_and_no_duplicate_inside_a_group(self):
        self.assertEqual(self.groups.shape[-1], 32)
        for group in self.groups.reshape(-1, 32):
            self.assertEqual(len(set(group.tolist())), 32)
            self.assertEqual(len(set(self.y[group].tolist())), 4)
            self.assertEqual(len(set(self.subj[group].tolist())), 4)

    def test_every_anchor_has_cross_user_positives(self):
        for group in self.groups.reshape(-1, 32)[:64]:
            labels = torch.tensor(self.y[group].astype('int64'))
            subjects = torch.tensor(self.subj[group].astype('int64'))
            allowed, positive = REL.cross_user_masks(labels, subjects)
            self.assertGreaterEqual(int(positive.sum(1).min()), 6)
            self.assertGreaterEqual(int(allowed.sum(1).min()), 24)

    def test_incomplete_class_users_never_enter_an_unsupported_class(self):
        for group in self.groups.reshape(-1, 32):
            for label, user in zip(self.y[group], self.subj[group]):
                self.assertGreater(((self.y[self.tr] == label) & (self.subj[self.tr] == user)).sum(), 1)

    def test_a_group_that_cannot_be_formed_raises_instead_of_shrinking(self):
        y = np.repeat(np.arange(12), 4)
        subj = np.ones(len(y), dtype=int)                      # a single user only
        with self.assertRaises(RuntimeError):
            sampling.relation_epochs(y, subj, np.arange(len(y)), 1, 0)

    def test_saved_order_is_reproducible(self):
        again = sampling.relation_epochs(self.y, self.subj, self.tr, 1, P.SAMPLER_SEED + 1)
        self.assertEqual(sampling.manifest_hash(self.groups), sampling.manifest_hash(again))


class RelationLossTests(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor([0, 0, 1, 1] * 4)
        self.subjects = torch.tensor([1, 2, 1, 2] * 4)
        torch.manual_seed(0)
        self.teacher = torch.randn(16, 300)

    def test_matched_representations_give_zero_relation_loss(self):
        rel, _ = REL.cross_user_relation_losses(self.teacher, self.teacher, self.labels, self.subjects)
        self.assertLess(abs(float(rel)), 1e-5)

    def test_same_user_teacher_neighbours_do_not_move_the_loss(self):
        student = torch.randn(16, 64)
        before, _ = REL.cross_user_relation_losses(student, self.teacher, self.labels, self.subjects)
        poisoned = self.teacher.clone()
        same_user = self.subjects == self.subjects[0]
        poisoned[same_user] = poisoned[same_user] + 0.0     # same-user rows stay neighbours of others
        after, _ = REL.cross_user_relation_losses(student, poisoned, self.labels, self.subjects)
        torch.testing.assert_close(before, after)

    def test_widths_may_differ(self):
        rel, con = REL.cross_user_relation_losses(torch.randn(16, 64), self.teacher,
                                                  self.labels, self.subjects)
        self.assertTrue(torch.isfinite(rel) and torch.isfinite(con))

    def test_single_user_batch_raises_instead_of_dividing_by_zero(self):
        with self.assertRaises(ValueError):
            REL.cross_user_relation_losses(torch.randn(8, 64), torch.randn(8, 300),
                                           torch.arange(8) % 2, torch.ones(8, dtype=torch.long))

    def test_teacher_stays_detached_and_student_receives_gradient(self):
        student = torch.randn(16, 64, requires_grad=True)
        teacher = self.teacher.clone().requires_grad_(True)
        rel, con = REL.cross_user_relation_losses(student, teacher, self.labels, self.subjects)
        (rel + con).backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0)

    def test_global_derangement_actually_replaces_every_target(self):
        """The microbatch-local variant left 68.4% of exposures on their own
        target, so it could not test instance correspondence. The global one
        must replace all of them while keeping class and user fixed."""
        y, subj = P.labels()
        tr, va, te = P.fold_split(1, y, subj)
        lookup, changed = REL.global_within_class_derangement(y, subj, tr, 1)
        self.assertEqual(changed, 1.0)
        np.testing.assert_array_equal(y[lookup[tr]], y[tr])
        np.testing.assert_array_equal(subj[lookup[tr]], subj[tr])
        self.assertFalse(bool((lookup[tr] == tr).any()))
        np.testing.assert_array_equal(lookup[te], te)

    def test_within_class_shuffle_keeps_class_and_user_statistics(self):
        ids = torch.arange(16)
        generator = torch.Generator().manual_seed(3)
        shuffled = REL.shuffle_within_class(ids, self.labels, self.subjects, generator)
        np.testing.assert_array_equal(self.labels[shuffled].numpy(), self.labels.numpy())
        np.testing.assert_array_equal(self.subjects[shuffled].numpy(), self.subjects.numpy())
        self.assertTrue(bool((shuffled != ids).any()))


@unittest.skipUnless(CUDA, 'needs the training GPU')
class TeacherSourceTests(unittest.TestCase):
    def test_the_three_relation_sources_are_distinguishable(self):
        y, subj = P.labels()
        tr, va, te = P.fold_split(1, y, subj)
        ledgers = {}
        for source in ('multi', 'target_single', 'teacher_single'):
            code, ledger = teacher_targets.relation_targets(1, source, tr, device='cpu')
            ledgers[source] = ledger
            self.assertAlmostEqual(float(code[tr].norm(dim=-1).mean()), 1.0, places=4)
        self.assertEqual(ledgers['multi']['teacher_checkpoint_sha256'],
                         ledgers['target_single']['teacher_checkpoint_sha256'])
        self.assertNotEqual(ledgers['multi']['teacher_target_views'],
                            ledgers['target_single']['teacher_target_views'])
        self.assertNotEqual(ledgers['teacher_single']['teacher_checkpoint_sha256'],
                            ledgers['multi']['teacher_checkpoint_sha256'])
        self.assertEqual(ledgers['teacher_single']['teacher_training_views'], ['Right'])


class ProcessTargetTests(unittest.TestCase):
    @staticmethod
    def _bursty(segment=2, views=3, batch=1):
        energy = torch.zeros(batch, views, P.FRAMES)
        energy[:, :, segment * P.FRAMES_PER_SEGMENT:(segment + 1) * P.FRAMES_PER_SEGMENT] = 1.0
        return energy, torch.ones_like(energy), torch.ones_like(energy, dtype=torch.bool)

    def test_a_single_active_segment_concentrates_the_target(self):
        activity, progress, seg_ok, valid = MT.fuse_process_targets(*self._bursty(2))
        self.assertEqual(int(activity.argmax(-1)), 2)
        self.assertGreater(float(activity[0, 2]), 0.99)
        self.assertTrue(bool(valid.all()) and bool(seg_ok.all()))
        self.assertTrue(bool((progress.diff(dim=-1) >= -1e-6).all()))
        self.assertAlmostEqual(float(progress[0, -1]), 1.0, places=5)

    def test_a_dead_clip_is_marked_invalid_and_contributes_zero(self):
        zero = torch.zeros(1, 3, P.FRAMES)
        activity, progress, seg_ok, valid = MT.fuse_process_targets(
            zero, torch.ones_like(zero), torch.ones_like(zero, dtype=torch.bool))
        self.assertFalse(bool(valid.any()))
        logits = torch.zeros(1, P.SEGMENTS, requires_grad=True)
        kl, cdf = process_loss(logits, activity, progress, seg_ok, valid)
        self.assertEqual(float(kl), 0.0)
        self.assertEqual(float(cdf), 0.0)
        (kl + cdf).backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_partly_invalid_segments_average_over_valid_frames_only(self):
        energy, quality, frame_valid = self._bursty(2)
        energy[:, :, 12] = 4.0                                   # a strong frame in segment 2
        frame_valid[:, :, 13:18] = False                         # the rest of segment 2 is padding
        activity, _, seg_ok, valid = MT.fuse_process_targets(energy, quality, frame_valid)
        self.assertTrue(bool(seg_ok.all()))
        self.assertGreater(float(activity[0, 2]), float(activity[0, 3]))

    def test_three_identical_views_fuse_to_the_single_view_answer(self):
        energy, quality, frame_valid = self._bursty(4, views=3)
        energy = energy * torch.tensor([1.0, 1.0, 1.0]).reshape(1, 3, 1)
        fused = MT.fuse_process_targets(energy, quality, frame_valid)[0]
        alone = MT.fuse_process_targets(energy[:, :1], quality[:, :1], frame_valid[:, :1])[0]
        torch.testing.assert_close(fused, alone)

    def test_opposite_phase_motion_does_not_cancel(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=(1, 40, 64))
        forward = np.cumsum(base, axis=1)
        e_forward = MT.frame_evidence(forward)[0]
        e_backward = MT.frame_evidence(forward[:, ::-1])[0]
        self.assertGreater(float(e_forward.sum()), 0)
        self.assertGreater(float(e_backward.sum()), 0)

    def test_time_shuffle_rebuilds_the_cdf_from_the_shuffled_activity(self):
        activity = torch.rand(3, P.SEGMENTS)
        activity = activity / activity.sum(-1, keepdim=True)
        generator = torch.Generator().manual_seed(5)
        shuffled, progress = MT.shuffle_time(activity, generator)
        torch.testing.assert_close(progress, shuffled.cumsum(-1))
        self.assertAlmostEqual(float(progress[0, -1]), 1.0, places=5)

    def test_scales_are_fitted_on_training_rows_only(self):
        energy, _ = MT.load_energy()
        y, subj = P.labels()
        tr, va, te = P.fold_split(1, y, subj)
        before = MT.fold_scale(energy, tr)
        poisoned = energy.copy()
        poisoned[te] *= 100.0
        poisoned[va] *= 100.0
        np.testing.assert_allclose(before, MT.fold_scale(poisoned, tr))


class PredictabilityTests(unittest.TestCase):
    """Method C selection, checked on synthetic data with a known answer."""

    @staticmethod
    def _population(n_users=7, per_user=120, classes=6, dim=20, seed=0):
        # Seven inner users, matching the real leave-one-source-user-out layout,
        # so the 5-of-7 eligibility rule is exercised rather than trivially failed.
        rng = np.random.default_rng(seed)
        n = n_users * per_user
        subj = np.repeat(np.arange(1, n_users + 1), per_user)
        y = rng.integers(0, classes, size=n)
        features = rng.normal(size=(n, dim))
        return features, y, subj, np.arange(n), rng

    def test_skill_separates_predictable_from_independent_coordinates(self):
        import predictability as C
        features, y, subj, ids, rng = self._population()
        weights = rng.normal(size=(features.shape[1], 5))
        target = np.concatenate([features @ weights, rng.normal(size=(len(features), 5))], -1)
        skills, users = C.inner_ridge_skill(features, target, ids, y, subj, residual=True)
        self.assertEqual(skills.shape, (7, 10))
        self.assertGreater(skills[:, :5].mean(), 0.8)
        self.assertLess(skills[:, 5:].mean(), 0.05)
        chosen, summary = C.select(skills)
        self.assertEqual(sorted(chosen.tolist()), [0, 1, 2, 3, 4])
        self.assertEqual(summary['eligible'], 5)

    def test_class_centre_only_target_reports_no_residual_detail(self):
        import predictability as C
        features, y, subj, ids, rng = self._population(seed=1)
        centres = rng.normal(size=(12, 8))
        target = centres[y] + 0.01 * rng.normal(size=(len(y), 8))
        skills, _ = C.inner_ridge_skill(features, target, ids, y, subj, residual=True)
        self.assertLess(skills.mean(), 0.05)
        chosen, summary = C.select(skills)
        self.assertTrue(summary['stop'], 'a class-centre-only target must not pass selection')

    def test_probe_user_never_enters_the_fit(self):
        import predictability as C
        features, y, subj, ids, rng = self._population(seed=2)
        target = rng.normal(size=(len(y), 6))
        reserved = ids[:10]
        with self.assertRaises(AssertionError):
            C.inner_ridge_skill(features, target, ids, y, subj, reserved=(reserved,))

    def test_missing_classes_do_not_shift_the_centre_index(self):
        import predictability as C
        target = np.arange(24, dtype='float64').reshape(12, 2)
        labels = np.array([0, 3, 11, 3])
        centres, present = C.class_centres(target[labels], labels)
        self.assertEqual(centres.shape, (12, 2))
        np.testing.assert_array_equal(present.nonzero()[0], [0, 3, 11])
        np.testing.assert_allclose(centres[11], target[11])
        np.testing.assert_allclose(centres[1], 0)

    def test_selection_stops_when_too_few_coordinates_qualify(self):
        import predictability as C
        skills = np.full((7, 100), -1.0)
        skills[:, :3] = 0.5
        chosen, summary = C.select(skills)
        self.assertEqual(summary['selected'], 3)
        self.assertTrue(summary['stop'])

    def test_radar_probe_features_are_420_fixed_pooled_coordinates(self):
        import predictability as C
        features = C.build_probe_features()
        self.assertEqual(features.shape[1], 420)
        self.assertTrue(np.isfinite(features).all())


class ResultAuditTests(unittest.TestCase):
    def test_reported_metrics_match_the_saved_predictions(self):
        import summarize
        rows = summarize.collect()
        if not rows:
            self.skipTest('no completed runs yet')
        for row in rows:
            self.assertAlmostEqual(row['accuracy'], row['recomputed_accuracy'], places=9)
            self.assertAlmostEqual(row['macro_f1'], row['recomputed_macro_f1'], places=9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
