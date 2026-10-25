import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import diagnostic


class GuardedLabels:
    """Synthetic label store that fails if sealed participant rows are read."""

    def __init__(self, values, forbidden):
        self.values = np.asarray(values)
        self.forbidden = set(forbidden)
        self.read_rows = []

    def __getitem__(self, item):
        rows = np.asarray(item, dtype=np.int64)
        if self.forbidden.intersection(rows.tolist()):
            raise AssertionError("sealed HV1/HV2 sentinel label row was accessed")
        self.read_rows.extend(rows.tolist())
        return self.values[item]


class GuardedMaps:
    def __init__(self, forbidden):
        self.forbidden = set(forbidden)
        self.read_rows = []

    def __getitem__(self, row):
        row = int(row)
        if row in self.forbidden:
            raise AssertionError("sealed HV1/HV2 sentinel map row was accessed")
        self.read_rows.append(row)
        return np.full((3, 90, 189), row, dtype=np.float32)


class BudgetAndProtocolTests(unittest.TestCase):
    def test_outer_user_sentinel_labels_are_never_read(self):
        subjects = np.array([1, 2, 3, 3, 7, 8], dtype=np.int8)
        rows = diagnostic.select_source_rows(subjects)
        labels = GuardedLabels([91, 92, 3, 4, 7, 8], forbidden={0, 1})
        selected = diagnostic.read_selected_source_labels(labels, rows)
        self.assertEqual(selected.tolist(), [3, 4, 7, 8])
        self.assertEqual(labels.read_rows, [2, 3, 4, 5])
        self.assertTrue(np.isin(subjects[rows], diagnostic.SOURCE_USERS).all())

    def test_outer_user_sentinel_maps_are_never_read(self):
        subjects = np.array([1, 2, 3, 3, 7, 8], dtype=np.int8)
        rows = diagnostic.select_source_rows(subjects)
        maps = GuardedMaps(forbidden={0, 1})
        features = list(diagnostic.iter_source_features(maps, rows))
        self.assertEqual(len(features), 4)
        self.assertEqual(maps.read_rows, [2, 3, 4, 5])

    def test_audit_users_are_never_in_group_fit_rows(self):
        rng = np.random.default_rng(41)
        users = np.array([3] * 10 + [4] * 10 + [5] * 10 + [6] * 10 + [7] * 10 + [8] * 10)
        y = np.zeros(len(users), dtype=np.int8)
        features = rng.normal(size=(len(users), 3, 210)).astype(np.float32)
        latent = np.repeat(np.array([-1.0, 1.0]), 30)
        features[:, 0, 0] += latent
        features[:, 1, 1] += latent
        fit, audit = diagnostic.fit_audit_masks(users)
        _gf, _ga, models = diagnostic.classwise_groups(features, y, users, fit, audit, "lt")
        model = models[0]
        self.assertEqual(set(users[model["fit_indices"]]), {3, 4, 5, 6})
        self.assertEqual(model["fit_users"].tolist(), [3, 4, 5, 6])

    def test_audit_row_permutation_does_not_change_frozen_group_identity(self):
        rng = np.random.default_rng(42)
        users = np.array([3] * 12 + [4] * 12 + [5] * 12 + [6] * 12 + [7] * 12 + [8] * 12)
        y = np.zeros(len(users), dtype=np.int8)
        features = rng.normal(size=(len(users), 3, 210)).astype(np.float32)
        features[:, 0, 3] += np.repeat(np.array([-1.2, 1.2]), len(users) // 2)
        features[:, 1, 7] += np.repeat(np.array([-1.2, 1.2]), len(users) // 2)
        fit, audit = diagnostic.fit_audit_masks(users)
        _, group_a, models_a = diagnostic.classwise_groups(features, y, users, fit, audit, "lt")

        audit_indices = np.flatnonzero(audit)
        perm = rng.permutation(audit_indices)
        order = np.concatenate([np.flatnonzero(fit), perm])
        perm_features, perm_y, perm_users = features[order], y[order], users[order]
        perm_fit, perm_audit = diagnostic.fit_audit_masks(perm_users)
        _, group_b, models_b = diagnostic.classwise_groups(
            perm_features, perm_y, perm_users, perm_fit, perm_audit, "lt")
        self.assertTrue(np.array_equal(models_a[0]["kmeans"].cluster_centers_,
                                       models_b[0]["kmeans"].cluster_centers_))
        original_by_row = {int(idx): int(group_a[idx]) for idx in audit_indices}
        permuted_by_row = {int(order[pos]): int(group_b[pos]) for pos in np.flatnonzero(perm_audit)}
        self.assertEqual(original_by_row, permuted_by_row)

    def test_audit_labels_do_not_change_fitted_group_models(self):
        rng = np.random.default_rng(43)
        users = np.array([3] * 12 + [4] * 12 + [5] * 12 + [6] * 12 + [7] * 12 + [8] * 12)
        y = np.zeros(len(users), dtype=np.int8)
        features = rng.normal(size=(len(users), 3, 210)).astype(np.float32)
        fit, audit = diagnostic.fit_audit_masks(users)
        _, _, model_a = diagnostic.classwise_groups(features, y, users, fit, audit, "lt")
        y_changed = y.copy()
        y_changed[np.flatnonzero(audit)[0]] = 11
        _, _, model_b = diagnostic.classwise_groups(features, y_changed, users, fit, audit, "lt")
        self.assertEqual(model_a[0]["fit_indices"].tolist(), model_b[0]["fit_indices"].tolist())
        self.assertTrue(np.array_equal(model_a[0]["kmeans"].cluster_centers_,
                                       model_b[0]["kmeans"].cluster_centers_))

    def test_cmi_independence_and_perfect_dependency_toys(self):
        # Exact balanced 2x2 table has zero empirical mutual information.
        a = np.tile([0, 0, 1, 1], 16)
        b = np.tile([0, 1, 0, 1], 16)
        strata = [np.zeros(len(a), dtype=np.int8)]
        self.assertAlmostEqual(diagnostic.cmi_bits(a, b, strata), 0.0, places=12)
        self.assertAlmostEqual(diagnostic.cmi_bits(a, a, strata), 1.0, places=12)

    def test_environment_manifest_records_one_thread_pool(self):
        with diagnostic.threadpool_limits(limits=1):
            manifest = diagnostic.environment_manifest()
        self.assertTrue(manifest["thread_limit_verified"])
        self.assertTrue(all(pool["num_threads"] == 1 for pool in manifest["threadpools"]))

    def test_permutation_preserves_counts_in_every_frozen_stratum(self):
        x = np.array([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0], dtype=np.int8)
        y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], dtype=np.int8)
        u = np.repeat([7, 8], 6)
        r = np.tile([0, 0, 0, 1, 1, 1], 2)
        shuffled = diagnostic.permute_within_strata(x, [y, u, r], np.random.default_rng(7))
        before = Counter(zip(y.tolist(), u.tolist(), r.tolist(), x.tolist()))
        after = Counter(zip(y.tolist(), u.tolist(), r.tolist(), shuffled.tolist()))
        self.assertEqual(before, after)

    def test_predata_failure_retry_reuses_descriptive_protocol(self):
        protocol = json.loads((Path(diagnostic.__file__).with_name("protocol.json")).read_text())
        self.assertTrue(protocol["source_exchangeability"]["run_authorized"])
        self.assertIn("retry_authorized", protocol["one_shot_execution"]["state"])
        self.assertTrue(protocol["one_shot_execution"]["prior_attempt_read_arrays"] is False)
        self.assertNotIn("permutation_gate", protocol["fixed_decision_rule"])
        self.assertIn("No record-level permutation p-value", protocol["fixed_decision_rule"]["inferential_boundary"])
        self.assertIn("not a scientific GO", protocol["fixed_decision_rule"]["pass_action"])
        self.assertEqual(diagnostic.N_PERMUTATIONS, 1999)

    def test_exact_source_alias_rule_blocks_conflicting_labels_before_fit(self):
        maps = np.zeros((6, 3, 90, 189), dtype=np.float32)
        maps[2] = 1.25
        maps[3] = maps[2]
        audit = diagnostic.audit_exact_triplet_aliases(
            maps, np.array([2, 3]), np.array([10, 11]), np.array([3, 4]))
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["exact_alias_pair_count"], 1)
        pair = audit["duplicate_pairs"][0]
        self.assertTrue(pair["byte_identical_triplet"])
        self.assertFalse(pair["same_class"])
        self.assertEqual((pair["first_class"], pair["duplicate_class"]), (10, 11))

    def test_exact_source_alias_scan_reads_only_selected_source_rows(self):
        maps = GuardedMaps(forbidden={0, 1})
        audit = diagnostic.audit_exact_triplet_aliases(
            maps, np.array([2, 3, 4]), np.array([0, 1, 2]), np.array([3, 4, 5]))
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["rows_checked"], 3)
        self.assertEqual(maps.read_rows, [2, 3, 4])

    def test_restore_provenance_checks_receipt_and_sizes_without_opening_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "cache"
            cache.mkdir()
            entries = []
            for rel, payload in [
                ("cache/X_clutterremoved.npy", b"sealed map bytes"),
                ("cache/y_clutterremoved.npy", b"label bytes"),
                ("cache/subj_clutterremoved.npy", b"subject bytes"),
            ]:
                p = root / rel
                p.write_bytes(payload)
                entries.append({"path": rel, "bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest()})
            manifest = root / "TRANSFER_MANIFEST.json"
            manifest.write_text(json.dumps({"state": "complete", "files": entries}))
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            (root / "RESTORE_VERIFIED.json").write_text(json.dumps({
                "state": "verified", "transfer_manifest_sha256": manifest_sha}))
            proof = diagnostic.restore_provenance(cache, manifest)
            self.assertTrue(proof["restore_receipt_manifest_sha256_matches"])
            self.assertEqual(proof["restored_input_files"]["cache/X_clutterremoved.npy"]["bytes"],
                             len(b"sealed map bytes"))
            self.assertIn("selected source rows", proof["restored_input_files"]["cache/X_clutterremoved.npy"]["content_hash_scope"])

    def _synthetic_command(self, code):
        return [sys.executable, "-c", code]

    def test_wall_timeout_is_inconclusive_and_preserves_partial_file(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "wall"
            code = (
                "import json,os,time; from pathlib import Path; "
                "p=Path(os.environ['UWB_STAGE_A_OUTPUT_DIR']); "
                "(p/'partial.bin').write_bytes(b'checkpoint'); "
                "(p/'run_status.json').write_text(json.dumps({'state':'running','phase':'synthetic_compute','rows_completed':3,'rows_total':9})); "
                "time.sleep(3)"
            )
            status = diagnostic.run_supervised(self._synthetic_command(code), out,
                                               cpu_seconds=3, wall_seconds=0.3, poll_seconds=0.02)
            self.assertEqual(status["state"], "aborted_wall_timeout")
            self.assertFalse(status["is_scientific_negative"])
            self.assertTrue((out / "partial.bin").is_file())
            self.assertIn("partial.bin", status["partial_artifacts"])
            self.assertEqual(status["last_worker_phase"], "synthetic_compute")
            self.assertEqual(status["last_progress"]["rows_completed"], 3)
            self.assertFalse((out / "diagnostic.json").exists())

    def test_cpu_limit_is_kernel_enforced_and_preserves_partial_file(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "cpu"
            code = "import os; from pathlib import Path; Path(os.environ['UWB_STAGE_A_OUTPUT_DIR'],'partial.bin').write_bytes(b'checkpoint'); x=0\nwhile True: x+=1"
            status = diagnostic.run_supervised(self._synthetic_command(code), out,
                                               cpu_seconds=1, wall_seconds=8, poll_seconds=0.02)
            self.assertEqual(status["state"], "aborted_cpu_limit")
            self.assertFalse(status["is_scientific_negative"])
            self.assertTrue((out / "partial.bin").is_file())
            self.assertIn("partial.bin", status["partial_artifacts"])
            self.assertGreaterEqual(status["child_cpu_seconds"], 0.8)
            self.assertFalse((out / "diagnostic.json").exists())

    def test_normal_synthetic_child_completes(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "normal"
            code = (
                "import os,sys; from pathlib import Path; "
                f"sys.path.insert(0,{str(Path(__file__).resolve().parents[1])!r}); "
                "import diagnostic,json; p=Path(os.environ['UWB_STAGE_A_OUTPUT_DIR']); "
                "(p/'synthetic.txt').write_text('ok'); "
                "(p/'thread_env.json').write_text(json.dumps({k:os.environ.get(k) for k in "
                "['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']})); "
                "(p/'threadpools.json').write_text(json.dumps(diagnostic.threadpool_info())); "
                "diagnostic.write_run_status(p,'completed','synthetic',extra={'scientific_result_recorded':False})"
            )
            status = diagnostic.run_supervised(self._synthetic_command(code), out,
                                               cpu_seconds=3, wall_seconds=5, poll_seconds=0.02)
            self.assertEqual(status["state"], "completed")
            self.assertTrue((out / "synthetic.txt").is_file())
            threads = json.loads((out / "thread_env.json").read_text())
            self.assertTrue(all(v == "1" for v in threads.values()))
            pools = json.loads((out / "threadpools.json").read_text())
            self.assertTrue(all(int(pool["num_threads"]) == 1 for pool in pools))


if __name__ == "__main__":
    unittest.main()
