"""Protocol and numerical checks for source-only, partial-action calibration."""

from importlib import import_module, util

import numpy as np
import pytest


def core():
    # Keep the initial red run an explicit missing-feature assertion.
    assert util.find_spec("experiments.new_routes.calibration") is not None, (
        "the statistical calibration core has not been implemented"
    )
    return import_module("experiments.new_routes.calibration")


def source_fixture():
    rng = np.random.default_rng(71)
    centers = rng.normal(size=(12, 4)) * 3
    loadings = rng.normal(size=(12, 4, 2)) * 0.25
    latent = np.array([[-2, -1], [-1, 2], [0, -2], [0, 2], [1, -2], [2, 1]])
    features, labels, subjects = [], [], []
    for user in range(6):
        for label in range(12):
            features.append(centers[label] + loadings[label] @ latent[user])
            labels.append(label)
            subjects.append(user + 1)
    return np.array(features), np.array(labels), np.array(subjects), centers, loadings


def fitted():
    x, y, users, _, _ = source_fixture()
    return core().fit_source(x, y, users, n_components=4, rank=2)


def test_translation_has_analytic_mean_shift_and_shrinkage():
    adapter = fitted()
    _, _, _, centers, _ = source_fixture()
    labels = np.array([0, 0, 1, 2])
    shift = np.array([1.5, -0.7, 0.2, 0.4])
    support = centers[labels] + shift
    expected_shift = adapter.transform(centers[[0]] + shift)[0] - adapter.mu[0]
    delta = adapter.estimate_delta(support, labels, method="translation", reg=2)
    np.testing.assert_allclose(delta, np.tile(expected_shift * (4 / 6), (12, 1)))


def test_supported_only_never_moves_unobserved_classes():
    adapter = fitted()
    _, _, _, centers, _ = source_fixture()
    labels = np.array([0, 0, 4])
    shifts = np.array([[0.8, 0.2, -0.5, 0.1], [-0.2, 0.4, 0.1, 0.3], [1, 0, 0, 0]])
    support = centers[labels] + shifts
    delta = adapter.estimate_delta(support, labels, method="supported_only", reg=1)
    np.testing.assert_array_equal(delta[[1, 2, 3, 5, 6, 7, 8, 9, 10, 11]], 0)
    residual = adapter.transform(support) - adapter.mu[labels]
    np.testing.assert_allclose(delta[0], residual[:2].sum(axis=0) / 3)
    np.testing.assert_allclose(delta[4], residual[2] / 2)


@pytest.mark.parametrize("method", ["supported_only", "translation", "ridge_affine", "lowrank"])
def test_no_support_produces_identity(method):
    adapter = fitted()
    delta = adapter.estimate_delta(np.empty((0, 4)), np.empty(0, dtype=int), method, 1)
    np.testing.assert_array_equal(delta, np.zeros((12, 4)))


def test_zero_strength_is_bitwise_frozen_classifier_identity():
    module = core()
    rng = np.random.default_rng(6)
    base = rng.normal(size=(7, 12)).astype(np.float32)
    original = base.copy()
    z, mu, delta = rng.normal(size=(7, 3)), rng.normal(size=(12, 3)), rng.normal(size=(12, 3))
    output = module.correction_scores(base, z, mu, delta, tau=0)
    assert output.dtype == base.dtype
    np.testing.assert_array_equal(output, original)
    np.testing.assert_array_equal(base, original)
    expected = base + 0.5 * (((z[:, None, :] - mu) * delta).sum(-1) - 0.5 * (delta**2).sum(-1))
    np.testing.assert_allclose(module.correction_scores(base, z, mu, delta, tau=0.5), expected)


def test_source_prototype_scores_equal_squared_distance():
    module = core()
    z = np.array([[1.0, 2.0], [-1.0, 0.0]])
    mu = np.array([[0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(module.prototype_scores(z, mu), [[-2.5, -2.0], [-0.5, -2.0]])


def test_ridge_affine_uses_sample_weights_and_regularizes_intercept():
    adapter = fitted()
    _, _, _, centers, _ = source_fixture()
    labels = np.array([0, 0, 0, 1, 2, 2, 7])
    support = centers[labels] + np.arange(28).reshape(7, 4) * 0.01
    design = np.column_stack((adapter.mu[labels], np.ones(len(labels))))
    residual = adapter.transform(support) - adapter.mu[labels]
    coef = np.linalg.solve(design.T @ design + 2 * np.eye(5), design.T @ residual)
    expected = np.column_stack((adapter.mu, np.ones(12))) @ coef
    np.testing.assert_allclose(adapter.estimate_delta(support, labels, "ridge_affine", 2), expected)


def test_known_lowrank_user_shift_transfers_to_unsupported_classes():
    x, y, users, centers, loadings = source_fixture()
    adapter = core().fit_source(x, y, users, n_components=4, rank=2)
    target_latent = np.array([0.7, -0.3])
    target_centers = centers + np.einsum("cdr,r->cd", loadings, target_latent)
    selected = np.array([0, 1, 2])
    delta = adapter.estimate_delta(target_centers[selected], selected, "lowrank", 0)
    expected = adapter.transform(target_centers) - adapter.mu
    np.testing.assert_allclose(delta[3:], expected[3:], atol=2e-12)
    assert np.linalg.norm(delta[3:]) > 0.1


def test_whitening_and_loading_scale_match_frozen_source_statistics():
    x, y, users, _, _ = source_fixture()
    adapter = core().fit_source(x, y, users, n_components=4, rank=2)
    projected = (x - adapter.feature_mean) @ adapter.pca_components.T
    centers = np.stack([projected[y == label].mean(axis=0) for label in range(12)])
    expected_variance = np.maximum(((projected - centers[y]) ** 2).mean(axis=0), 1e-6)
    np.testing.assert_allclose(adapter.variance, expected_variance)
    np.testing.assert_allclose(adapter.transform(x), projected / np.sqrt(expected_variance))
    residuals = (adapter.user_class_means - adapter.mu).reshape(6, -1)
    loadings = adapter.B.reshape(-1, 2)
    np.testing.assert_allclose(loadings @ loadings.T, residuals.T @ residuals / 6, atol=2e-14)
    assert adapter.metadata["variance_includes_between_user_residual"] is True
    assert adapter.metadata["loading_scale"] == "singular_values/sqrt(n_source_users)"


def test_zero_within_class_variance_uses_finite_absolute_floor():
    _, _, _, centers, _ = source_fixture()
    x, y, users = np.tile(centers, (6, 1)), np.tile(np.arange(12), 6), np.repeat(np.arange(1, 7), 12)
    adapter = core().fit_source(x, y, users, n_components=4, rank=2)
    np.testing.assert_array_equal(adapter.variance, np.full(4, 1e-6))
    assert np.isfinite(adapter.transform(x)).all()
    assert np.isfinite(adapter.B).all()


def test_masked_imputation_ignores_missing_placeholders():
    module = core()
    users = np.array([-3.0, -1.0, 1.0, 3.0])
    class_vector = np.array([[1.0, 2.0], [3.0, -2.0], [2.0, 1.0]])
    residuals = users[:, None, None] * class_vector[None, :, :]
    observed = np.ones((4, 3), dtype=bool)
    observed[[1, 2], 1] = False  # symmetric missingness preserves class centering
    nan_input, large_input = residuals.copy(), residuals.copy()
    nan_input[~observed], large_input[~observed] = np.nan, 1e25
    a = module.masked_low_rank(nan_input, observed, rank=1)
    b = module.masked_low_rank(large_input, observed, rank=1)
    np.testing.assert_array_equal(a.completed[observed], residuals[observed])
    np.testing.assert_allclose(a.completed, b.completed, atol=0, rtol=0)
    np.testing.assert_allclose(a.reconstructed[~observed], residuals[~observed], atol=2e-5)
    assert a.diagnostics["iterations"] <= 100
    assert a.diagnostics["converged"]
    assert np.isfinite(a.diagnostics["relative_change"])
    assert a.diagnostics["missing_cells"] == 2


def test_imputation_honestly_reports_nonconvergence_at_frozen_cap():
    users = np.array([-3.0, -1.0, 1.0, 3.0])
    class_vector = np.array([[1.0, 2.0], [3.0, -2.0], [2.0, 1.0]])
    residuals = users[:, None, None] * class_vector[None, :, :]
    observed = np.ones((4, 3), dtype=bool)
    observed[[0, 3], 1] = False
    result = core().masked_low_rank(residuals, observed, rank=1)
    # Low-leverage observed cells make this case slow. A hard cap must not
    # silently become a claim that the matrix completion converged.
    assert result.diagnostics["iterations"] == 100
    assert not result.diagnostics["converged"]
    assert result.diagnostics["relative_change"] > result.diagnostics["tol"]
    np.testing.assert_array_equal(result.completed[observed], residuals[observed])
    with pytest.raises(ValueError, match="100"):
        core().masked_low_rank(residuals, observed, rank=1, max_iter=101)


def test_balanced_centers_exclude_missing_users_and_preserve_source_trace():
    x, y, users, _, _ = source_fixture()
    keep = ~(((users == 2) & np.isin(y, [10, 11])) | ((users == 6) & (y == 3)))
    x, y, users = x[keep], y[keep], users[keep]
    # Unequal recordings must not give one user extra weight in the class center.
    repeat = np.flatnonzero((users == 1) & (y == 10))
    x = np.concatenate([x, np.repeat(x[repeat], 11, axis=0)])
    y = np.concatenate([y, np.repeat(y[repeat], 11)])
    users = np.concatenate([users, np.repeat(users[repeat], 11)])
    ids = np.array([f"record-{index}" for index in range(len(y))])
    adapter = core().fit_source(x, y, users, n_components=4, rank=2, source_ids=ids)
    z = adapter.transform(x)
    expected = np.mean([z[(users == user) & (y == 10)].mean(axis=0) for user in [1, 3, 4, 5, 6]], axis=0)
    np.testing.assert_allclose(adapter.mu[10], expected)
    assert adapter.source_user_ids.tolist() == [1, 2, 3, 4, 5, 6]
    assert adapter.source_user_count == 6
    assert adapter.source_sample_count == len(y)
    np.testing.assert_array_equal(adapter.source_ids, ids)
    assert not adapter.observed_mask[1, 10]
    assert not adapter.observed_mask[1, 11]
    assert not adapter.observed_mask[5, 3]
    assert adapter.cell_counts[0, 10] == 12


def test_d_optimal_selection_uses_fixed_candidates_and_lexicographic_ties():
    module = core()
    zero = np.zeros((12, 3, 2))
    assert module.select_classes(zero) == (0, 1, 2)
    loadings = zero.copy()
    loadings[4, 0, 0] = 6
    loadings[7, 0, 1] = 5
    loadings[9, 1, :] = [1, 1]
    loadings[3] = 1000  # excluded candidate, despite much larger information
    assert module.select_classes(loadings) == (4, 7, 9)
    assert fitted().select_classes() == module.select_classes(fitted().B)


def partition_fixture():
    labels = np.repeat(np.arange(12), 27)
    users = np.repeat([2, 6], len(labels))
    labels = np.tile(labels, 2)
    keep = ~(((users == 2) & np.isin(labels, [10, 11])) | ((users == 6) & (labels == 3)))
    return np.array([f"HV-{i:04d}" for i in range(keep.sum())]), labels[keep], users[keep]


def test_fixed_partition_and_support_are_exact_disjoint_and_deterministic():
    module = core()
    ids, labels, users = partition_fixture()
    parts = module.fixed_partition(ids, labels, users)
    duplicate = module.fixed_partition(ids, labels, users)
    candidate_ids = set(parts.support_ids.tolist())
    assert not candidate_ids.intersection(parts.query_ids.tolist())
    assert candidate_ids.union(parts.query_ids.tolist()) == set(ids.tolist())
    np.testing.assert_array_equal(parts.support_indices, duplicate.support_indices)
    np.testing.assert_array_equal(parts.query_indices, duplicate.query_indices)
    for user in [2, 6]:
        for label in np.unique(labels[users == user]):
            pool = np.flatnonzero((users == user) & (labels == label))
            expected = np.random.default_rng(91000 + user * 100 + label).permutation(pool)
            actual = parts.support_indices[(parts.support_subjects == user) & (parts.support_labels == label)]
            np.testing.assert_array_equal(actual, expected[:20])
        for seed in range(92000, 92020):
            chosen = module.sample_support(parts, user, (0, 1, 2), seed)
            same = module.sample_support(parts, user, (0, 1, 2), seed)
            np.testing.assert_array_equal(chosen.indices, same.indices)
            rng = np.random.default_rng(seed)
            expected_positions = np.concatenate([
                rng.choice(np.flatnonzero((parts.support_subjects == user) & (parts.support_labels == label)), 3, replace=False)
                for label in [0, 1, 2]
            ])
            np.testing.assert_array_equal(chosen.indices, parts.support_indices[expected_positions])
            assert len(chosen.ids) == 9
            assert len(set(chosen.ids.tolist())) == 9
            assert set(chosen.ids.tolist()) <= candidate_ids
            assert not set(chosen.ids.tolist()).intersection(parts.query_ids.tolist())
            np.testing.assert_array_equal(chosen.ids, ids[chosen.indices])
            np.testing.assert_array_equal(np.bincount(chosen.labels, minlength=12)[:3], [3, 3, 3])


def test_all_available_class_reference_preserves_missing_class_budgets():
    module = core()
    ids, labels, users = partition_fixture()
    for user, budget in [(2, 30), (6, 33)]:
        mask = users == user
        partition = module.fixed_partition(ids[mask], labels[mask], users[mask])
        available = np.unique(labels[mask])
        sample = module.sample_support(partition, user, available, 92000)
        assert len(sample.ids) == budget
        np.testing.assert_array_equal(sample.ids, ids[mask][sample.indices])
        absent = next(label for label in range(12) if label not in available)
        with pytest.raises(ValueError, match="fewer than three"):
            module.sample_support(partition, user, [absent], 92000)


def test_duplicate_ids_and_short_support_pools_are_rejected():
    module = core()
    ids, labels, users = partition_fixture()
    ids[1] = ids[0]
    with pytest.raises(ValueError, match="unique"):
        module.fixed_partition(ids, labels, users)
    with pytest.raises(ValueError, match="20"):
        module.fixed_partition(np.arange(19), np.zeros(19, dtype=int), np.ones(19, dtype=int))


def test_fitting_and_inference_do_not_mutate_caller_arrays():
    module = core()
    x, y, users, _, _ = source_fixture()
    ids = np.arange(len(y))
    originals = [arr.copy() for arr in (x, y, users, ids)]
    adapter = module.fit_source(x, y, users, n_components=4, rank=2, source_ids=ids)
    adapter.estimate_delta(x[:9], y[:9], "lowrank", 1)
    for arr, original in zip((x, y, users, ids), originals):
        np.testing.assert_array_equal(arr, original)
    ids[:] = -1
    np.testing.assert_array_equal(adapter.source_ids, originals[-1])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_source_and_support_are_rejected(bad):
    module = core()
    x, y, users, _, _ = source_fixture()
    x[0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        module.fit_source(x, y, users, n_components=4, rank=2)
    with pytest.raises(ValueError, match="finite"):
        fitted().estimate_delta(x[:3], y[:3], "translation", 1)


def test_labels_missing_source_classes_and_negative_regularization_are_rejected():
    module = core()
    x, y, users, _, _ = source_fixture()
    with pytest.raises(ValueError, match="12|0.*11|label"):
        module.fit_source(x, np.where(y == 0, 12, y), users, n_components=4, rank=2)
    keep = y != 11
    with pytest.raises(ValueError, match="missing.*11|class.*11"):
        module.fit_source(x[keep], y[keep], users[keep], n_components=4, rank=2)
    with pytest.raises(ValueError, match="reg"):
        fitted().estimate_delta(x[:3], y[:3], "translation", -1)


def test_fit_signatures_have_no_query_or_hidden_target_inputs():
    import inspect

    module = core()
    source_parameters = inspect.signature(module.fit_source).parameters
    support_parameters = inspect.signature(module.SourceCalibration.estimate_delta).parameters
    assert not any("query" in name or "target" in name for name in source_parameters)
    assert list(support_parameters) == ["self", "support_x", "support_y", "method", "reg"]
