"""Source-only partial-action user calibration, frozen protocol 2026-09-28.

``fit_source`` receives only explicit source arrays. It fits PCA (32 components
by default), balances available source users when forming each class center,
and whitens by the pooled source mean squared residual about those centers.
That diagonal in-class variance includes between-user variation; it is not an
estimate of physical measurement noise. Its absolute floor is 1e-6.

Missing user/class means are masked during iterative rank-2 SVD imputation.
Loadings are ``V_r * singular_values / sqrt(number_of_source_users)``. Thus
latent unit scale represents empirical source-user variation. Both conventions
were fixed before observing experiment outcomes and are retained in metadata.

Shapes: raw features [N,D] (normally D=256); transformed features [N,K]; mu
[12,K]; B [12,K,R]; support-derived delta [12,K]. No query arrays or labels
are accepted by either source fitting or support adaptation. Partition row
indices are local to the arrays passed to ``fixed_partition``; IDs retain their
original global identity and type. Input row order must remain fixed when a
saved partition is reused.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, Optional, Tuple

import numpy as np


N_CLASSES = 12
CANDIDATE_CLASSES = (0, 1, 2, 4, 5, 6, 7, 8, 9)
SUPPORT_SEEDS = tuple(range(92000, 92020))
METHODS = ("supported_only", "translation", "ridge_affine", "lowrank")


def _matrix(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape [N,D] with D > 0")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _integers(value, name, length=None):
    array = np.asarray(value)
    if array.ndim != 1 or (length is not None and len(array) != length):
        raise ValueError(f"{name} must be a vector matching the number of rows")
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite integer values")
    if not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} must contain integer values")
    return array.astype(np.int64, copy=True)


def _labels(value, length=None):
    labels = _integers(value, "labels", length)
    if ((labels < 0) | (labels >= N_CLASSES)).any():
        raise ValueError("labels must be in 0..11 (12 known output classes)")
    return labels


def _subjects(value, length=None):
    subjects = _integers(value, "subjects", length)
    if (subjects < 0).any():
        raise ValueError("subjects must have nonnegative integer IDs")
    return subjects


def _ids(value, length):
    ids = np.asarray(value)
    if ids.ndim != 1 or len(ids) != length:
        raise ValueError("IDs must be a vector matching the number of rows")
    if ids.dtype.kind in "iuf" and not np.isfinite(ids).all():
        raise ValueError("IDs must be finite")
    try:
        unique = len(set(ids.tolist()))
    except TypeError as error:
        raise ValueError("IDs must be hashable scalar values") from error
    if unique != len(ids):
        raise ValueError("IDs must be unique")
    return ids.copy()


def _positive_integer(value, name, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return int(value)


def _nonnegative(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _canonical_svd(matrix):
    """Fix otherwise arbitrary signs by each right vector's largest loading."""
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    signs = np.sign(vt[np.arange(len(vt)), np.argmax(np.abs(vt), axis=1)])
    signs[signs == 0] = 1
    return u * signs, singular_values, vt * signs[:, None]


@dataclass
class MaskedLowRankResult:
    completed: np.ndarray
    reconstructed: np.ndarray
    singular_values: np.ndarray
    right_vectors: np.ndarray
    diagnostics: Dict[str, Any]


def masked_low_rank(residuals, observed, rank=2, max_iter=100, tol=1e-8):
    """Impute only missing [user,class] cells, leaving observed values fixed.

    ``residuals`` is [U,C,K], ``observed`` is boolean [U,C]. Values behind a
    false mask are ignored completely, including NaN/large placeholders.
    Initialization uses observed column means; each iteration replaces missing
    values by a rank-R reconstruction. The relative change denominator is
    max(1, previous missing-value norm). At most 100 iterations are permitted.
    """
    values = np.asarray(residuals, dtype=np.float64)
    mask = np.asarray(observed)
    if values.ndim != 3 or min(values.shape) == 0:
        raise ValueError("residuals must have nonempty shape [U,C,K]")
    if mask.dtype.kind != "b" or mask.shape != values.shape[:2]:
        raise ValueError("observed must be a boolean [U,C] mask")
    if not np.isfinite(values[mask]).all():
        raise ValueError("observed residuals must be finite")
    if (~mask.any(axis=0)).any():
        raise ValueError("every class needs at least one observed source user")
    rank = _positive_integer(rank, "rank", min(values.shape[0], values.shape[1] * values.shape[2]))
    max_iter = _positive_integer(max_iter, "max_iter", 100)
    tol = _nonnegative(tol, "tol")
    if tol == 0:
        raise ValueError("tol must be positive")
    flat = values.reshape(values.shape[0], -1)
    flat_mask = np.repeat(mask, values.shape[2], axis=1)
    observed_values = np.where(flat_mask, flat, 0.0)
    column_mean = observed_values.sum(axis=0) / flat_mask.sum(axis=0)
    filled = np.where(flat_mask, observed_values, column_mean)
    missing = ~flat_mask
    converged, relative_change = False, 0.0
    for iteration in range(1, max_iter + 1):
        u, singular_values, vt = _canonical_svd(filled)
        reconstructed = (u[:, :rank] * singular_values[:rank]) @ vt[:rank]
        previous = filled[missing].copy()
        replacement = reconstructed[missing]
        relative_change = float(np.linalg.norm(replacement - previous) / max(1.0, np.linalg.norm(previous)))
        filled[missing] = replacement
        if relative_change <= tol:
            converged = True
            break
    # Refit to the final imputed matrix so factors and diagnostics agree.
    u, singular_values, vt = _canonical_svd(filled)
    reconstructed = (u[:, :rank] * singular_values[:rank]) @ vt[:rank]
    observed_rmse = float(np.sqrt(np.mean((reconstructed[flat_mask] - flat[flat_mask]) ** 2)))
    diagnostics = {
        "iterations": iteration,
        "max_iter": max_iter,
        "tol": tol,
        "converged": converged,
        "relative_change": relative_change,
        "observed_rmse": observed_rmse,
        "missing_cells": int((~mask).sum()),
        "observed_cells": int(mask.sum()),
        "rank": rank,
    }
    return MaskedLowRankResult(
        completed=filled.reshape(values.shape),
        reconstructed=reconstructed.reshape(values.shape),
        singular_values=singular_values[:rank].copy(),
        right_vectors=vt[:rank].copy(),
        diagnostics=diagnostics,
    )


def _ridge(design, response, reg):
    # At zero regularization, an underdetermined support set has a deterministic
    # minimum-norm solution rather than a singular normal equation.
    if reg == 0:
        return np.linalg.lstsq(design, response, rcond=None)[0]
    return np.linalg.solve(design.T @ design + reg * np.eye(design.shape[1]), design.T @ response)


@dataclass
class SourceCalibration:
    feature_mean: np.ndarray
    pca_components: np.ndarray
    variance: np.ndarray
    scale: np.ndarray
    mu: np.ndarray
    B: np.ndarray
    source_user_ids: np.ndarray
    source_user_count: int
    source_sample_count: int
    source_ids: Optional[np.ndarray]
    observed_mask: np.ndarray
    cell_counts: np.ndarray
    user_class_means: np.ndarray
    diagnostics: Dict[str, Any]
    metadata: Dict[str, Any]

    def transform(self, features):
        """Apply the source-only PCA/diagonal whitening to raw [N,D] features."""
        features = _matrix(features, "features")
        if features.shape[1] != len(self.feature_mean):
            raise ValueError("features have a different raw dimension from source fitting")
        return ((features - self.feature_mean) @ self.pca_components.T) / self.scale

    def select_classes(self):
        """Return the source-only D-optimal triple from the nine fixed candidates."""
        return select_classes(self.B)

    def estimate_delta(self, support_x, support_y, method, reg):
        """Estimate class shifts [12,K] using only raw support [S,D] and [S].

        Supported-only/translation use n/(n+reg) shrinkage. Affine regression
        repeats each source class center per support example, which is exactly
        class-mean regression weighted by sample counts; its constant is also
        regularized. Low-rank least squares uses every support residual.
        """
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        reg = _nonnegative(reg, "reg")
        z = self.transform(support_x)
        labels = _labels(support_y, len(z))
        delta = np.zeros_like(self.mu)
        if len(z) == 0:
            return delta
        residual = z - self.mu[labels]
        if method == "supported_only":
            for label in np.unique(labels):
                selected = labels == label
                delta[label] = residual[selected].sum(axis=0) / (selected.sum() + reg)
        elif method == "translation":
            delta[:] = residual.sum(axis=0) / (len(z) + reg)
        elif method == "ridge_affine":
            design = np.column_stack((self.mu[labels], np.ones(len(z))))
            coefficients = _ridge(design, residual, reg)
            delta = np.column_stack((self.mu, np.ones(N_CLASSES))) @ coefficients
        else:
            design = self.B[labels].reshape(-1, self.B.shape[-1])
            latent = _ridge(design, residual.reshape(-1), reg)
            delta = np.einsum("ckr,r->ck", self.B, latent)
        return delta


def fit_source(features, labels, subjects, *, n_components=32, rank=2,
               source_ids=None, variance_floor=1e-6, max_iter=100, tol=1e-8):
    """Fit from explicitly supplied source rows; never discover data globally.

    The experiment caller must supply exactly its six training users. The core
    permits other positive source-user counts for numerical tests, and records
    their IDs/count for independent enforcement and audits.
    """
    features = _matrix(features, "source features")
    labels = _labels(labels, len(features))
    subjects = _subjects(subjects, len(features))
    source_ids = None if source_ids is None else _ids(source_ids, len(features))
    if len(features) == 0:
        raise ValueError("source features must not be empty")
    n_components = _positive_integer(n_components, "n_components", min(len(features), features.shape[1]))
    variance_floor = _nonnegative(variance_floor, "variance_floor")
    if variance_floor == 0:
        raise ValueError("variance_floor must be positive")
    missing_classes = sorted(set(range(N_CLASSES)) - set(labels.tolist()))
    if missing_classes:
        raise ValueError(f"missing source classes: {missing_classes}")
    user_ids = np.unique(subjects)
    rank = _positive_integer(rank, "rank", min(len(user_ids), N_CLASSES * n_components))
    feature_mean = features.mean(axis=0)
    _, _, vt = _canonical_svd(features - feature_mean)
    pca_components = vt[:n_components].copy()
    projected = (features - feature_mean) @ pca_components.T
    cell_counts = np.zeros((len(user_ids), N_CLASSES), dtype=np.int64)
    cell_means = np.full((len(user_ids), N_CLASSES, n_components), np.nan)
    for user_index, user in enumerate(user_ids):
        for label in range(N_CLASSES):
            selected = (subjects == user) & (labels == label)
            cell_counts[user_index, label] = int(selected.sum())
            if selected.any():
                cell_means[user_index, label] = projected[selected].mean(axis=0)
    observed = cell_counts > 0
    mu_projected = np.stack([cell_means[observed[:, label], label].mean(axis=0) for label in range(N_CLASSES)])
    variance = np.maximum(np.mean((projected - mu_projected[labels]) ** 2, axis=0), variance_floor)
    scale = np.sqrt(variance)
    mu = mu_projected / scale
    user_class_means = cell_means / scale
    residuals = user_class_means - mu[None, :, :]
    low_rank = masked_low_rank(residuals, observed, rank, max_iter, tol)
    B = (low_rank.right_vectors.T * (low_rank.singular_values / np.sqrt(len(user_ids)))).reshape(N_CLASSES, n_components, rank)
    metadata = {
        "protocol_date": "2026-09-28",
        "n_classes": N_CLASSES,
        "n_components": n_components,
        "rank": rank,
        "source_user_ids": user_ids.tolist(),
        "source_user_count": len(user_ids),
        "source_sample_count": len(features),
        "candidate_classes": list(CANDIDATE_CLASSES),
        "variance_floor": variance_floor,
        "variance_definition": "mean_source_squared_PCA_residual_about_user_balanced_class_center",
        "variance_includes_between_user_residual": True,
        "loading_scale": "singular_values/sqrt(n_source_users)",
        "ridge_affine_regularizes_intercept": True,
        "imputation": dict(low_rank.diagnostics),
    }
    return SourceCalibration(
        feature_mean=feature_mean,
        pca_components=pca_components,
        variance=variance,
        scale=scale,
        mu=mu,
        B=B,
        source_user_ids=user_ids.copy(),
        source_user_count=len(user_ids),
        source_sample_count=len(features),
        source_ids=source_ids,
        observed_mask=observed.copy(),
        cell_counts=cell_counts,
        user_class_means=user_class_means,
        diagnostics=low_rank.diagnostics,
        metadata=metadata,
    )


def select_classes(B):
    """Maximize logdet(I + sum_c B_c.T B_c); lexicographic numerical ties."""
    B = np.asarray(B, dtype=np.float64)
    if B.ndim != 3 or B.shape[0] != N_CLASSES or min(B.shape[1:]) < 1:
        raise ValueError("B must have shape [12,K,R]")
    if not np.isfinite(B).all():
        raise ValueError("B must be finite")
    information = np.einsum("ckr,cks->crs", B, B)
    best_classes, best_score = None, -np.inf
    for classes in combinations(CANDIDATE_CLASSES, 3):
        _, score = np.linalg.slogdet(np.eye(B.shape[-1]) + information[list(classes)].sum(axis=0))
        # A fixed numerical tolerance prevents roundoff from changing exact ties.
        if score > best_score + 1e-12:
            best_classes, best_score = classes, float(score)
    return best_classes


def prototype_scores(z, mu):
    """Frozen source Gaussian-prototype scores, [N,C], in whitened coordinates."""
    z, mu = _matrix(z, "z"), _matrix(mu, "mu")
    if z.shape[1] != mu.shape[1]:
        raise ValueError("z and mu must have matching feature dimensions")
    return -0.5 * np.sum((z[:, None, :] - mu[None, :, :]) ** 2, axis=-1)


def correction_scores(base_logits, z, mu, delta, tau):
    """Add the Gaussian mean-shift log likelihood ratio to frozen logits.

    tau=0 returns an independent, bitwise-identical copy with the input dtype.
    """
    base = np.asarray(base_logits)
    if base.ndim != 2 or base.dtype.kind not in "iuf" or not np.isfinite(base).all():
        raise ValueError("base_logits must be a finite [N,C] numeric matrix")
    z, mu, delta = _matrix(z, "z"), _matrix(mu, "mu"), _matrix(delta, "delta")
    if base.shape != (len(z), len(mu)) or delta.shape != mu.shape or z.shape[1] != mu.shape[1]:
        raise ValueError("base_logits, z, mu and delta shapes do not match")
    tau = _nonnegative(tau, "tau")
    if tau == 0:
        return base.copy()
    correction = ((z[:, None, :] - mu[None, :, :]) * delta[None, :, :]).sum(axis=-1)
    correction -= 0.5 * (delta ** 2).sum(axis=-1)
    return base + tau * correction


@dataclass
class FixedPartition:
    """Indices refer to input row positions; IDs are preserved global identifiers."""
    support_indices: np.ndarray
    support_ids: np.ndarray
    support_labels: np.ndarray
    support_subjects: np.ndarray
    query_indices: np.ndarray
    query_ids: np.ndarray
    query_labels: np.ndarray
    query_subjects: np.ndarray


@dataclass
class SupportSample:
    indices: np.ndarray
    ids: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    seed: int
    classes: Tuple[int, ...]


def fixed_partition(ids, labels, subjects):
    """Reserve 20 support candidates per available user/class; use the rest for query.

    Per-cell RNG seed is 91000 + user*100 + class. A missing class is skipped;
    an observed cell with fewer than 20 samples raises instead of changing the
    fixed support budget. Inputs may contain one user or several users.
    """
    labels = _labels(labels)
    subjects = _subjects(subjects, len(labels))
    ids = _ids(ids, len(labels))
    support_parts, query_parts = [], []
    for user in np.unique(subjects):
        for label in np.unique(labels[subjects == user]):
            pool = np.flatnonzero((subjects == user) & (labels == label))
            if len(pool) < 20:
                raise ValueError(f"user {user}, class {label} needs at least 20 support candidates")
            shuffled = np.random.default_rng(91000 + int(user) * 100 + int(label)).permutation(pool)
            support_parts.append(shuffled[:20])
            query_parts.append(shuffled[20:])
    support = np.concatenate(support_parts) if support_parts else np.empty(0, dtype=np.int64)
    query = np.concatenate(query_parts) if query_parts else np.empty(0, dtype=np.int64)
    if np.intersect1d(support, query).size or len(support) + len(query) != len(ids):
        raise ValueError("support/query indices must be disjoint and cover all rows")
    if set(ids[support].tolist()).intersection(ids[query].tolist()):
        raise ValueError("support/query IDs must be disjoint")
    return FixedPartition(support, ids[support], labels[support], subjects[support],
                          query, ids[query], labels[query], subjects[query])


def sample_support(partition, user, classes, seed):
    """Select three per requested class, using one RNG in ascending class order.

    Seeds are fixed to 92000..92019. All available classes may be passed for
    the explicitly larger-budget reference. An unavailable class raises.
    """
    user = int(_subjects([user], 1)[0])
    selected_classes = _labels(classes)
    if len(set(selected_classes.tolist())) != len(selected_classes):
        raise ValueError("support classes must be unique")
    selected_classes = np.sort(selected_classes)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed not in SUPPORT_SEEDS:
        raise ValueError("support seed must be in 92000..92019")
    rng = np.random.default_rng(seed)
    positions = []
    for label in selected_classes:
        pool = np.flatnonzero((partition.support_subjects == user) & (partition.support_labels == label))
        if len(pool) < 3:
            raise ValueError(f"user {user}, class {label} has fewer than three support candidates")
        positions.append(rng.choice(pool, size=3, replace=False))
    positions = np.concatenate(positions) if positions else np.empty(0, dtype=np.int64)
    indices, ids = partition.support_indices[positions], partition.support_ids[positions]
    if len(set(ids.tolist())) != len(ids) or set(ids.tolist()).intersection(partition.query_ids.tolist()):
        raise ValueError("sampled support IDs must be unique and disjoint from query")
    return SupportSample(indices, ids, partition.support_labels[positions],
                         partition.support_subjects[positions], int(seed), tuple(selected_classes.tolist()))
