# P0b Causal Kinematic Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved P0b-v1 minimal causal F0 plus deterministic Last/CV(2/5/20)/CA(5/20) future-trajectory baselines, with pure-synthetic C0 verification and no real-data execution.

**Architecture:** Build one new library module on top of the already-reviewed P0a `ForecastInput`/`FoldStats` contract. The module restores the training-normalized envelope, performs causal forward-only Viterbi peak tracking, analytic positive-depth trilateration, constant-velocity Kalman filtering, and deterministic kinematic extrapolation. A separate unittest file owns every synthetic/C0 proof, and a thin CLI exposes only a memory-only `--selftest` in P0b.

**Tech Stack:** Python 3.9.23, NumPy 2.0.2, standard-library `dataclasses/hashlib/struct/json/argparse/unittest/pathlib`; CPU only; no SciPy, PyTorch, GPU, cache reads, W1, or W3 execution.

**Plan status:** Inline implementation and evidence capture completed on 2026-08-01 in the isolated worktree; stopped before real-data selection/scoring for user review.

---

## Execution and repository guardrails

**Implementation worktree:**

```text
D:\uwb_gestures\.worktrees\p0a-causal-prefix
```

**Documentation workspace:**

```text
D:\uwb_gestures\proposal_v2
```

**Python command used by every test step:**

```powershell
& "D:\D_conda\envs\pytorch\python.exe"
```

**User-specific Git override:** Do not stage or commit. The earlier P0a evidence is intentionally uncommitted, and the user has not authorized a Git commit. Every task ends with a read-only status/hash checkpoint instead of `git add`/`git commit`.

**Files that must stay byte-identical:**

```text
uwb_osr/causal_data.py
tests/test_causal_forecast_protocol.py
tools/w4_causal_gate.py
```

Recorded P0a SHA-256 values:

```text
44C7CADB41596D5044EABAE17E47E55D9342AB65FDF104664FC9A11C7BC31676  uwb_osr/causal_data.py
DC6B024481CF8CDD8748F3048FFF3ED2607671F2EBDE9EAC62592296B8AD828E  tests/test_causal_forecast_protocol.py
8BA5B354D4833C9EA559E74F7B45CE553AD337FA238C01CC7D94F051BBB27AE4  tools/w4_causal_gate.py
```

**Forbidden during this plan:**

- no real `cache/` or `logs/` reads;
- no W1 trajectory generation;
- no parameter/window winner selection on real subjects;
- no ADE/FDE/FRLS/FRNLL or model ranking;
- no W3 `--track/--analyze`;
- no creation of `logs/w3_readout_scores.npz`;
- no modification of 17/18 or `tools/w3_readout_predict.py`;
- no F1/F2 or learned predictor implementation.

---

## File responsibility map

| File | Action | Single responsibility |
|---|---|---|
| `uwb_osr/causal_baselines.py` | Create | Frozen P0b types/constants; forward peak tracking; geometry/Kalman; Last/CV/CA; pure selectors; selftest manifest |
| `tests/test_causal_baselines.py` | Create | Pure-synthetic unit, C0, CLI-boundary, and determinism tests for the approved spec |
| `tools/w4_kinematic_gate.py` | Create | Thin `--selftest` JSON CLI; no real-data mode in P0b-v1 |
| `proposal_v2/specs/2026-08-01-p0b-kinematic-baseline-design.md` | Modify at final evidence step | Change status to implemented only after all gates pass |
| `proposal_v2/15_块二重构_因果状态估计_未来轨迹预测_未来回波生成.md` | Modify at final evidence step | Record exact P0b evidence and the next stop before real execution |
| `proposal_v2/specs/2026-08-01-p0b-kinematic-baseline-implementation-plan.md` | Modify while executing | Check completed boxes and append final evidence |

The production module stays separate from `uwb_osr/trajectory.py`; importing or calling its backtracking `viterbi_track` is prohibited.

---

### Task 0: Reconfirm the clean P0b starting boundary

**Files:**
- Read only: `uwb_osr/causal_data.py`
- Read only: `tests/test_causal_forecast_protocol.py`
- Read only: `tools/w4_causal_gate.py`
- Read only: `D:\uwb_gestures\proposal_v2\specs\2026-08-01-p0b-kinematic-baseline-design.md`

- [x] **Step 1: Verify the approved design status and forbidden W3 output**

Run from the documentation workspace:

```powershell
rg -n "书面规范均已.*获用户批准|P0b-v1|纯合成|NO-GO" "specs\2026-08-01-p0b-kinematic-baseline-design.md"
if (Test-Path -LiteralPath "D:\uwb_gestures\logs\w3_readout_scores.npz") { throw "forbidden W3 score exists" }
```

Expected: the approved status and boundaries are printed; no exception.

- [x] **Step 2: Verify the three P0a hashes**

Run from the implementation worktree:

```powershell
$expected = @{
  "uwb_osr/causal_data.py" = "44C7CADB41596D5044EABAE17E47E55D9342AB65FDF104664FC9A11C7BC31676"
  "tests/test_causal_forecast_protocol.py" = "DC6B024481CF8CDD8748F3048FFF3ED2607671F2EBDE9EAC62592296B8AD828E"
  "tools/w4_causal_gate.py" = "8BA5B354D4833C9EA559E74F7B45CE553AD337FA238C01CC7D94F051BBB27AE4"
}
foreach ($path in $expected.Keys) {
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
  if ($actual -ne $expected[$path]) { throw "$path hash changed: $actual" }
}
```

Expected: exit 0 with no output.

- [x] **Step 3: Run the existing P0a tests before adding P0b**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_forecast_protocol.py" -v
```

Expected: 47 tests PASS, 0 failures/errors/skips.

- [x] **Step 4: Record the uncommitted starting state**

```powershell
git status --short -- uwb_osr/causal_data.py tests/test_causal_forecast_protocol.py tools/w4_causal_gate.py uwb_osr/causal_baselines.py tests/test_causal_baselines.py tools/w4_kinematic_gate.py
```

Expected: exactly the three existing P0a files are untracked; the three P0b files do not exist yet.

---

### Task 1: Freeze constants, immutable records, validation, and provenance

**Files:**
- Create: `tests/test_causal_baselines.py`
- Create: `uwb_osr/causal_baselines.py`

- [x] **Step 1: Write the first failing contract tests**

Create `tests/test_causal_baselines.py` with imports plus this first class:

```python
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import numpy as np

from uwb_osr.causal_data import (
    CausalPrefixStore,
    FoldStats,
    ForecastInput,
    ForecastKey,
    ProtocolError,
    _fold_stats_provenance,
    build_fold_assignment,
)
import uwb_osr.causal_baselines as cb


class ContractTypesTest(unittest.TestCase):
    def test_constants_and_registries_exact(self):
        self.assertEqual(cb.PROTOCOL_VERSION, "P0b-v1")
        self.assertEqual(cb.BIN_M, 1.2 / 189)
        self.assertEqual(cb.CV_WINDOWS, (2, 5, 20))
        self.assertEqual(cb.CA_WINDOWS, (5, 20))
        self.assertEqual(cb.FORBIDDEN_CA_WINDOWS, (2,))
        self.assertEqual(cb.QR_GRID, tuple(
            cb.F0Hyperparameters(sr * cb.BIN_M, sa)
            for sa in (1.0, 3.0, 10.0)
            for sr in (1.0, 3.0, 8.0)
        ))

    def test_hyperparameters_are_frozen_and_validated(self):
        hp = cb.F0Hyperparameters(cb.BIN_M, 1.0)
        with self.assertRaises(FrozenInstanceError):
            hp.sigma_range_m = 2.0
        for bad in (0.0, -1.0, np.nan, np.inf):
            with self.assertRaises(ProtocolError):
                cb.F0Hyperparameters(bad, 1.0)
            with self.assertRaises(ProtocolError):
                cb.F0Hyperparameters(cb.BIN_M, bad)

    def test_public_record_field_names_exact(self):
        self.assertEqual(
            tuple(field.name for field in fields(cb.CausalStateTrace)),
            (
                "position_m", "velocity_mps", "state_cov", "state_available",
                "geometry_valid", "update_accepted", "peak_bin", "range_m",
                "terminal_viterbi_score", "tracker_initialized",
                "config_provenance", "state_provenance",
            ),
        )
        self.assertEqual(
            tuple(field.name for field in fields(cb.KinematicForecast)),
            (
                "key", "model_name", "window", "anchor_position_m",
                "displacement_m", "position_m", "samples_m",
                "valid_horizon_mask", "constraint_active",
                "distribution_kind", "state_provenance",
            ),
        )

    def test_config_provenance_is_deterministic(self):
        hp = cb.F0Hyperparameters(cb.BIN_M, 1.0)
        radar = np.array([[-0.55, 0.0, 0.0], [0.0, 0.55, 0.0], [0.55, 0.0, 0.0]])
        mask = np.ones(3, dtype=np.bool_)
        a = cb.f0_config_provenance("fold-stats", radar, mask, hp)
        b = cb.f0_config_provenance("fold-stats", radar.copy(), mask.copy(), hp)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)
        self.assertNotEqual(a, cb.f0_config_provenance("other", radar, mask, hp))
```

- [x] **Step 2: Run the contract test to prove the module is missing**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -k ContractTypesTest -v
```

Expected: FAIL during import with `ModuleNotFoundError: No module named 'uwb_osr.causal_baselines'`.

- [x] **Step 3: Implement constants, immutable records, and binary hashing**

Create `uwb_osr/causal_baselines.py` with the exact public constants and records below. Use manual `__slots__` because the approved interpreter is Python 3.9.

```python
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

import uwb_osr.causal_data as causal_data
from uwb_osr.causal_data import FoldStats, ForecastInput, ForecastKey, ProtocolError


PROTOCOL_VERSION = "P0b-v1"
N_BINS = 189
RANGE_WINDOW_M = 1.2
BIN_M = RANGE_WINDOW_M / N_BINS
FPS = 20.0
DT = 1.0 / FPS
MAX_JUMP = 20
JUMP_PENALTY = 0.15
EMISSION_FLOOR = 1e-6
POWER_EPS = 1e-30
Z_MIN_M = 0.2
Z_MAX_M = 1.2
GEOMETRY_COND_MAX = 1e8
REPROJECTION_ATOL_M = 1e-9
INNOVATION_D2_MAX = 16.26623619623813
CV_WINDOWS = (2, 5, 20)
CA_WINDOWS = (5, 20)
FORBIDDEN_CA_WINDOWS = (2,)


@dataclass(frozen=True)
class F0Hyperparameters:
    __slots__ = ("sigma_range_m", "sigma_accel_mps2")
    sigma_range_m: float
    sigma_accel_mps2: float

    def __post_init__(self) -> None:
        for name, value in (
            ("sigma_range_m", self.sigma_range_m),
            ("sigma_accel_mps2", self.sigma_accel_mps2),
        ):
            if not np.isfinite(value) or value <= 0.0:
                raise ProtocolError(f"{name} must be finite and strictly positive")


QR_GRID = tuple(
    F0Hyperparameters(sr * BIN_M, sa)
    for sa in (1.0, 3.0, 10.0)
    for sr in (1.0, 3.0, 8.0)
)


@dataclass(frozen=True)
class CausalStateTrace:
    __slots__ = (
        "position_m", "velocity_mps", "state_cov", "state_available",
        "geometry_valid", "update_accepted", "peak_bin", "range_m",
        "terminal_viterbi_score", "tracker_initialized",
        "config_provenance", "state_provenance",
    )
    position_m: np.ndarray
    velocity_mps: np.ndarray
    state_cov: np.ndarray
    state_available: np.ndarray
    geometry_valid: np.ndarray
    update_accepted: np.ndarray
    peak_bin: np.ndarray
    range_m: np.ndarray
    terminal_viterbi_score: np.ndarray
    tracker_initialized: np.ndarray
    config_provenance: str
    state_provenance: str


@dataclass(frozen=True)
class KinematicForecast:
    __slots__ = (
        "key", "model_name", "window", "anchor_position_m",
        "displacement_m", "position_m", "samples_m",
        "valid_horizon_mask", "constraint_active",
        "distribution_kind", "state_provenance",
    )
    key: ForecastKey
    model_name: str
    window: int | None
    anchor_position_m: np.ndarray
    displacement_m: np.ndarray
    position_m: np.ndarray
    samples_m: np.ndarray
    valid_horizon_mask: np.ndarray
    constraint_active: bool
    distribution_kind: str
    state_provenance: str


@dataclass(frozen=True)
class MotionCaps:
    __slots__ = ("speed_mps", "accel_mps2")
    speed_mps: float
    accel_mps2: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.speed_mps) or self.speed_mps <= 0.0:
            raise ProtocolError("speed_mps must be finite and strictly positive")
        if not np.isfinite(self.accel_mps2) or self.accel_mps2 <= 0.0:
            raise ProtocolError("accel_mps2 must be finite and strictly positive")


def _readonly(value: np.ndarray, dtype: np.dtype | str) -> np.ndarray:
    out = np.array(value, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


def _hash_text(digest: "hashlib._Hash", value: str) -> None:
    payload = value.encode("utf-8")
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def f0_config_provenance(
    norm_provenance: str,
    radar_positions: np.ndarray,
    valid_view_mask: np.ndarray,
    hyperparameters: F0Hyperparameters,
) -> str:
    radar = np.asarray(radar_positions, dtype="<f8")
    mask = np.asarray(valid_view_mask, dtype=np.uint8)
    if radar.shape != (3, 3) or mask.shape != (3,):
        raise ProtocolError("radar geometry or view mask has the wrong shape")
    if not np.all(np.isfinite(radar)):
        raise ProtocolError("radar geometry must be finite")
    digest = hashlib.sha256()
    _hash_text(digest, PROTOCOL_VERSION)
    _hash_text(digest, norm_provenance)
    digest.update(struct.pack("<d", FPS))
    digest.update(struct.pack("<q", N_BINS))
    digest.update(struct.pack("<d", RANGE_WINDOW_M))
    digest.update(struct.pack("<q", MAX_JUMP))
    for value in (
        JUMP_PENALTY, EMISSION_FLOOR, POWER_EPS, hyperparameters.sigma_range_m,
        hyperparameters.sigma_accel_mps2,
    ):
        digest.update(struct.pack("<d", value))
    digest.update(np.ascontiguousarray(radar).tobytes())
    digest.update(np.ascontiguousarray(mask).tobytes())
    return digest.hexdigest()
```

- [x] **Step 4: Run the contract tests**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -k ContractTypesTest -v
```

Expected: all `ContractTypesTest` tests PASS.

- [x] **Step 5: Checkpoint without staging**

```powershell
git status --short -- uwb_osr/causal_baselines.py tests/test_causal_baselines.py
Get-FileHash -Algorithm SHA256 uwb_osr/causal_baselines.py,tests/test_causal_baselines.py
```

Expected: both files are untracked; hashes print; no staging or commit.

---

### Task 2: Restore the envelope and implement forward-only Viterbi/sub-bin range

**Files:**
- Modify: `tests/test_causal_baselines.py`
- Modify: `uwb_osr/causal_baselines.py`

- [x] **Step 1: Add failing forward-tracker tests**

Append a `ForwardViterbiTest` that checks known paths, deterministic ties, invalid frames, prefix extension, sub-bin interpolation, and provenance mismatch. The fixture must create a valid envelope `FoldStats` with this helper:

```python
def make_stats(
    source_shape: tuple[int, int, int, int] = (8, 3, 90, 189),
    representation: str = "envelope",
) -> FoldStats:
    fold = build_fold_assignment(1)
    center = np.zeros(3, dtype="<f8")
    scale = np.ones(3, dtype="<f8")
    center.setflags(write=False)
    scale.setflags(write=False)
    provenance = _fold_stats_provenance(
        fold, representation, center, scale, source_shape, np.dtype(np.float32).str
    )
    return FoldStats(
        fold=fold,
        representation=representation,
        center=center,
        scale=scale,
        source_shape=source_shape,
        source_dtype=np.dtype(np.float32).str,
        provenance=provenance,
    )


class ForwardViterbiTest(unittest.TestCase):
    def test_known_path_and_prefix_extension(self):
        envelope = np.zeros((3, 8, 189), dtype=np.float64)
        path = np.array([40, 41, 42, 44, 45, 46, 48, 49])
        for view in range(3):
            envelope[view, np.arange(path.size), path + view] = 10.0
        full = cb.forward_peak_trace(envelope, np.ones(3, dtype=np.bool_))
        short = cb.forward_peak_trace(envelope[:, :5], np.ones(3, dtype=np.bool_))
        np.testing.assert_array_equal(full.peak_bin[:, :5], short.peak_bin)
        np.testing.assert_array_equal(full.peak_bin[0], path)

    def test_tie_uses_lowest_bin(self):
        envelope = np.zeros((3, 1, 189), dtype=np.float64)
        envelope[:, 0, 10] = 2.0
        envelope[:, 0, 11] = 2.0
        out = cb.forward_peak_trace(envelope, np.ones(3, dtype=np.bool_))
        np.testing.assert_array_equal(out.peak_bin[:, 0], np.array([10, 10, 10]))

    def test_invalid_frame_is_not_a_measurement(self):
        envelope = np.zeros((3, 3, 189), dtype=np.float64)
        envelope[:, 0, 30] = 2.0
        envelope[:, 2, 31] = 2.0
        out = cb.forward_peak_trace(envelope, np.ones(3, dtype=np.bool_))
        np.testing.assert_array_equal(out.peak_bin[:, 1], -np.ones(3, dtype=np.int64))
        self.assertTrue(np.isnan(out.range_m[:, 1]).all())
        self.assertTrue(out.tracker_initialized.all())

    def test_subbin_parabola_exact(self):
        power = np.zeros(189, dtype=np.float64)
        power[49:52] = (3.0, 4.0, 3.5)
        delta = cb.parabolic_subbin(power, 50)
        self.assertAlmostEqual(delta, 1.0 / 6.0, places=14)

    def test_restore_envelope_requires_matching_stats(self):
        stats = make_stats()
        prefix = np.zeros((3, 20, 189), dtype=np.float32)
        prefix.setflags(write=False)
        radar = np.array([[-0.55, 0.0, 0.0], [0.0, 0.55, 0.0], [0.55, 0.0, 0.0]])
        radar.setflags(write=False)
        mask = np.ones(3, dtype=np.bool_)
        mask.setflags(write=False)
        good = ForecastInput(ForecastKey(0, 20, 5), prefix, radar, mask, stats.provenance)
        restored = cb.restore_envelope(good, stats)
        np.testing.assert_array_equal(restored, 0.0)
        bad = ForecastInput(ForecastKey(0, 20, 5), prefix, radar, mask, "wrong")
        with self.assertRaises(ProtocolError):
            cb.restore_envelope(bad, stats)
        with self.assertRaises(ProtocolError):
            cb.restore_envelope(good, make_stats(representation="signed"))
```

- [x] **Step 2: Run the tracker tests and confirm missing APIs**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -k ForwardViterbiTest -v
```

Expected: FAIL because `forward_peak_trace`, `parabolic_subbin`, and `restore_envelope` do not exist.

- [x] **Step 3: Implement the exact forward-only APIs**

Add an internal frozen `ForwardPeakTrace`, `restore_envelope`, `parabolic_subbin`, `_transition_only`, and `forward_peak_trace`. The recurrence must keep no backpointer and must expose a copy of the terminal score for validation-only continuation.

```python
@dataclass(frozen=True)
class ForwardPeakTrace:
    __slots__ = ("peak_bin", "range_m", "terminal_score", "tracker_initialized")
    peak_bin: np.ndarray
    range_m: np.ndarray
    terminal_score: np.ndarray
    tracker_initialized: np.ndarray


def restore_envelope(forecast_input: ForecastInput, stats: FoldStats) -> np.ndarray:
    causal_data._validate_stats(stats)
    if stats.representation != "envelope":
        raise ProtocolError("F0 requires envelope FoldStats")
    if forecast_input.norm_provenance != stats.provenance:
        raise ProtocolError("ForecastInput and FoldStats provenance differ")
    prefix = np.asarray(forecast_input.prefix)
    if prefix.ndim != 3 or prefix.shape[0] != 3 or prefix.shape[2] != N_BINS:
        raise ProtocolError("prefix must have shape (3,t,189)")
    restored = prefix.astype(np.float64) * stats.scale[:, None, None]
    restored += stats.center[:, None, None]
    return np.maximum(restored, 0.0)


def parabolic_subbin(power: np.ndarray, peak_bin: int) -> float:
    if peak_bin <= 0 or peak_bin >= N_BINS - 1:
        return 0.0
    left, center, right = power[peak_bin - 1 : peak_bin + 2]
    denominator = left - 2.0 * center + right
    if not np.isfinite(denominator) or abs(denominator) <= POWER_EPS:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))


def _transition_only(score: np.ndarray) -> np.ndarray:
    result = np.full(N_BINS, -np.inf, dtype=np.float64)
    for offset in range(-MAX_JUMP, MAX_JUMP + 1):
        penalty = JUMP_PENALTY * abs(offset)
        if offset > 0:
            result[offset:] = np.maximum(result[offset:], score[:-offset] - penalty)
        elif offset < 0:
            result[:offset] = np.maximum(result[:offset], score[-offset:] - penalty)
        else:
            result = np.maximum(result, score)
    return result


def forward_peak_trace(envelope: np.ndarray, valid_view_mask: np.ndarray) -> ForwardPeakTrace:
    values = np.asarray(envelope, dtype=np.float64)
    mask = np.asarray(valid_view_mask, dtype=np.bool_)
    if values.ndim != 3 or values.shape[0] != 3 or values.shape[2] != N_BINS:
        raise ProtocolError("envelope must have shape (3,t,189)")
    if mask.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ProtocolError("invalid envelope or view mask")
    frames = values.shape[1]
    peaks = np.full((3, frames), -1, dtype=np.int64)
    ranges = np.full((3, frames), np.nan, dtype=np.float64)
    terminal = np.full((3, N_BINS), -np.inf, dtype=np.float64)
    initialized = np.zeros(3, dtype=np.bool_)
    for view in range(3):
        score = terminal[view].copy()
        for frame in range(frames):
            power = values[view, frame] ** 2
            valid = bool(mask[view] and np.max(power) > 0.0)
            if not initialized[view]:
                if not valid:
                    continue
                emission = np.log(power / (np.max(power) + POWER_EPS) + EMISSION_FLOOR)
                score = emission
                initialized[view] = True
            else:
                score = _transition_only(score)
                if valid:
                    score += np.log(power / (np.max(power) + POWER_EPS) + EMISSION_FLOOR)
            if valid:
                peak = int(np.argmax(score))
                peaks[view, frame] = peak
                ranges[view, frame] = (peak + parabolic_subbin(power, peak)) * BIN_M
        terminal[view] = score
    return ForwardPeakTrace(
        _readonly(peaks, np.int64),
        _readonly(ranges, np.float64),
        _readonly(terminal, np.float64),
        _readonly(initialized, np.bool_),
    )
```

- [x] **Step 4: Run tracker plus contract tests**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -v
```

Expected: all selected tests PASS.

- [x] **Step 5: Check the static no-backtracking boundary**

```powershell
$forbidden = rg -n "viterbi_track|fit_trajectory|backpointer|backtrack" uwb_osr/causal_baselines.py
$code = $LASTEXITCODE
if ($code -eq 0) { $forbidden; throw "forbidden backtracking dependency found" }
if ($code -gt 1) { throw "rg failed with exit $code" }
Write-Output "no forbidden backtracking dependency"
```

Expected: no matches.

---

### Task 3: Implement analytic positive-depth trilateration and covariance propagation

**Files:**
- Modify: `tests/test_causal_baselines.py`
- Modify: `uwb_osr/causal_baselines.py`

- [x] **Step 1: Add failing geometry tests**

Add tests for exact point recovery, negative $z^2$, range limits, degenerate geometry, and the fact that an exact residual is not an accuracy certificate.

```python
class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.radar = np.array([
            [-0.55, 0.0, 0.0],
            [0.0, 0.55, 0.0],
            [0.55, 0.0, 0.0],
        ])
        self.hp = cb.F0Hyperparameters(cb.BIN_M, 3.0)

    def test_recovers_known_positive_depth_point(self):
        truth = np.array([0.12, -0.08, 0.45])
        ranges = np.linalg.norm(truth[None, :] - self.radar, axis=1)
        obs = cb.trilaterate_ranges(ranges, self.radar, self.hp.sigma_range_m)
        self.assertTrue(obs.valid)
        np.testing.assert_allclose(obs.position_m, truth, atol=1e-12, rtol=0.0)
        self.assertTrue(np.all(np.linalg.eigvalsh(obs.position_cov) > 0.0))

    def test_negative_z_squared_is_invalid_not_clipped(self):
        obs = cb.trilaterate_ranges(np.array([0.1, 0.1, 0.1]), self.radar, cb.BIN_M)
        self.assertFalse(obs.valid)
        self.assertTrue(np.isnan(obs.position_m).all())

    def test_range_window_is_half_open_and_fail_closed(self):
        for ranges in (
            np.array([-0.01, 0.7, 0.7]),
            np.array([cb.RANGE_WINDOW_M, 0.7, 0.7]),
        ):
            self.assertFalse(cb.trilaterate_ranges(ranges, self.radar, cb.BIN_M).valid)

    def test_degenerate_radar_geometry_is_invalid(self):
        radar = np.array([[-0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
        obs = cb.trilaterate_ranges(np.array([0.7, 0.5, 0.7]), radar, cb.BIN_M)
        self.assertFalse(obs.valid)

    def test_exact_reprojection_is_only_consistency(self):
        truth = np.array([0.0, 0.0, 0.45])
        ranges = np.linalg.norm(truth[None, :] - self.radar, axis=1)
        obs = cb.trilaterate_ranges(ranges, self.radar, cb.BIN_M)
        reprojection = np.linalg.norm(obs.position_m[None, :] - self.radar, axis=1)
        self.assertLess(np.max(np.abs(reprojection - ranges)), cb.REPROJECTION_ATOL_M)
        self.assertEqual(obs.quality_semantics, "numerical_consistency_only")
```

- [x] **Step 2: Run geometry tests and observe the missing API**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -k GeometryTest -v
```

Expected: FAIL because `trilaterate_ranges` and `GeometryObservation` do not exist.

- [x] **Step 3: Implement geometry without nonlinear repair or clipping**

```python
@dataclass(frozen=True)
class GeometryObservation:
    __slots__ = ("valid", "position_m", "position_cov", "quality_semantics")
    valid: bool
    position_m: np.ndarray
    position_cov: np.ndarray
    quality_semantics: str


def _invalid_geometry() -> GeometryObservation:
    return GeometryObservation(
        False,
        _readonly(np.full(3, np.nan), np.float64),
        _readonly(np.full((3, 3), np.nan), np.float64),
        "numerical_consistency_only",
    )


def trilaterate_ranges(
    ranges_m: np.ndarray,
    radar_positions: np.ndarray,
    sigma_range_m: float,
) -> GeometryObservation:
    ranges = np.asarray(ranges_m, dtype=np.float64)
    radar = np.asarray(radar_positions, dtype=np.float64)
    if ranges.shape != (3,) or radar.shape != (3, 3):
        raise ProtocolError("trilateration requires three ranges and (3,3) geometry")
    if not np.all(np.isfinite(ranges)) or not np.all(np.isfinite(radar)):
        return _invalid_geometry()
    if np.any(ranges < 0.0) or np.any(ranges >= RANGE_WINDOW_M):
        return _invalid_geometry()
    if not np.isfinite(sigma_range_m) or sigma_range_m <= 0.0:
        raise ProtocolError("sigma_range_m must be finite and positive")
    xy = radar[:, :2]
    matrix = 2.0 * (xy[1:] - xy[0])
    if np.any(radar[:, 2] != 0.0) or np.linalg.cond(matrix) > GEOMETRY_COND_MAX:
        return _invalid_geometry()
    rhs = (
        ranges[0] ** 2 - ranges[1:] ** 2
        + np.sum(xy[1:] ** 2, axis=1) - np.sum(xy[0] ** 2)
    )
    horizontal = np.linalg.solve(matrix, rhs)
    z_squared = float(np.mean(ranges**2 - np.sum((horizontal - xy) ** 2, axis=1)))
    if not np.isfinite(z_squared) or z_squared <= 0.0:
        return _invalid_geometry()
    position = np.array([horizontal[0], horizontal[1], np.sqrt(z_squared)])
    if not (Z_MIN_M < position[2] < Z_MAX_M):
        return _invalid_geometry()
    delta = position[None, :] - radar
    predicted = np.linalg.norm(delta, axis=1)
    if np.max(np.abs(predicted - ranges)) > REPROJECTION_ATOL_M:
        return _invalid_geometry()
    jacobian = delta / predicted[:, None]
    information = jacobian.T @ jacobian
    if np.linalg.cond(information) > GEOMETRY_COND_MAX:
        return _invalid_geometry()
    covariance = sigma_range_m**2 * np.linalg.inv(information)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)) or np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
        return _invalid_geometry()
    return GeometryObservation(
        True,
        _readonly(position, np.float64),
        _readonly(covariance, np.float64),
        "numerical_consistency_only",
    )
```

- [x] **Step 4: Run all tests accumulated through geometry**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -v
```

Expected: all current P0b tests PASS.

---

### Task 4: Implement the constant-velocity Kalman and end-to-end F0 trace

**Files:**
- Modify: `tests/test_causal_baselines.py`
- Modify: `uwb_osr/causal_baselines.py`

- [x] **Step 1: Add failing Kalman and F0 C0 tests**

The tests must lock F/G/H/Q, initialization, predict-only behavior, innovation rejection, covariance properties, state/horizon independence, and prefix extension. Add helpers that render a narrow positive envelope at the three geometric ranges of a static or linear synthetic point.

```python
class KalmanAndF0Test(unittest.TestCase):
    def test_kalman_matrices_exact(self):
        hp = cb.F0Hyperparameters(cb.BIN_M, 3.0)
        transition, control, observation, process_cov = cb.kalman_matrices(hp)
        expected_transition = np.block([
            [np.eye(3), cb.DT * np.eye(3)],
            [np.zeros((3, 3)), np.eye(3)],
        ])
        expected_control = np.vstack([0.5 * cb.DT**2 * np.eye(3), cb.DT * np.eye(3)])
        np.testing.assert_array_equal(transition, expected_transition)
        np.testing.assert_array_equal(control, expected_control)
        np.testing.assert_array_equal(observation, np.hstack([np.eye(3), np.zeros((3, 3))]))
        np.testing.assert_allclose(process_cov, expected_control @ (9.0 * np.eye(3)) @ expected_control.T)

    def test_predict_only_grows_covariance(self):
        hp = cb.F0Hyperparameters(cb.BIN_M, 3.0)
        state = np.zeros(6)
        covariance = np.eye(6)
        predicted_state, predicted_cov = cb.kalman_predict(state, covariance, hp)
        self.assertGreater(np.trace(predicted_cov), np.trace(covariance))
        np.testing.assert_array_equal(predicted_state, state)

    def test_innovation_rejection_returns_prediction(self):
        hp = cb.F0Hyperparameters(cb.BIN_M, 1.0)
        predicted_state = np.zeros(6)
        predicted_cov = 1e-6 * np.eye(6)
        measurement = np.array([1.0, 1.0, 1.0])
        updated, covariance, accepted = cb.kalman_update(
            predicted_state, predicted_cov, measurement, 1e-6 * np.eye(3)
        )
        self.assertFalse(accepted)
        np.testing.assert_array_equal(updated, predicted_state)
        np.testing.assert_array_equal(covariance, predicted_cov)

    def test_joseph_update_is_symmetric_positive_definite(self):
        predicted_state = np.zeros(6)
        predicted_cov = np.diag([0.2, 0.3, 0.4, 1.0, 1.0, 1.0])
        measurement = np.array([0.01, -0.02, 0.03])
        updated, covariance, accepted = cb.kalman_update(
            predicted_state, predicted_cov, measurement, 0.01 * np.eye(3)
        )
        self.assertTrue(accepted)
        self.assertTrue(np.isfinite(updated).all())
        np.testing.assert_array_equal(covariance, covariance.T)
        self.assertGreater(np.min(np.linalg.eigvalsh(covariance)), 0.0)

    def test_initialization_and_invalid_geometry_predict_only_are_exact(self):
        stats, forecast_input = make_static_input_with_invalid_last_frame()
        hp = cb.F0Hyperparameters(8.0 * cb.BIN_M, 10.0)
        trace = cb.run_f0(forecast_input, stats, hp)
        first = int(np.flatnonzero(trace.state_available)[0])
        first_geometry = cb.trilaterate_ranges(
            trace.range_m[:, first], forecast_input.radar_positions, hp.sigma_range_m
        )
        np.testing.assert_array_equal(trace.velocity_mps[first], np.zeros(3))
        np.testing.assert_allclose(
            trace.state_cov[first, :3, :3], first_geometry.position_cov,
            atol=0.0, rtol=0.0,
        )
        self.assertFalse(trace.geometry_valid[-1])
        self.assertFalse(trace.update_accepted[-1])
        self.assertTrue(trace.state_available[-1])
        expected_state, expected_cov = cb.kalman_predict(
            np.concatenate([trace.position_m[-2], trace.velocity_mps[-2]]),
            trace.state_cov[-2], hp,
        )
        np.testing.assert_array_equal(trace.position_m[-1], expected_state[:3])
        np.testing.assert_array_equal(trace.velocity_mps[-1], expected_state[3:])
        np.testing.assert_array_equal(trace.state_cov[-1], expected_cov)

    def test_f0_state_is_horizon_independent(self):
        stats, input_h5, input_h20 = make_static_forecast_inputs()
        hp = cb.F0Hyperparameters(8.0 * cb.BIN_M, 10.0)
        a = cb.run_f0(input_h5, stats, hp)
        b = cb.run_f0(input_h20, stats, hp)
        self.assertEqual(a.state_provenance, b.state_provenance)
        for name in (
            "position_m", "velocity_mps", "state_cov", "state_available",
            "geometry_valid", "update_accepted", "peak_bin", "range_m",
            "terminal_viterbi_score", "tracker_initialized",
        ):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))

    def test_prefix_extension_preserves_history(self):
        stats, input_short, input_long = make_static_prefix_extension_inputs()
        hp = cb.F0Hyperparameters(8.0 * cb.BIN_M, 10.0)
        short = cb.run_f0(input_short, stats, hp)
        long = cb.run_f0(input_long, stats, hp)
        frames = input_short.key.prefix_t
        for name in (
            "position_m", "velocity_mps", "state_cov", "state_available",
            "geometry_valid", "update_accepted",
        ):
            self.assertEqual(
                getattr(short, name).tobytes(), getattr(long, name)[:frames].tobytes()
            )
        self.assertEqual(short.peak_bin.tobytes(), long.peak_bin[:, :frames].tobytes())
        self.assertEqual(short.range_m.tobytes(), long.range_m[:, :frames].tobytes())

    def test_trace_outputs_are_read_only_and_complete(self):
        stats, forecast_input, _ = make_static_forecast_inputs()
        trace = cb.run_f0(
            forecast_input, stats, cb.F0Hyperparameters(8.0 * cb.BIN_M, 10.0)
        )
        self.assertEqual(trace.position_m.shape, (20, 3))
        self.assertEqual(trace.state_cov.shape, (20, 6, 6))
        for field in (
            "position_m", "velocity_mps", "state_cov", "state_available",
            "geometry_valid", "update_accepted", "peak_bin", "range_m",
            "terminal_viterbi_score", "tracker_initialized",
        ):
            self.assertFalse(getattr(trace, field).flags.writeable)
```

Add these exact target-free helpers above the test class. They render already-represented envelope prefixes, so Task 4 tests isolate P0b from P0a's Hilbert transform while still using the real `ForecastInput` contract:

```python
RADAR = np.array([
    [-0.55, 0.0, 0.0],
    [0.0, 0.55, 0.0],
    [0.55, 0.0, 0.0],
], dtype=np.float64)
RADAR.setflags(write=False)
VALID_VIEWS = np.ones(3, dtype=np.bool_)
VALID_VIEWS.setflags(write=False)


def render_envelope(positions_m: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise AssertionError("positions_m must have shape (t,3)")
    envelope = np.zeros((3, positions.shape[0], 189), dtype=np.float32)
    ranges = np.linalg.norm(positions[:, None, :] - RADAR[None, :, :], axis=2)
    bins = np.rint(ranges / cb.BIN_M).astype(np.int64)
    if np.any((bins <= 0) | (bins >= 188)):
        raise AssertionError("synthetic range peak is outside the interior bins")
    for frame in range(positions.shape[0]):
        for view in range(3):
            envelope[view, frame, bins[frame, view]] = 10.0
    envelope.setflags(write=False)
    return envelope


def make_input(envelope: np.ndarray, stats: FoldStats, key: ForecastKey) -> ForecastInput:
    prefix = np.array(envelope, dtype=np.float32, copy=True)
    prefix.setflags(write=False)
    return ForecastInput(key, prefix, RADAR, VALID_VIEWS, stats.provenance)


def static_positions(frames: int) -> np.ndarray:
    return np.repeat(np.array([[0.0, 0.0, 0.45]]), frames, axis=0)


def make_static_forecast_inputs() -> tuple[FoldStats, ForecastInput, ForecastInput]:
    stats = make_stats()
    envelope = render_envelope(static_positions(20))
    return (
        stats,
        make_input(envelope, stats, ForecastKey(0, 20, 5)),
        make_input(envelope, stats, ForecastKey(0, 20, 20)),
    )


def make_static_prefix_extension_inputs() -> tuple[FoldStats, ForecastInput, ForecastInput]:
    stats = make_stats()
    long_envelope = np.array(render_envelope(static_positions(30)), copy=True)
    long_envelope[:, 20:] = np.roll(long_envelope[:, 20:], shift=9, axis=2)
    long_envelope.setflags(write=False)
    return (
        stats,
        make_input(long_envelope[:, :20], stats, ForecastKey(0, 20, 5)),
        make_input(long_envelope, stats, ForecastKey(0, 30, 5)),
    )


def make_static_input_with_invalid_last_frame() -> tuple[FoldStats, ForecastInput]:
    stats = make_stats()
    envelope = np.array(render_envelope(static_positions(20)), copy=True)
    envelope[0, -1] = 0.0
    envelope.setflags(write=False)
    return stats, make_input(envelope, stats, ForecastKey(0, 20, 5))
```

- [x] **Step 2: Run the Kalman/F0 tests and observe missing APIs**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -k KalmanAndF0Test -v
```

Expected: FAIL because the Kalman and `run_f0` functions do not exist.

- [x] **Step 3: Implement exact matrix, predict, update, and state hashing helpers**

Implement these APIs exactly as frozen. `kalman_update` must use the Joseph form and reject before changing state/covariance.

```python
def kalman_matrices(hyperparameters: F0Hyperparameters) -> tuple[np.ndarray, ...]:
    transition = np.block([
        [np.eye(3), DT * np.eye(3)],
        [np.zeros((3, 3)), np.eye(3)],
    ])
    control = np.vstack([0.5 * DT**2 * np.eye(3), DT * np.eye(3)])
    observation = np.hstack([np.eye(3), np.zeros((3, 3))])
    process_cov = control @ (
        hyperparameters.sigma_accel_mps2**2 * np.eye(3)
    ) @ control.T
    return transition, control, observation, process_cov


def kalman_predict(
    state: np.ndarray,
    covariance: np.ndarray,
    hyperparameters: F0Hyperparameters,
) -> tuple[np.ndarray, np.ndarray]:
    transition, _, _, process_cov = kalman_matrices(hyperparameters)
    predicted_state = transition @ state
    predicted_cov = transition @ covariance @ transition.T + process_cov
    return predicted_state, 0.5 * (predicted_cov + predicted_cov.T)


def kalman_update(
    predicted_state: np.ndarray,
    predicted_cov: np.ndarray,
    measurement: np.ndarray,
    measurement_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    observation = np.hstack([np.eye(3), np.zeros((3, 3))])
    innovation = measurement - observation @ predicted_state
    innovation_cov = observation @ predicted_cov @ observation.T + measurement_cov
    d_squared = float(innovation @ np.linalg.solve(innovation_cov, innovation))
    if not np.isfinite(d_squared) or d_squared > INNOVATION_D2_MAX:
        return predicted_state.copy(), predicted_cov.copy(), False
    gain = np.linalg.solve(innovation_cov, observation @ predicted_cov).T
    updated_state = predicted_state + gain @ innovation
    identity = np.eye(6)
    residual = identity - gain @ observation
    updated_cov = residual @ predicted_cov @ residual.T + gain @ measurement_cov @ gain.T
    updated_cov = 0.5 * (updated_cov + updated_cov.T)
    if not np.all(np.isfinite(updated_cov)) or np.min(np.linalg.eigvalsh(updated_cov)) <= 0.0:
        raise ProtocolError("Kalman covariance is not finite positive definite")
    return updated_state, updated_cov, True
```

- [x] **Step 4: Implement `run_f0` as the only end-to-end state entrypoint**

Add the exact input validator, state-array registry, state hash, and frame loop below. This fixes the state hash field order and makes the exclusion of `horizon` mechanically visible.

```python
TRACE_ARRAY_DTYPES = (
    ("position_m", "<f8"),
    ("velocity_mps", "<f8"),
    ("state_cov", "<f8"),
    ("state_available", np.dtype(np.bool_)),
    ("geometry_valid", np.dtype(np.bool_)),
    ("update_accepted", np.dtype(np.bool_)),
    ("peak_bin", "<i8"),
    ("range_m", "<f8"),
    ("terminal_viterbi_score", "<f8"),
    ("tracker_initialized", np.dtype(np.bool_)),
)


def _validate_forecast_input(forecast_input: ForecastInput) -> ForecastInput:
    if not isinstance(forecast_input, ForecastInput):
        raise ProtocolError("run_f0 requires exactly one ForecastInput")
    causal_data.validate_key(forecast_input.key)
    prefix = forecast_input.prefix
    radar = forecast_input.radar_positions
    mask = forecast_input.valid_view_mask
    if not isinstance(prefix, np.ndarray) or prefix.dtype != np.dtype(np.float32):
        raise ProtocolError("ForecastInput.prefix must be a float32 NumPy array")
    if prefix.shape != (3, forecast_input.key.prefix_t, N_BINS):
        raise ProtocolError("ForecastInput.prefix shape disagrees with its key")
    if not isinstance(radar, np.ndarray) or radar.dtype != np.dtype(np.float64):
        raise ProtocolError("radar_positions must be a float64 NumPy array")
    if radar.shape != (3, 3) or not np.all(np.isfinite(radar)):
        raise ProtocolError("radar_positions must be finite with shape (3,3)")
    if not isinstance(mask, np.ndarray) or mask.dtype != np.dtype(np.bool_):
        raise ProtocolError("valid_view_mask must be a bool NumPy array")
    if mask.shape != (3,):
        raise ProtocolError("valid_view_mask must have shape (3,)")
    if prefix.flags.writeable or radar.flags.writeable or mask.flags.writeable:
        raise ProtocolError("ForecastInput arrays must be read-only")
    if not np.all(np.isfinite(prefix)):
        raise ProtocolError("ForecastInput.prefix contains NaN or Inf")
    return forecast_input


def state_trace_provenance(
    sample_idx: int,
    prefix_t: int,
    config_provenance: str,
    arrays: Mapping[str, np.ndarray],
) -> str:
    names = tuple(name for name, _ in TRACE_ARRAY_DTYPES)
    if tuple(arrays) != names:
        raise ProtocolError("state provenance array order is not canonical")
    digest = hashlib.sha256()
    _hash_text(digest, config_provenance)
    digest.update(struct.pack("<q", int(sample_idx)))
    digest.update(struct.pack("<q", int(prefix_t)))
    for name, dtype in TRACE_ARRAY_DTYPES:
        value = np.ascontiguousarray(np.asarray(arrays[name], dtype=dtype))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _trace_array_mapping(
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    state_cov: np.ndarray,
    state_available: np.ndarray,
    geometry_valid: np.ndarray,
    update_accepted: np.ndarray,
    peak_bin: np.ndarray,
    range_m: np.ndarray,
    terminal_viterbi_score: np.ndarray,
    tracker_initialized: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "position_m": position_m,
        "velocity_mps": velocity_mps,
        "state_cov": state_cov,
        "state_available": state_available,
        "geometry_valid": geometry_valid,
        "update_accepted": update_accepted,
        "peak_bin": peak_bin,
        "range_m": range_m,
        "terminal_viterbi_score": terminal_viterbi_score,
        "tracker_initialized": tracker_initialized,
    }


def run_f0(
    forecast_input: ForecastInput,
    stats: FoldStats,
    hyperparameters: F0Hyperparameters,
) -> CausalStateTrace:
    forecast_input = _validate_forecast_input(forecast_input)
    if not isinstance(hyperparameters, F0Hyperparameters):
        raise ProtocolError("hyperparameters must be F0Hyperparameters")
    envelope = restore_envelope(forecast_input, stats)
    peaks = forward_peak_trace(envelope, forecast_input.valid_view_mask)
    frames = forecast_input.key.prefix_t
    position = np.full((frames, 3), np.nan, dtype=np.float64)
    velocity = np.full((frames, 3), np.nan, dtype=np.float64)
    state_cov = np.full((frames, 6, 6), np.nan, dtype=np.float64)
    state_available = np.zeros(frames, dtype=np.bool_)
    geometry_valid = np.zeros(frames, dtype=np.bool_)
    update_accepted = np.zeros(frames, dtype=np.bool_)
    state = None
    covariance = None
    for frame in range(frames):
        geometry = trilaterate_ranges(
            peaks.range_m[:, frame],
            forecast_input.radar_positions,
            hyperparameters.sigma_range_m,
        )
        geometry_valid[frame] = geometry.valid
        accepted = False
        if state is None:
            if geometry.valid:
                velocity_std = MAX_JUMP * BIN_M / DT
                state = np.concatenate([geometry.position_m, np.zeros(3)])
                covariance = np.block([
                    [geometry.position_cov, np.zeros((3, 3))],
                    [np.zeros((3, 3)), velocity_std**2 * np.eye(3)],
                ])
                accepted = True
        else:
            state, covariance = kalman_predict(state, covariance, hyperparameters)
            if geometry.valid:
                state, covariance, accepted = kalman_update(
                    state, covariance, geometry.position_m, geometry.position_cov
                )
        if state is not None:
            state_available[frame] = True
            update_accepted[frame] = accepted
            position[frame] = state[:3]
            velocity[frame] = state[3:]
            state_cov[frame] = covariance

    config_hash = f0_config_provenance(
        stats.provenance,
        forecast_input.radar_positions,
        forecast_input.valid_view_mask,
        hyperparameters,
    )
    arrays = _trace_array_mapping(
        _readonly(position, np.float64),
        _readonly(velocity, np.float64),
        _readonly(state_cov, np.float64),
        _readonly(state_available, np.bool_),
        _readonly(geometry_valid, np.bool_),
        _readonly(update_accepted, np.bool_),
        peaks.peak_bin,
        peaks.range_m,
        peaks.terminal_score,
        peaks.tracker_initialized,
    )
    state_hash = state_trace_provenance(
        forecast_input.key.sample_idx,
        forecast_input.key.prefix_t,
        config_hash,
        arrays,
    )
    return CausalStateTrace(
        arrays["position_m"], arrays["velocity_mps"], arrays["state_cov"],
        arrays["state_available"], arrays["geometry_valid"],
        arrays["update_accepted"], arrays["peak_bin"], arrays["range_m"],
        arrays["terminal_viterbi_score"], arrays["tracker_initialized"],
        config_hash, state_hash,
    )
```

- [x] **Step 5: Run accumulated P0b tests and existing P0a tests**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -v
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_forecast_protocol.py" -v
```

Expected: all P0b tests so far PASS; P0a remains 47/47 PASS.

---

### Task 5: Implement training-fold motion caps and Last/CV/CA forecasts

**Files:**
- Modify: `tests/test_causal_baselines.py`
- Modify: `uwb_osr/causal_baselines.py`

- [x] **Step 1: Add failing exact-trajectory tests**

Create synthetic `CausalStateTrace` fixtures directly from exact position arrays so this task tests prediction formulas independently of range quantization. Add these helpers before the test class:

```python
def frozen(value: np.ndarray, dtype: object) -> np.ndarray:
    out = np.array(value, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def linear_positions(
    length: int,
    velocity: np.ndarray = np.array([0.2, -0.1, 0.05]),
) -> np.ndarray:
    current = np.array([0.05, -0.03, 0.45])
    times = (np.arange(length) - (length - 1))[:, None] * cb.DT
    return current + times * np.asarray(velocity, dtype=np.float64)


def quadratic_positions(
    length: int,
    velocity: np.ndarray = np.array([0.2, -0.1, 0.05]),
    acceleration: np.ndarray = np.array([0.5, 0.3, -0.2]),
) -> np.ndarray:
    current = np.array([0.05, -0.03, 0.45])
    times = (np.arange(length) - (length - 1))[:, None] * cb.DT
    return (
        current
        + times * np.asarray(velocity, dtype=np.float64)
        + 0.5 * times**2 * np.asarray(acceleration, dtype=np.float64)
    )


def make_exact_trace(
    positions_m: np.ndarray,
    available_from: int = 0,
    rejected_frames: tuple[int, ...] = (),
) -> cb.CausalStateTrace:
    source = np.asarray(positions_m, dtype=np.float64)
    frames = source.shape[0]
    available = np.arange(frames) >= available_from
    accepted = available.copy()
    accepted[list(rejected_frames)] = False
    position = source.copy()
    position[~available] = np.nan
    velocity = np.zeros((frames, 3), dtype=np.float64)
    velocity[~available] = np.nan
    covariance = np.repeat(np.eye(6)[None, :, :], frames, axis=0)
    covariance[~available] = np.nan
    peak_bin = np.full((3, frames), -1, dtype=np.int64)
    range_m = np.full((3, frames), np.nan, dtype=np.float64)
    terminal = np.full((3, 189), -np.inf, dtype=np.float64)
    tracker = np.zeros(3, dtype=np.bool_)
    arrays = {
        "position_m": frozen(position, np.float64),
        "velocity_mps": frozen(velocity, np.float64),
        "state_cov": frozen(covariance, np.float64),
        "state_available": frozen(available, np.bool_),
        "geometry_valid": frozen(available, np.bool_),
        "update_accepted": frozen(accepted, np.bool_),
        "peak_bin": frozen(peak_bin, np.int64),
        "range_m": frozen(range_m, np.float64),
        "terminal_viterbi_score": frozen(terminal, np.float64),
        "tracker_initialized": frozen(tracker, np.bool_),
    }
    config_hash = hashlib.sha256(b"synthetic-exact-trace").hexdigest()
    state_hash = cb.state_trace_provenance(0, frames, config_hash, arrays)
    return cb.CausalStateTrace(
        arrays["position_m"], arrays["velocity_mps"], arrays["state_cov"],
        arrays["state_available"], arrays["geometry_valid"],
        arrays["update_accepted"], arrays["peak_bin"], arrays["range_m"],
        arrays["terminal_viterbi_score"], arrays["tracker_initialized"],
        config_hash, state_hash,
    )
```

Then add tests for Last, all CV windows, both CA windows, anchoring, missing windows, constraints, read-only outputs, K=1, and CA-2 rejection.

```python
class KinematicForecastTest(unittest.TestCase):
    def test_last_is_zero_displacement(self):
        trace = make_exact_trace(linear_positions(20))
        key = ForecastKey(0, 20, 20)
        forecast = cb.forecast_last(trace, key)
        np.testing.assert_array_equal(forecast.displacement_m, np.zeros((20, 3)))
        np.testing.assert_array_equal(forecast.position_m, np.repeat(trace.position_m[-1:], 20, axis=0))
        self.assertEqual(forecast.samples_m.shape, (1, 20, 3))
        self.assertEqual(forecast.distribution_kind, "deterministic_delta")

    def test_all_cv_windows_recover_linear_motion(self):
        velocity = np.array([0.3, -0.2, 0.1])
        trace = make_exact_trace(linear_positions(20, velocity))
        caps = cb.MotionCaps(10.0, 100.0)
        for window in cb.CV_WINDOWS:
            forecast = cb.forecast_cv(trace, ForecastKey(0, 20, 20), window, caps)
            expected = trace.position_m[-1] + np.arange(1, 21)[:, None] * cb.DT * velocity
            np.testing.assert_allclose(forecast.position_m, expected, atol=1e-12, rtol=0.0)

    def test_cv2_equals_last_finite_difference(self):
        positions = linear_positions(20, np.array([0.4, 0.0, 0.0]))
        trace = make_exact_trace(positions)
        slope = cb.fit_cv_velocity(trace.position_m[-2:])
        np.testing.assert_array_equal(slope, (positions[-1] - positions[-2]) / cb.DT)

    def test_ca_windows_recover_quadratic_motion(self):
        velocity = np.array([0.2, -0.1, 0.05])
        acceleration = np.array([0.5, 0.3, -0.2])
        trace = make_exact_trace(quadratic_positions(20, velocity, acceleration))
        caps = cb.MotionCaps(10.0, 100.0)
        for window in cb.CA_WINDOWS:
            forecast = cb.forecast_ca(trace, ForecastKey(0, 20, 20), window, caps)
            h = np.arange(1, 21)[:, None] * cb.DT
            expected = trace.position_m[-1] + h * velocity + 0.5 * h**2 * acceleration
            np.testing.assert_allclose(forecast.position_m, expected, atol=1e-10, rtol=0.0)

    def test_ca2_is_rejected(self):
        trace = make_exact_trace(linear_positions(20))
        key = ForecastKey(0, 20, 5)
        with self.assertRaises(ProtocolError):
            cb.forecast_ca(trace, key, 2, cb.MotionCaps(1.0, 1.0))
        with self.assertRaises(ProtocolError):
            cb._build_forecast(
                key, "CA-2", 2, trace.position_m[-1], np.zeros((5, 3)),
                np.repeat(trace.position_m[-1:], 5, axis=0), np.ones(5, dtype=np.bool_),
                False, trace.state_provenance,
            )

    def test_unavailable_window_keeps_all_horizon_steps(self):
        trace = make_exact_trace(linear_positions(20), available_from=18)
        forecast = cb.forecast_cv(trace, ForecastKey(0, 20, 10), 5, cb.MotionCaps(1.0, 1.0))
        self.assertEqual(forecast.position_m.shape, (10, 3))
        self.assertFalse(forecast.valid_horizon_mask.any())
        self.assertTrue(np.isnan(forecast.position_m).all())

    def test_regressions_anchor_at_current_state(self):
        positions = linear_positions(20)
        positions[-5:-1] += np.array([0.02, -0.01, 0.03])
        trace = make_exact_trace(positions)
        caps = cb.MotionCaps(10.0, 100.0)
        for forecast in (
            cb.forecast_cv(trace, ForecastKey(0, 20, 5), 5, caps),
            cb.forecast_ca(trace, ForecastKey(0, 20, 5), 5, caps),
        ):
            np.testing.assert_array_equal(forecast.anchor_position_m, positions[-1])
            np.testing.assert_allclose(
                forecast.position_m,
                forecast.anchor_position_m + forecast.displacement_m,
                atol=0.0, rtol=0.0,
            )

    def test_norm_constraints_flags_and_read_only_k1(self):
        trace = make_exact_trace(quadratic_positions(
            20, np.array([4.0, 0.0, 0.0]), np.array([20.0, 0.0, 0.0])
        ))
        caps = cb.MotionCaps(0.5, 1.0)
        forecasts = (
            cb.forecast_last(trace, ForecastKey(0, 20, 5)),
            cb.forecast_cv(trace, ForecastKey(0, 20, 5), 5, caps),
            cb.forecast_ca(trace, ForecastKey(0, 20, 5), 5, caps),
        )
        self.assertFalse(forecasts[0].constraint_active)
        self.assertTrue(forecasts[1].constraint_active)
        self.assertTrue(forecasts[2].constraint_active)
        for forecast in forecasts:
            self.assertEqual(forecast.samples_m.shape, (1, 5, 3))
            self.assertEqual(forecast.distribution_kind, "deterministic_delta")
            for array in (
                forecast.anchor_position_m, forecast.displacement_m,
                forecast.position_m, forecast.samples_m,
                forecast.valid_horizon_mask,
            ):
                self.assertFalse(array.flags.writeable)
```

- [x] **Step 2: Run forecast tests and confirm missing functions**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -k KinematicForecastTest -v
```

Expected: FAIL because forecast APIs are missing.

- [x] **Step 3: Implement exact OLS and deterministic forecast packaging**

```python
def _clipnorm(vector: np.ndarray, maximum: float) -> tuple[np.ndarray, bool]:
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ProtocolError("clip maximum must be finite and positive")
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm):
        raise ProtocolError("clip vector must be finite")
    if norm <= maximum:
        return vector.copy(), False
    return vector * (maximum / norm), True


def fit_cv_velocity(positions_m: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 2:
        raise ProtocolError("CV fit requires shape (L,3) with L>=2")
    if not np.all(np.isfinite(positions)):
        raise ProtocolError("CV fit positions must be finite")
    length = positions.shape[0]
    if length == 2:
        return (positions[1] - positions[0]) / DT
    times = (np.arange(length) - (length - 1)) * DT
    design = np.column_stack([np.ones(length), times])
    coefficients = np.linalg.lstsq(design, positions, rcond=None)[0]
    return coefficients[1]


def fit_ca_coefficients(positions_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 3:
        raise ProtocolError("CA fit requires shape (L,3) with L>=3")
    if not np.all(np.isfinite(positions)):
        raise ProtocolError("CA fit positions must be finite")
    length = positions.shape[0]
    times = (np.arange(length) - (length - 1)) * DT
    design = np.column_stack([np.ones(length), times, 0.5 * times**2])
    coefficients = np.linalg.lstsq(design, positions, rcond=None)[0]
    return coefficients[1], coefficients[2]


def _trace_arrays_from_record(trace: CausalStateTrace) -> dict[str, np.ndarray]:
    return {
        "position_m": trace.position_m,
        "velocity_mps": trace.velocity_mps,
        "state_cov": trace.state_cov,
        "state_available": trace.state_available,
        "geometry_valid": trace.geometry_valid,
        "update_accepted": trace.update_accepted,
        "peak_bin": trace.peak_bin,
        "range_m": trace.range_m,
        "terminal_viterbi_score": trace.terminal_viterbi_score,
        "tracker_initialized": trace.tracker_initialized,
    }


def _validate_trace_for_key(trace: CausalStateTrace, key: ForecastKey) -> None:
    causal_data.validate_key(key)
    if not isinstance(trace, CausalStateTrace):
        raise ProtocolError("trace must be a CausalStateTrace")
    frames = key.prefix_t
    expected_shapes = {
        "position_m": (frames, 3),
        "velocity_mps": (frames, 3),
        "state_cov": (frames, 6, 6),
        "state_available": (frames,),
        "geometry_valid": (frames,),
        "update_accepted": (frames,),
        "peak_bin": (3, frames),
        "range_m": (3, frames),
        "terminal_viterbi_score": (3, N_BINS),
        "tracker_initialized": (3,),
    }
    expected_dtypes = {
        "position_m": np.dtype(np.float64),
        "velocity_mps": np.dtype(np.float64),
        "state_cov": np.dtype(np.float64),
        "state_available": np.dtype(np.bool_),
        "geometry_valid": np.dtype(np.bool_),
        "update_accepted": np.dtype(np.bool_),
        "peak_bin": np.dtype(np.int64),
        "range_m": np.dtype(np.float64),
        "terminal_viterbi_score": np.dtype(np.float64),
        "tracker_initialized": np.dtype(np.bool_),
    }
    arrays = _trace_arrays_from_record(trace)
    for name, value in arrays.items():
        if not isinstance(value, np.ndarray) or value.shape != expected_shapes[name]:
            raise ProtocolError(f"trace field {name} has the wrong shape")
        if value.dtype != expected_dtypes[name]:
            raise ProtocolError(f"trace field {name} has the wrong dtype")
        if value.flags.writeable:
            raise ProtocolError(f"trace field {name} must be read-only")
    expected_hash = state_trace_provenance(
        key.sample_idx, key.prefix_t, trace.config_provenance, arrays
    )
    if trace.state_provenance != expected_hash:
        raise ProtocolError("trace provenance disagrees with ForecastKey or arrays")


def _build_forecast(
    key: ForecastKey,
    model_name: str,
    window: int | None,
    anchor_position_m: np.ndarray,
    displacement_m: np.ndarray,
    position_m: np.ndarray,
    valid_horizon_mask: np.ndarray,
    constraint_active: bool,
    state_provenance: str,
) -> KinematicForecast:
    causal_data.validate_key(key)
    registry = {
        "Last": None,
        "CV-2": 2,
        "CV-5": 5,
        "CV-20": 20,
        "CA-5": 5,
        "CA-20": 20,
    }
    if model_name not in registry or registry[model_name] != window:
        raise ProtocolError("model name/window is not in the frozen registry")
    anchor = np.asarray(anchor_position_m, dtype=np.float64)
    displacement = np.asarray(displacement_m, dtype=np.float64)
    positions = np.asarray(position_m, dtype=np.float64)
    mask = np.asarray(valid_horizon_mask, dtype=np.bool_)
    if anchor.shape != (3,) or displacement.shape != (key.horizon, 3):
        raise ProtocolError("forecast anchor/displacement shape is invalid")
    if positions.shape != (key.horizon, 3) or mask.shape != (key.horizon,):
        raise ProtocolError("forecast position/mask shape is invalid")
    if np.any(mask):
        if not np.all(mask) or not np.all(np.isfinite(anchor)):
            raise ProtocolError("P0b forecasts are valid for all steps or none")
        if not np.all(np.isfinite(displacement)) or not np.all(np.isfinite(positions)):
            raise ProtocolError("valid forecast values must be finite")
        if not np.allclose(
            positions, anchor[None, :] + displacement, atol=1e-12, rtol=0.0
        ):
            raise ProtocolError("position must equal anchor plus displacement")
    elif not (np.isnan(displacement).all() and np.isnan(positions).all()):
        raise ProtocolError("invalid forecast values must be NaN")
    if not isinstance(constraint_active, (bool, np.bool_)):
        raise ProtocolError("constraint_active must be boolean")
    if not isinstance(state_provenance, str) or len(state_provenance) != 64:
        raise ProtocolError("state_provenance must be a SHA-256 hex string")
    frozen_position = _readonly(positions, np.float64)
    return KinematicForecast(
        key,
        model_name,
        window,
        _readonly(anchor, np.float64),
        _readonly(displacement, np.float64),
        frozen_position,
        _readonly(frozen_position[None, ...], np.float64),
        _readonly(mask, np.bool_),
        bool(constraint_active),
        "deterministic_delta",
        state_provenance,
    )


def _invalid_forecast(
    trace: CausalStateTrace, key: ForecastKey, model_name: str, window: int | None
) -> KinematicForecast:
    anchor = (
        trace.position_m[-1].copy()
        if trace.state_available[-1]
        else np.full(3, np.nan)
    )
    values = np.full((key.horizon, 3), np.nan)
    return _build_forecast(
        key, model_name, window, anchor, values, values,
        np.zeros(key.horizon, dtype=np.bool_), False, trace.state_provenance,
    )


def forecast_last(trace: CausalStateTrace, key: ForecastKey) -> KinematicForecast:
    _validate_trace_for_key(trace, key)
    if not trace.state_available[-1]:
        return _invalid_forecast(trace, key, "Last", None)
    anchor = trace.position_m[-1]
    displacement = np.zeros((key.horizon, 3), dtype=np.float64)
    positions = np.repeat(anchor[None, :], key.horizon, axis=0)
    return _build_forecast(
        key, "Last", None, anchor, displacement, positions,
        np.ones(key.horizon, dtype=np.bool_), False, trace.state_provenance,
    )


def forecast_cv(
    trace: CausalStateTrace,
    key: ForecastKey,
    window: int,
    caps: MotionCaps,
) -> KinematicForecast:
    if window not in CV_WINDOWS:
        raise ProtocolError("unregistered CV window")
    if not isinstance(caps, MotionCaps):
        raise ProtocolError("caps must be MotionCaps")
    _validate_trace_for_key(trace, key)
    if not np.all(trace.state_available[-window:]):
        return _invalid_forecast(trace, key, f"CV-{window}", window)
    anchor = trace.position_m[-1]
    velocity, clipped = _clipnorm(fit_cv_velocity(trace.position_m[-window:]), caps.speed_mps)
    times = np.arange(1, key.horizon + 1, dtype=np.float64)[:, None] * DT
    displacement = times * velocity
    return _build_forecast(
        key, f"CV-{window}", window, anchor, displacement,
        anchor[None, :] + displacement,
        np.ones(key.horizon, dtype=np.bool_), clipped, trace.state_provenance,
    )


def forecast_ca(
    trace: CausalStateTrace,
    key: ForecastKey,
    window: int,
    caps: MotionCaps,
) -> KinematicForecast:
    if window in FORBIDDEN_CA_WINDOWS or window not in CA_WINDOWS:
        raise ProtocolError("unregistered or forbidden CA window")
    if not isinstance(caps, MotionCaps):
        raise ProtocolError("caps must be MotionCaps")
    _validate_trace_for_key(trace, key)
    if not np.all(trace.state_available[-window:]):
        return _invalid_forecast(trace, key, f"CA-{window}", window)
    anchor = trace.position_m[-1]
    fitted_velocity, fitted_acceleration = fit_ca_coefficients(
        trace.position_m[-window:]
    )
    velocity, velocity_clipped = _clipnorm(fitted_velocity, caps.speed_mps)
    acceleration, acceleration_clipped = _clipnorm(
        fitted_acceleration, caps.accel_mps2
    )
    displacement = np.empty((key.horizon, 3), dtype=np.float64)
    current_position = anchor.copy()
    constraint_active = velocity_clipped or acceleration_clipped
    for step in range(key.horizon):
        next_velocity, clipped = _clipnorm(
            velocity + acceleration * DT, caps.speed_mps
        )
        constraint_active = constraint_active or clipped
        current_position = current_position + 0.5 * (velocity + next_velocity) * DT
        displacement[step] = current_position - anchor
        velocity = next_velocity
    return _build_forecast(
        key, f"CA-{window}", window, anchor, displacement,
        anchor[None, :] + displacement,
        np.ones(key.horizon, dtype=np.bool_),
        constraint_active, trace.state_provenance,
    )
```

- [x] **Step 4: Implement exact nearest-rank training motion caps**

Expose:

```python
def nearest_rank_99(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    if (
        finite.ndim != 1
        or finite.size < 100
        or not np.all(np.isfinite(finite))
        or np.any(finite < 0.0)
    ):
        raise ProtocolError("at least 100 finite non-negative values are required")
    ordered = np.sort(finite)
    index = int(np.ceil(0.99 * ordered.size) - 1)
    return float(ordered[index])


def accepted_motion_norms(trace: CausalStateTrace) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(trace, CausalStateTrace):
        raise ProtocolError("trace must be a CausalStateTrace")
    accepted = np.asarray(trace.update_accepted, dtype=np.bool_)
    positions = np.asarray(trace.position_m, dtype=np.float64)
    if accepted.ndim != 1 or positions.shape != (accepted.size, 3):
        raise ProtocolError("trace motion fields have incompatible shapes")
    speed_mask = accepted[1:] & accepted[:-1]
    accel_mask = accepted[2:] & accepted[1:-1] & accepted[:-2]
    speeds = np.linalg.norm(np.diff(positions, axis=0)[speed_mask] / DT, axis=1)
    accelerations = np.linalg.norm(
        np.diff(positions, n=2, axis=0)[accel_mask] / DT**2, axis=1
    )
    if not np.all(np.isfinite(speeds)) or not np.all(np.isfinite(accelerations)):
        raise ProtocolError("accepted motion differences must be finite")
    return _readonly(speeds, np.float64), _readonly(accelerations, np.float64)


def aggregate_motion_caps(
    speed_norms_by_subject: Mapping[int, np.ndarray],
    accel_norms_by_subject: Mapping[int, np.ndarray],
    *,
    role: str,
) -> MotionCaps:
    if role != "train":
        raise ProtocolError("motion caps can only be built from training subjects")
    if set(speed_norms_by_subject) != set(accel_norms_by_subject):
        raise ProtocolError("speed and acceleration subjects differ")
    if len(speed_norms_by_subject) != 5:
        raise ProtocolError("exactly five training subjects are required")
    speed = sorted(nearest_rank_99(speed_norms_by_subject[s]) for s in speed_norms_by_subject)
    accel = sorted(nearest_rank_99(accel_norms_by_subject[s]) for s in accel_norms_by_subject)
    return MotionCaps(speed[2], accel[2])
```

Add these tests to `KinematicForecastTest` so nearest-rank indexing, the five-subject median, the role boundary, and no fallback are executable rather than prose-only:

```python
    def test_nearest_rank_and_five_subject_median_exact(self):
        self.assertEqual(cb.nearest_rank_99(np.arange(1.0, 101.0)), 99.0)
        speed = {subject: np.full(100, float(subject)) for subject in range(1, 6)}
        accel = {subject: np.full(100, float(10 * subject)) for subject in range(1, 6)}
        caps = cb.aggregate_motion_caps(speed, accel, role="train")
        self.assertEqual(caps, cb.MotionCaps(3.0, 30.0))

    def test_motion_norms_use_only_consecutive_accepted_updates(self):
        trace = make_exact_trace(
            linear_positions(20, np.array([0.4, 0.0, 0.0])),
            rejected_frames=(10,),
        )
        speed, accel = cb.accepted_motion_norms(trace)
        np.testing.assert_allclose(speed, 0.4, atol=1e-14, rtol=0.0)
        np.testing.assert_allclose(accel, 0.0, atol=1e-12, rtol=0.0)
        self.assertEqual(speed.size, 17)
        self.assertEqual(accel.size, 15)

    def test_motion_caps_fail_closed_without_fallback(self):
        speed = {subject: np.ones(100) for subject in range(1, 6)}
        accel = {subject: np.ones(100) for subject in range(1, 6)}
        speed[1] = np.ones(99)
        with self.assertRaises(ProtocolError):
            cb.aggregate_motion_caps(speed, accel, role="train")
        with self.assertRaises(ProtocolError):
            cb.aggregate_motion_caps(
                {subject: np.ones(100) for subject in range(1, 6)},
                accel,
                role="validation",
            )
```

- [x] **Step 5: Run forecast and cumulative tests**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -v
```

Expected: all P0b tests PASS.

---

### Task 6: Implement pure validation selectors and the CV relative-improvement helper

**Files:**
- Modify: `tests/test_causal_baselines.py`
- Modify: `uwb_osr/causal_baselines.py`

- [x] **Step 1: Add failing selector tests**

```python
class SelectorTest(unittest.TestCase):
    def test_huberized_absolute_loss_has_metre_units(self):
        delta = 3.0 * cb.BIN_M
        self.assertAlmostEqual(cb.huberized_absolute_loss(delta, delta), 0.5 * delta)
        self.assertAlmostEqual(cb.huberized_absolute_loss(2.0 * delta, delta), 1.5 * delta)

    def test_qr_selector_requires_all_nine_validation_candidates(self):
        losses = {hp: np.full(4, 0.1 + i * 0.01) for i, hp in enumerate(cb.QR_GRID)}
        selected = cb.select_f0_hyperparameters(losses, role="validation")
        self.assertEqual(selected, cb.QR_GRID[0])
        incomplete = dict(losses)
        incomplete.pop(cb.QR_GRID[-1])
        with self.assertRaises(ProtocolError):
            cb.select_f0_hyperparameters(incomplete, role="validation")
        unequal = dict(losses)
        unequal[cb.QR_GRID[-1]] = np.ones(3)
        with self.assertRaises(ProtocolError):
            cb.select_f0_hyperparameters(unequal, role="validation")

    def test_qr_tie_breaks_by_smaller_accel_then_range(self):
        losses = {hp: np.ones(4) for hp in cb.QR_GRID}
        selected = cb.select_f0_hyperparameters(losses, role="validation")
        self.assertEqual(selected.sigma_accel_mps2, 1.0)
        self.assertEqual(selected.sigma_range_m, cb.BIN_M)

    def test_qr_selector_rejects_calibration_or_test_role(self):
        losses = {hp: np.ones(4) for hp in cb.QR_GRID}
        for role in ("calibration", "test", "train"):
            with self.assertRaises(ProtocolError):
                cb.select_f0_hyperparameters(losses, role=role)

    def test_unavailable_f0_key_has_fixed_full_range_penalty(self):
        penalty = cb.f0_key_loss(
            np.array([np.nan, np.nan, np.nan]),
            np.ones(3, dtype=np.bool_),
            key_available=False,
        )
        self.assertEqual(penalty, cb.RANGE_WINDOW_M)
        self.assertEqual(cb.f0_sample_loss(np.array([0.1, 0.2, 0.3, 0.4])), 0.25)

    def test_window_selector_uses_one_window_for_all_horizons(self):
        tied = {window: np.ones((4, 3)) for window in cb.CV_WINDOWS}
        self.assertEqual(cb.select_cv_window(tied, role="validation"), 2)
        cells = {
            2: np.full((4, 3), 1.0),
            5: np.full((4, 3), 0.8),
            20: np.full((4, 3), 0.9),
        }
        self.assertEqual(cb.select_cv_window(cells, role="validation"), 5)
        cells[5][0, 0] = np.inf
        self.assertEqual(cb.select_cv_window(cells, role="validation"), 20)

    def test_ca_window_selector_uses_only_five_or_twenty(self):
        cells = {
            5: np.full((4, 3), 0.9),
            20: np.full((4, 3), 0.8),
        }
        self.assertEqual(cb.select_ca_window(cells, role="validation"), 20)
        cells[2] = np.zeros((4, 3))
        with self.assertRaises(ProtocolError):
            cb.select_ca_window(cells, role="validation")

    def test_window_selectors_reject_non_validation_roles(self):
        cv_cells = {window: np.ones((4, 3)) for window in cb.CV_WINDOWS}
        ca_cells = {window: np.ones((4, 3)) for window in cb.CA_WINDOWS}
        for role in ("train", "calibration", "test"):
            with self.assertRaises(ProtocolError):
                cb.select_cv_window(cv_cells, role=role)
            with self.assertRaises(ProtocolError):
                cb.select_ca_window(ca_cells, role=role)

    def test_relative_frnll_reduction_exact(self):
        self.assertAlmostEqual(cb.relative_frnll_reduction(10.0, 9.0), 0.1)
        with self.assertRaises(ProtocolError):
            cb.relative_frnll_reduction(0.0, 0.0)
```

- [x] **Step 2: Run selector tests and confirm missing APIs**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -k SelectorTest -v
```

Expected: FAIL because selector APIs are missing.

- [x] **Step 3: Implement deterministic pure selectors**

```python
def huberized_absolute_loss(error_m: float, delta_m: float = 3.0 * BIN_M) -> float:
    error = abs(float(error_m))
    if not np.isfinite(error) or not np.isfinite(delta_m) or delta_m <= 0.0:
        raise ProtocolError("Huber inputs must be finite and delta positive")
    return error**2 / (2.0 * delta_m) if error <= delta_m else error - 0.5 * delta_m


def f0_key_loss(
    absolute_errors_m: np.ndarray,
    valid_view_mask: np.ndarray,
    *,
    key_available: bool,
) -> float:
    if not isinstance(key_available, (bool, np.bool_)):
        raise ProtocolError("key_available must be boolean")
    if not key_available:
        return RANGE_WINDOW_M
    errors = np.asarray(absolute_errors_m, dtype=np.float64)
    mask = np.asarray(valid_view_mask, dtype=np.bool_)
    if errors.shape != (3,) or mask.shape != (3,) or not np.any(mask):
        raise ProtocolError("F0 key loss requires three errors and a non-empty view mask")
    selected = errors[mask]
    if not np.all(np.isfinite(selected)) or np.any(selected < 0.0):
        raise ProtocolError("available F0 key errors must be finite and non-negative")
    return float(np.mean([huberized_absolute_loss(value) for value in selected]))


def f0_sample_loss(four_prefix_key_losses_m: np.ndarray) -> float:
    losses = np.asarray(four_prefix_key_losses_m, dtype=np.float64)
    if losses.shape != (4,) or not np.all(np.isfinite(losses)):
        raise ProtocolError("each validation sample requires all four prefix losses")
    return float(np.mean(losses))


def select_f0_hyperparameters(
    losses_by_candidate: Mapping[F0Hyperparameters, np.ndarray],
    *,
    role: str,
) -> F0Hyperparameters:
    if role != "validation":
        raise ProtocolError("F0 hyperparameters can only be selected on validation data")
    if set(losses_by_candidate) != set(QR_GRID):
        raise ProtocolError("all and only the nine frozen candidates are required")
    scored = []
    expected_count = None
    for candidate in QR_GRID:
        losses = np.asarray(losses_by_candidate[candidate], dtype=np.float64)
        if losses.ndim != 1 or losses.size == 0 or not np.all(np.isfinite(losses)):
            raise ProtocolError("candidate losses must be a non-empty finite vector")
        if expected_count is None:
            expected_count = losses.size
        elif losses.size != expected_count:
            raise ProtocolError("all candidates must contain the same validation keys")
        scored.append((float(np.mean(losses)), candidate.sigma_accel_mps2,
                       candidate.sigma_range_m, candidate))
    return min(scored)[-1]


def _select_window(
    cells: Mapping[int, np.ndarray],
    allowed: tuple[int, ...],
    *,
    role: str,
) -> int:
    if role != "validation":
        raise ProtocolError("kinematic windows can only be selected on validation data")
    if set(cells) != set(allowed):
        raise ProtocolError("window registry mismatch")
    scored = []
    for window in allowed:
        value = np.asarray(cells[window], dtype=np.float64)
        if value.shape != (4, 3) or np.isnan(value).any():
            raise ProtocolError("window cells must have shape (4,3) without NaN")
        score = float(np.mean(value)) if np.isfinite(value).all() else np.inf
        scored.append((score, window))
    winner = min(scored)
    if not np.isfinite(winner[0]):
        raise ProtocolError("no window has all twelve finite cells")
    return winner[1]


def select_cv_window(cells: Mapping[int, np.ndarray], *, role: str) -> int:
    return _select_window(cells, CV_WINDOWS, role=role)


def select_ca_window(cells: Mapping[int, np.ndarray], *, role: str) -> int:
    return _select_window(cells, CA_WINDOWS, role=role)


def relative_frnll_reduction(cv_frnll: float, model_frnll: float) -> float:
    if not np.isfinite(cv_frnll) or cv_frnll <= 0.0 or not np.isfinite(model_frnll):
        raise ProtocolError("FRNLL values are invalid")
    return float((cv_frnll - model_frnll) / cv_frnll)
```

- [x] **Step 4: Run selectors and all prior tests**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -v
```

Expected: all P0b tests PASS.

---

### Task 7: Add end-to-end future replacement, horizon-prefix, and static-boundary tests

**Files:**
- Modify: `tests/test_causal_baselines.py`
- Modify: `uwb_osr/causal_baselines.py`

- [x] **Step 1: Add predictor interface and C0 tests**

Use `inspect.signature` to prove public predictor functions accept no `ForecastTarget`. Add this exact fixture: it creates two full signed echo cubes whose first 20 frames are identical and whose later bins differ, but exposes only the prefix through P0a `CausalPrefixStore`.

```python
def make_identical_prefix_inputs_with_different_futures(
) -> tuple[FoldStats, ForecastInput, ForecastInput]:
    cube_a = np.zeros((1, 3, 90, 189), dtype=np.float32)
    cube_b = np.zeros_like(cube_a)
    point = np.array([0.0, 0.0, 0.45])
    nominal_bins = np.rint(
        np.linalg.norm(point[None, :] - RADAR, axis=1) / cb.BIN_M
    ).astype(np.int64)
    for frame in range(90):
        for view in range(3):
            cube_a[0, view, frame, nominal_bins[view]] = 10.0
            shifted = nominal_bins[view] if frame < 20 else nominal_bins[view] + 7
            cube_b[0, view, frame, shifted] = 10.0
    stats = make_stats(tuple(cube_a.shape))
    key = ForecastKey(0, 20, 20)
    input_a = CausalPrefixStore(cube_a, stats).get(key)
    input_b = CausalPrefixStore(cube_b, stats).get(key)
    np.testing.assert_array_equal(input_a.prefix, input_b.prefix)
    return stats, input_a, input_b
```

Then prove all six forecasts are byte-identical under fixed F0/caps.

```python
class EndToEndC0Test(unittest.TestCase):
    def test_predictor_signatures_have_no_future_or_label(self):
        import inspect
        forbidden = ("future", "target", "gesture", "label", "subject")
        for function in (cb.run_f0, cb.forecast_last, cb.forecast_cv, cb.forecast_ca):
            names = tuple(inspect.signature(function).parameters)
            self.assertFalse(any(token in name for name in names for token in forbidden))

    def test_all_six_forecasts_ignore_replaced_future(self):
        stats, input_a, input_b = make_identical_prefix_inputs_with_different_futures()
        hp = cb.F0Hyperparameters(8.0 * cb.BIN_M, 10.0)
        caps = cb.MotionCaps(10.0, 100.0)
        trace_a = cb.run_f0(input_a, stats, hp)
        trace_b = cb.run_f0(input_b, stats, hp)
        self.assertEqual(trace_a.state_provenance, trace_b.state_provenance)
        forecasts_a = cb.forecast_all_baselines(trace_a, input_a.key, caps)
        forecasts_b = cb.forecast_all_baselines(trace_b, input_b.key, caps)
        self.assertEqual(tuple(f.model_name for f in forecasts_a),
                         ("Last", "CV-2", "CV-5", "CV-20", "CA-5", "CA-20"))
        for left, right in zip(forecasts_a, forecasts_b):
            self.assertEqual(left.position_m.tobytes(), right.position_m.tobytes())
            self.assertEqual(
                left.valid_horizon_mask.tobytes(), right.valid_horizon_mask.tobytes()
            )

    def test_horizon_prefix_consistency(self):
        trace = make_exact_trace(quadratic_positions(20))
        caps = cb.MotionCaps(10.0, 100.0)
        for factory, window in ((cb.forecast_cv, 5), (cb.forecast_ca, 5)):
            h5 = factory(trace, ForecastKey(0, 20, 5), window, caps)
            h10 = factory(trace, ForecastKey(0, 20, 10), window, caps)
            h20 = factory(trace, ForecastKey(0, 20, 20), window, caps)
            np.testing.assert_array_equal(h5.position_m, h20.position_m[:5])
            np.testing.assert_array_equal(h10.position_m, h20.position_m[:10])
```

- [x] **Step 2: Run C0 tests and observe the missing aggregate API**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -k EndToEndC0Test -v
```

Expected: only `forecast_all_baselines`-dependent tests fail.

- [x] **Step 3: Implement the fixed six-baseline registry**

```python
def forecast_all_baselines(
    trace: CausalStateTrace,
    key: ForecastKey,
    caps: MotionCaps,
) -> tuple[KinematicForecast, ...]:
    return (
        forecast_last(trace, key),
        forecast_cv(trace, key, 2, caps),
        forecast_cv(trace, key, 5, caps),
        forecast_cv(trace, key, 20, caps),
        forecast_ca(trace, key, 5, caps),
        forecast_ca(trace, key, 20, caps),
    )
```

- [x] **Step 4: Add and run static source boundaries**

The test reads only source text and fails if production imports W1, target, labels, PyTorch, SciPy, or real I/O helpers.

```python
    def test_production_source_has_no_forbidden_dependencies(self):
        source = Path(cb.__file__).read_text(encoding="utf-8")
        for token in (
            "viterbi_track", "fit_trajectory", "ForecastTarget", "FutureTargetStore",
            "gesture_eval", "torch", "scipy", "np.load", "open(", "cache/", "logs/",
        ):
            self.assertNotIn(token, source)
```

Run:

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -k EndToEndC0Test -v
```

Expected: all C0/static tests PASS.

---

### Task 8: Add deterministic synthetic selftest and CLI

**Files:**
- Modify: `tests/test_causal_baselines.py`
- Modify: `uwb_osr/causal_baselines.py`
- Create: `tools/w4_kinematic_gate.py`

- [x] **Step 1: Add failing selftest and CLI tests**

```python
class SelftestAndCliTest(unittest.TestCase):
    def test_selftest_manifest_is_deterministic_and_explicit(self):
        first = cb.run_synthetic_selftest()
        second = cb.run_synthetic_selftest()
        self.assertEqual(first, second)
        self.assertEqual(first["gate"], "P0b")
        self.assertEqual(first["status"], "PASS")
        self.assertTrue(first["synthetic_only"])
        self.assertEqual(first["real_signal_reads"], 0)
        self.assertEqual(first["verdict_boundary"],
                         "P0b contract PASS != trajectory predictability PASS")
        self.assertEqual(first["checks"], sorted([
            "ca2_forbidden", "ca_exact", "ca_registry", "cv_exact",
            "cv_registry", "deterministic_k", "f0_selector",
            "future_replacement", "horizon_independence", "horizon_prefix",
            "invalid_geometry_predict_only", "qr_grid_size",
            "static_point_f0", "window_selector",
        ]))

    def test_cli_two_runs_are_byte_identical(self):
        script = Path(__file__).resolve().parents[1] / "tools" / "w4_kinematic_gate.py"
        command = [sys.executable, str(script), "--selftest"]
        first = subprocess.run(command, check=False, capture_output=True)
        second = subprocess.run(command, check=False, capture_output=True)
        self.assertEqual(first.returncode, 0)
        self.assertEqual(first.stdout, second.stdout)
        self.assertEqual(first.stderr, b"")
        payload = json.loads(first.stdout)
        self.assertEqual(payload["status"], "PASS")
        self.assertNotIn("SKIPPED", first.stdout.decode("utf-8"))

    def test_cli_has_no_real_data_mode(self):
        script = Path(__file__).resolve().parents[1] / "tools" / "w4_kinematic_gate.py"
        result = subprocess.run(
            [sys.executable, str(script), "--selftest", "--select-real"],
            check=False, capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unrecognized arguments", result.stderr)
```

- [x] **Step 2: Run selftest tests and observe missing APIs/files**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -k SelftestAndCliTest -v
```

Expected: FAIL because `run_synthetic_selftest` and the CLI do not exist.

- [x] **Step 3: Implement a memory-only deterministic manifest**

Add the following complete synthetic fixture and manifest. Every substantive check calls the production API; the only full 90-frame arrays are two tiny in-memory synthetic cubes used to prove future replacement invariance.

```python
def _selftest_stats(
    source_shape: tuple[int, int, int, int] = (1, 3, 90, 189),
) -> FoldStats:
    fold = causal_data.build_fold_assignment(1)
    center = _readonly(np.zeros(3), np.float64)
    scale = _readonly(np.ones(3), np.float64)
    source_dtype = np.dtype(np.float32).str
    provenance = causal_data._fold_stats_provenance(
        fold, "envelope", center, scale, source_shape, source_dtype
    )
    return FoldStats(
        fold, "envelope", center, scale, source_shape, source_dtype, provenance
    )


def _selftest_geometry() -> tuple[np.ndarray, np.ndarray]:
    radar = _readonly(np.array([
        [-0.55, 0.0, 0.0],
        [0.0, 0.55, 0.0],
        [0.55, 0.0, 0.0],
    ]), np.float64)
    mask = _readonly(np.ones(3), np.bool_)
    return radar, mask


def _selftest_input(
    stats: FoldStats,
    horizon: int,
    *,
    invalid_last: bool = False,
) -> ForecastInput:
    radar, mask = _selftest_geometry()
    point = np.array([0.0, 0.0, 0.45])
    bins = np.rint(np.linalg.norm(point[None, :] - radar, axis=1) / BIN_M).astype(np.int64)
    envelope = np.zeros((3, 20, N_BINS), dtype=np.float32)
    for frame in range(20):
        for view in range(3):
            envelope[view, frame, bins[view]] = 10.0
    if invalid_last:
        envelope[0, -1] = 0.0
    return ForecastInput(
        ForecastKey(0, 20, horizon),
        _readonly(envelope, np.float32),
        radar,
        mask,
        stats.provenance,
    )


def _selftest_exact_trace(positions_m: np.ndarray) -> CausalStateTrace:
    positions = np.asarray(positions_m, dtype=np.float64)
    frames = positions.shape[0]
    arrays = _trace_array_mapping(
        _readonly(positions, np.float64),
        _readonly(np.zeros((frames, 3)), np.float64),
        _readonly(np.repeat(np.eye(6)[None, :, :], frames, axis=0), np.float64),
        _readonly(np.ones(frames), np.bool_),
        _readonly(np.ones(frames), np.bool_),
        _readonly(np.ones(frames), np.bool_),
        _readonly(np.full((3, frames), -1), np.int64),
        _readonly(np.full((3, frames), np.nan), np.float64),
        _readonly(np.full((3, N_BINS), -np.inf), np.float64),
        _readonly(np.zeros(3), np.bool_),
    )
    config_hash = hashlib.sha256(b"p0b-selftest-trace").hexdigest()
    state_hash = state_trace_provenance(0, frames, config_hash, arrays)
    return CausalStateTrace(
        arrays["position_m"], arrays["velocity_mps"], arrays["state_cov"],
        arrays["state_available"], arrays["geometry_valid"],
        arrays["update_accepted"], arrays["peak_bin"], arrays["range_m"],
        arrays["terminal_viterbi_score"], arrays["tracker_initialized"],
        config_hash, state_hash,
    )


def _selftest_future_replacement_forecasts(
    hyperparameters: F0Hyperparameters,
    caps: MotionCaps,
) -> bool:
    cube_a = np.zeros((1, 3, 90, N_BINS), dtype=np.float32)
    cube_b = np.zeros_like(cube_a)
    radar, _ = _selftest_geometry()
    point = np.array([0.0, 0.0, 0.45])
    bins = np.rint(np.linalg.norm(point[None, :] - radar, axis=1) / BIN_M).astype(np.int64)
    for frame in range(90):
        for view in range(3):
            cube_a[0, view, frame, bins[view]] = 10.0
            other = bins[view] if frame < 20 else bins[view] + 7
            cube_b[0, view, frame, other] = 10.0
    stats = _selftest_stats(tuple(cube_a.shape))
    key = ForecastKey(0, 20, 20)
    input_a = causal_data.CausalPrefixStore(cube_a, stats).get(key)
    input_b = causal_data.CausalPrefixStore(cube_b, stats).get(key)
    trace_a = run_f0(input_a, stats, hyperparameters)
    trace_b = run_f0(input_b, stats, hyperparameters)
    forecasts_a = forecast_all_baselines(trace_a, key, caps)
    forecasts_b = forecast_all_baselines(trace_b, key, caps)
    return bool(
        trace_a.state_provenance == trace_b.state_provenance
        and all(
            left.position_m.tobytes() == right.position_m.tobytes()
            and left.valid_horizon_mask.tobytes() == right.valid_horizon_mask.tobytes()
            for left, right in zip(forecasts_a, forecasts_b)
        )
    )


def run_synthetic_selftest() -> dict[str, object]:
    stats = _selftest_stats()
    hyperparameters = F0Hyperparameters(8.0 * BIN_M, 10.0)
    caps = MotionCaps(10.0, 100.0)
    state_h5 = run_f0(_selftest_input(stats, 5), stats, hyperparameters)
    state_h20 = run_f0(_selftest_input(stats, 20), stats, hyperparameters)
    invalid = run_f0(
        _selftest_input(stats, 5, invalid_last=True), stats, hyperparameters
    )
    previous_state = np.concatenate([
        invalid.position_m[-2], invalid.velocity_mps[-2]
    ])
    predicted_state, predicted_cov = kalman_predict(
        previous_state, invalid.state_cov[-2], hyperparameters
    )

    times = (np.arange(20) - 19)[:, None] * DT
    current = np.array([0.05, -0.03, 0.45])
    velocity = np.array([0.2, -0.1, 0.05])
    acceleration = np.array([0.5, 0.3, -0.2])
    linear_trace = _selftest_exact_trace(current + times * velocity)
    quadratic_trace = _selftest_exact_trace(
        current + times * velocity + 0.5 * times**2 * acceleration
    )
    key_h5 = ForecastKey(0, 20, 5)
    key_h20 = ForecastKey(0, 20, 20)
    cv_h5 = forecast_cv(linear_trace, key_h5, 5, caps)
    cv_h20 = forecast_cv(linear_trace, key_h20, 5, caps)
    ca_h20 = forecast_ca(quadratic_trace, key_h20, 5, caps)
    future_times = np.arange(1, 21)[:, None] * DT
    expected_cv = current + future_times * velocity
    expected_ca = current + future_times * velocity + 0.5 * future_times**2 * acceleration

    losses = {
        candidate: np.full(4, float(index), dtype=np.float64)
        for index, candidate in enumerate(QR_GRID)
    }
    window_cells = {
        2: np.full((4, 3), 1.0),
        5: np.full((4, 3), 0.8),
        20: np.full((4, 3), 0.9),
    }
    ca2_rejected = False
    try:
        forecast_ca(linear_trace, key_h5, 2, caps)
    except ProtocolError:
        ca2_rejected = True

    checks = {
        "ca2_forbidden": ca2_rejected and 2 in FORBIDDEN_CA_WINDOWS and 2 not in CA_WINDOWS,
        "ca_registry": CA_WINDOWS == (5, 20),
        "cv_exact": bool(np.allclose(cv_h20.position_m, expected_cv, atol=1e-12, rtol=0.0)),
        "cv_registry": CV_WINDOWS == (2, 5, 20),
        "deterministic_k": cv_h20.samples_m.shape == (1, 20, 3),
        "f0_selector": select_f0_hyperparameters(losses, role="validation") == QR_GRID[0],
        "future_replacement": _selftest_future_replacement_forecasts(hyperparameters, caps),
        "horizon_independence": state_h5.state_provenance == state_h20.state_provenance,
        "horizon_prefix": cv_h5.position_m.tobytes() == cv_h20.position_m[:5].tobytes(),
        "invalid_geometry_predict_only": bool(
            not invalid.geometry_valid[-1]
            and not invalid.update_accepted[-1]
            and np.allclose(invalid.position_m[-1], predicted_state[:3], atol=0.0, rtol=0.0)
            and np.allclose(invalid.velocity_mps[-1], predicted_state[3:], atol=0.0, rtol=0.0)
            and np.allclose(invalid.state_cov[-1], predicted_cov, atol=0.0, rtol=0.0)
        ),
        "ca_exact": bool(np.allclose(ca_h20.position_m, expected_ca, atol=1e-12, rtol=0.0)),
        "qr_grid_size": len(QR_GRID) == 9,
        "static_point_f0": bool(
            state_h20.state_available.all()
            and state_h20.geometry_valid.all()
            and np.isfinite(state_h20.position_m).all()
        ),
        "window_selector": select_cv_window(window_cells, role="validation") == 5,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ProtocolError("synthetic selftest failed: " + ",".join(failed))
    return {
        "checks": sorted(checks),
        "gate": "P0b",
        "protocol_version": PROTOCOL_VERSION,
        "real_signal_reads": 0,
        "status": "PASS",
        "synthetic_only": True,
        "verdict_boundary": "P0b contract PASS != trajectory predictability PASS",
    }
```

- [x] **Step 4: Create the thin JSON CLI**

Create `tools/w4_kinematic_gate.py`:

```python
#!/usr/bin/env python
"""P0b-v1 memory-only causal-kinematic contract gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uwb_osr.causal_baselines import run_synthetic_selftest
from uwb_osr.causal_data import ProtocolError


def _json(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P0b causal-kinematic gate; performs no real-data selection or scoring."
    )
    parser.add_argument("--selftest", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    _build_parser().parse_args(argv)
    try:
        manifest = run_synthetic_selftest()
    except ProtocolError as exc:
        print(_json({"error": str(exc), "gate": "P0b", "status": "FAIL"}), file=sys.stderr)
        return 1
    print(_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 5: Run selftest twice and compare raw bytes**

```powershell
$py = "D:\D_conda\envs\pytorch\python.exe"
@'
from pathlib import Path
import subprocess
import sys

script = Path("tools/w4_kinematic_gate.py")
command = [sys.executable, str(script), "--selftest"]
first = subprocess.run(command, check=False, capture_output=True)
second = subprocess.run(command, check=False, capture_output=True)
if first.returncode != 0 or second.returncode != 0:
    raise SystemExit("selftest returned non-zero")
if first.stderr or second.stderr:
    raise SystemExit("selftest wrote stderr")
if first.stdout != second.stdout:
    raise SystemExit("selftest stdout bytes differ")
sys.stdout.buffer.write(first.stdout)
'@ | & $py -
if ($LASTEXITCODE -ne 0) { throw "raw-byte selftest comparison failed" }
```

Expected: one deterministic compact JSON line with `"status":"PASS"`, `"synthetic_only":true`, and the explicit verdict boundary.

- [x] **Step 6: Run the complete P0b test file**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -v
```

Expected: every P0b test PASS, 0 failures/errors/skips.

---

### Task 9: Full regression, evidence audit, and documentation handoff

**Files:**
- Verify: `uwb_osr/causal_baselines.py`
- Verify: `tests/test_causal_baselines.py`
- Verify: `tools/w4_kinematic_gate.py`
- Modify: `D:\uwb_gestures\proposal_v2\specs\2026-08-01-p0b-kinematic-baseline-design.md`
- Modify: `D:\uwb_gestures\proposal_v2\15_块二重构_因果状态估计_未来轨迹预测_未来回波生成.md`
- Modify: `D:\uwb_gestures\proposal_v2\specs\2026-08-01-p0b-kinematic-baseline-implementation-plan.md`

- [x] **Step 1: Run P0a and P0b focused tests together**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_forecast_protocol.py" -v
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -v
```

Expected: P0a 47/47 PASS; all P0b tests PASS; no skips.

- [x] **Step 2: Run the full repository unittest suite**

```powershell
$output = & "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_*.py" -v 2>&1
$code = $LASTEXITCODE
$output
if ($code -ne 0) {
  $joined = $output -join "`n"
  $oldFailure = "test_corrupted_view_downweighted (test_reliability.TestReliabilityCalib)"
  if (-not $joined.Contains($oldFailure) -or $joined -notmatch 'FAILED \(failures=1\)') {
    throw "full suite has a new failure or error"
  }
}
```

Expected comparison boundary: no new failures relative to the recorded P0a baseline. The only permitted failure is the pre-existing `test_reliability.TestReliabilityCalib.test_corrupted_view_downweighted`; any other failure blocks completion.

- [x] **Step 3: Recheck P0a hashes and forbidden dependencies**

```powershell
$expected = @{
  "uwb_osr/causal_data.py" = "44C7CADB41596D5044EABAE17E47E55D9342AB65FDF104664FC9A11C7BC31676"
  "tests/test_causal_forecast_protocol.py" = "DC6B024481CF8CDD8748F3048FFF3ED2607671F2EBDE9EAC62592296B8AD828E"
  "tools/w4_causal_gate.py" = "8BA5B354D4833C9EA559E74F7B45CE553AD337FA238C01CC7D94F051BBB27AE4"
}
foreach ($path in $expected.Keys) {
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
  if ($actual -ne $expected[$path]) { throw "$path hash changed: $actual" }
}
$forbidden = rg -n "viterbi_track|fit_trajectory|ForecastTarget|FutureTargetStore|gesture_eval|torch|scipy|np\.load|cache/|logs/" uwb_osr/causal_baselines.py tools/w4_kinematic_gate.py
if ($LASTEXITCODE -eq 0) { $forbidden; throw "forbidden P0b dependency found" }
if ($LASTEXITCODE -gt 1) { throw "rg failed with exit $LASTEXITCODE" }
```

Expected: hashes match; `rg` prints no forbidden matches.

- [x] **Step 4: Reconfirm W3/17/18 boundary**

```powershell
if (Test-Path -LiteralPath "D:\uwb_gestures\logs\w3_readout_scores.npz") { throw "forbidden W3 score exists" }
$items = @(
  "D:\uwb_gestures\proposal_v2\17_读出失败预测_预注册.md",
  "D:\uwb_gestures\proposal_v2\18_通道D整数bin相位修正.md",
  "D:\uwb_gestures\tools\w3_readout_predict.py"
)
foreach ($item in $items) {
  $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $item
  "{0}  {1}" -f $hash.Hash,$hash.Path
}
```

Expected hashes:

```text
FE35D845BB7FEEB62050A59423E4CF1088B056440B10822B27BFBC6595E1F32F  17
2335E804440EB7B34416543D647B42F662A0E769B4A6427EC2ECBA79E8ADF7BE  18
5E597A6CDBA426CCEBCEB6609CFBCCEC4E680487B32F981BB43DBE83664F4F90  w3_readout_predict.py
```

- [x] **Step 5: Record deterministic output and new file hashes**

```powershell
$py = "D:\D_conda\envs\pytorch\python.exe"
@'
from pathlib import Path
import hashlib
import subprocess
import sys

command = [sys.executable, str(Path("tools/w4_kinematic_gate.py")), "--selftest"]
first = subprocess.run(command, check=False, capture_output=True)
second = subprocess.run(command, check=False, capture_output=True)
if first.returncode != 0 or second.returncode != 0 or first.stderr or second.stderr:
    raise SystemExit("P0b selftest failed or wrote stderr")
if first.stdout != second.stdout:
    raise SystemExit("P0b selftest stdout bytes differ")
print(f"stdout_bytes={len(first.stdout)} stdout_sha256={hashlib.sha256(first.stdout).hexdigest()}")
'@ | & $py -
if ($LASTEXITCODE -ne 0) { throw "selftest evidence capture failed" }
Get-FileHash -Algorithm SHA256 uwb_osr/causal_baselines.py,tests/test_causal_baselines.py,tools/w4_kinematic_gate.py
git status --short -- uwb_osr/causal_data.py tests/test_causal_forecast_protocol.py tools/w4_causal_gate.py uwb_osr/causal_baselines.py tests/test_causal_baselines.py tools/w4_kinematic_gate.py
```

Expected: two selftests are byte-identical; three P0b hashes print; all six P0a/P0b files remain untracked; nothing is staged.

- [x] **Step 6: Update documentation with evidence only**

Use `apply_patch` from the documentation workspace to record:

- exact focused/full test counts and the one permitted pre-existing failure;
- two-run stdout length/hash;
- exact P0b file hashes;
- unchanged P0a and W3/17/18 hashes;
- `logs/w3_readout_scores.npz` absence;
- the explicit boundary “P0b contract PASS is not trajectory predictability PASS”;
- the next stop: user evidence review before any real parameter selection, O1 run, or trajectory comparison.

Do not write “trajectory prediction succeeded”, “CV wins/loses”, or any real-data conclusion.

- [x] **Step 7: Final no-stage/no-commit checkpoint**

```powershell
git diff --check
$trailing = rg -n '[ \t]+$' uwb_osr/causal_baselines.py tests/test_causal_baselines.py tools/w4_kinematic_gate.py
if ($LASTEXITCODE -eq 0) { $trailing; throw "trailing whitespace in untracked P0b file" }
if ($LASTEXITCODE -gt 1) { throw "rg failed with exit $LASTEXITCODE" }
git status --short
```

Expected: no whitespace errors. Existing unrelated dirty/untracked files remain untouched. Do not stage or commit.

---

## Execution evidence (2026-08-01)

- P0a focused: 47/47 PASS; P0b focused: 50/50 PASS; zero skips.
- Full repository: 126 tests, exactly the pre-existing
  `test_corrupted_view_downweighted (test_reliability.TestReliabilityCalib)` failure, zero new failures and zero errors.
- P0b CLI: two stdout byte streams identical; 430 bytes; SHA-256
  `3c8e727dc8ee677c33be14a6148c82dd17003794075299517cc1b673b3f79338`.
- P0b file hashes: `2EC5CF3C…9485D`, `C67FBB68…B3751`, `B995D067…21547` in module/test/CLI order.
- P0a three hashes unchanged; 17/18/W3 hashes unchanged; `logs/w3_readout_scores.npz` absent.
- Static forbidden-dependency scan, trailing-whitespace check, and both implementation/documentation staged-file checks passed.
- The original full-suite guard repeated the test method inside unittest's parenthesized class name and therefore
  false-failed; the recorded guard now uses unittest's actual identifier. The originally typed W3 expected hash had
  65 characters because of one extra `C`; the corrected 64-character value matches the unchanged file.
- Stop boundary: `P0b contract PASS != trajectory predictability PASS`. No real selection, O1 scoring, trajectory
  comparison, W1 run, or W3 `--track/--analyze` was performed.

---

## Completion criteria

Implementation is complete only when all of the following are evidenced:

1. P0a remains 47/47 PASS and its three hashes are unchanged;
2. every P0b synthetic/C0 test passes with zero skips;
3. full repository regression has no new failures beyond the one recorded reliability failure;
4. two P0b CLI selftest runs are byte-identical;
5. forward peak history, F0 state, and forecast prefixes are horizon/future invariant;
6. line/quadratic fixtures prove CV/CA formulas exactly;
7. CA-2 and all forbidden inputs/dependencies fail closed;
8. invalid geometry/innovation use predict-only without deleting frames or keys;
9. selectors are pure and accept no cal/test role for F0 tuning;
10. no real cache/log signal data was read;
11. W3 was not run and its score file remains absent;
12. no Git staging or commit occurred;
13. documentation stops before real selection/scoring and asks for user evidence review.

---

## Plan self-review record

The plan was checked against every row of design §7 before handoff:

| Approved design area | Executable coverage in this plan |
|---|---|
| ForecastInput-only inference, provenance, future/horizon C0 | Tasks 1, 2, 4, and 7 |
| Forward-only peak tracking and sub-bin distance | Task 2 |
| Positive-depth geometry and explicit invalid states | Task 3 |
| Exact Kalman matrices, initialization, predict-only, innovation gate, Joseph covariance | Task 4 |
| Last/CV/CA formulas, continuous windows, CA-2 rejection, deterministic K=1 | Task 5 |
| Accepted-update motion caps and train-only role | Task 5 |
| Validation-only F0/window selection, fixed unavailable-key penalty, CV relative helper | Task 6 |
| Memory-only deterministic CLI selftest and no-real-mode boundary | Task 8 |
| P0a/W3 hashes, full regression, no-stage/no-commit, evidence-only docs | Tasks 0 and 9 |

Self-review results:

- unfinished-marker scan found no deferred implementation text;
- every `unittest -k` command uses one substring only; cumulative test steps run the full file;
- raw CLI determinism is compared as subprocess stdout bytes, including the final newline;
- Python 3.9 annotation syntax was checked with the approved interpreter;
- current read-only preflight still shows exactly the three P0a files untracked with all three recorded hashes matching;
- current read-only preflight confirms `logs/w3_readout_scores.npz` is absent and the recorded W3/17/18 hashes are unchanged;
- no P0b production/test/CLI file was created during plan writing.
