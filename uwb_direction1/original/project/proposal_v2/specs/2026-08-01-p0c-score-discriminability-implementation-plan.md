# P0c-a Score Discriminability Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现获批的 P0c-a-v1 纯合成未来轨迹评分合同与判别力门，证明每个评分能检出自己负责的已知预测错误，同时保留“评分语义通过不等于轨迹可预测性通过”的研究边界。

**Architecture:** 在现有 P0a/P0b 只读合同之上新增一个纯 NumPy 评分模块。模块分成六层：只读评分记录与 provenance、位置样本到三视角距离的最小桥、FRNLL/range CRPS/ES/VS/coverage–sharpness 纯函数、冻结联合真分布与六个错误/保护臂、伪被试配对聚合与精确符号翻转、只输出完整 JSON 证据的纯合成门。测试和 CLI 分离；P0c-a 不连接 O1、真实回波、模型训练或真实模型排名。

**Tech Stack:** Python 3.9.23、NumPy 2.0.2、标准库 `argparse/dataclasses/functools/hashlib/itertools/json/math/struct/unittest/pathlib/subprocess`；CPU only；不使用 SciPy、PyTorch、文件缓存或 GPU。

**Plan status:** 已批准设计已转写为可执行 TDD 计划；实现尚未开始。

---

## 执行位置与不可越界条件

**实现 worktree：**

```text
D:\uwb_gestures\.worktrees\p0a-causal-prefix
```

**设计与计划文档位置：**

```text
D:\uwb_gestures\proposal_v2
```

**每个 Python 步骤固定解释器：**

```powershell
& "D:\D_conda\envs\pytorch\python.exe"
```

**用户指定的 Git 例外：** 本计划全程不 `git add`、不 commit、不 merge。P0a/P0b 六个文件当前是有意保留的未跟踪成果；每个任务用只读状态与 SHA-256 检查代替提交点。

**P0c-a 只允许新增：**

```text
uwb_osr/forecast_scores.py
tests/test_forecast_scores.py
tools/w4_score_gate.py
```

**必须保持字节不变的 P0a/P0b 文件与基线 SHA-256：**

```text
44C7CADB41596D5044EABAE17E47E55D9342AB65FDF104664FC9A11C7BC31676  uwb_osr/causal_data.py
DC6B024481CF8CDD8748F3048FFF3ED2607671F2EBDE9EAC62592296B8AD828E  tests/test_causal_forecast_protocol.py
8BA5B354D4833C9EA559E74F7B45CE553AD337FA238C01CC7D94F051BBB27AE4  tools/w4_causal_gate.py
2EC5CF3CCBAA651A731F82C8BF9613AB74F7A4A2A8DDA6617C3F0AF43379485D  uwb_osr/causal_baselines.py
C67FBB681242F30F9DC79E82366B9995D821BBD681FB1AF9C283F728CD3B3751  tests/test_causal_baselines.py
B995D067600F65F0FD1FB7FF1EB431D51B614E1B5ECD9AE74CE3CB7FEB921547  tools/w4_kinematic_gate.py
```

**本计划禁止：**

- 读取 `cache/`、`logs/` 或真实信号/标签；
- 运行 O1、W1、W3、`--track`、`--analyze`；
- 创建 `logs/w3_readout_scores.npz` 或任何真实评分产物；
- 修改 17、18、`tools/w3_readout_predict.py` 或 P0a/P0b 六个实现/测试/CLI 文件；
- 用合成 PASS 宣称真实手势可预测、某模型胜出或真实 5% 改善有功效；
- 暂存、提交或合并。

历史打乱的速度层只继承 P0b 已冻结的公开常量 `DT=0.05 s`。该量只决定匹配层，不进入 FRNLL、CRPS、ES、VS 或覆盖率公式。

---

## 文件职责图

| 文件 | 动作 | 单一职责 |
|---|---|---|
| `uwb_osr/forecast_scores.py` | 新建 | P0c-a 数据合同、纯评分函数、合成总体、错误臂、配对统计和 manifest |
| `tests/test_forecast_scores.py` | 新建 | 公式手算、合同失败、合成结构、错误臂、统计门、CLI 与静态边界 |
| `tools/w4_score_gate.py` | 新建 | 只接受必选 `--selftest`，输出确定性单行 JSON |
| `proposal_v2/specs/2026-08-01-p0c-score-discriminability-design.md` | 最终证据阶段只改状态 | 只有全部验收通过后才记为 implemented/PASS |
| `proposal_v2/specs/2026-08-01-p0c-score-discriminability-implementation-plan.md` | 执行时勾选 | 保存逐项证据；不写真实轨迹结论 |
| `proposal_v2/15_块二重构_因果状态估计_未来轨迹预测_未来回波生成.md` | 最终证据阶段只补状态 | 明确下一站只能是 P0c-b 书面设计 |

---

### Task 0：重确认获批边界与未污染起点

**Files:**
- Read only: `D:\uwb_gestures\proposal_v2\specs\2026-08-01-p0c-score-discriminability-design.md`
- Read only: 上述 P0a/P0b 六个文件
- Assert absent: 三个 P0c-a 拟新增文件

- [x] **Step 1：确认设计已获书面通过且主线仍是未来轨迹预测**

从文档目录运行：

```powershell
rg -n "已获用户书面复核通过|未来轨迹|P0c-a PASS|不表示|不得读取|GO 后只允许进入 P0c-b" "specs\2026-08-01-p0c-score-discriminability-design.md"
```

Expected: 命中设计状态、未来轨迹主线、合成结论边界和 P0c-b 停止点；没有实现结果表述。

- [x] **Step 2：验证 P0a/P0b 六个哈希与 W3 分数缺失**

从实现 worktree 运行：

```powershell
$expected = @{
  "uwb_osr/causal_data.py" = "44C7CADB41596D5044EABAE17E47E55D9342AB65FDF104664FC9A11C7BC31676"
  "tests/test_causal_forecast_protocol.py" = "DC6B024481CF8CDD8748F3048FFF3ED2607671F2EBDE9EAC62592296B8AD828E"
  "tools/w4_causal_gate.py" = "8BA5B354D4833C9EA559E74F7B45CE553AD337FA238C01CC7D94F051BBB27AE4"
  "uwb_osr/causal_baselines.py" = "2EC5CF3CCBAA651A731F82C8BF9613AB74F7A4A2A8DDA6617C3F0AF43379485D"
  "tests/test_causal_baselines.py" = "C67FBB681242F30F9DC79E82366B9995D821BBD681FB1AF9C283F728CD3B3751"
  "tools/w4_kinematic_gate.py" = "B995D067600F65F0FD1FB7FF1EB431D51B614E1B5ECD9AE74CE3CB7FEB921547"
}
foreach ($path in $expected.Keys) {
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
  if ($actual -ne $expected[$path]) { throw "$path hash changed: $actual" }
}
if (Test-Path -LiteralPath "D:\uwb_gestures\logs\w3_readout_scores.npz") {
  throw "forbidden W3 score exists"
}
```

Expected: exit 0、无输出。

- [x] **Step 3：重跑 P0a/P0b 聚焦回归作为起点**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_forecast_protocol.py" -v
if ($LASTEXITCODE -ne 0) { throw "P0a regression failed" }
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_baselines.py" -v
if ($LASTEXITCODE -ne 0) { throw "P0b regression failed" }
```

Expected: P0a 47/47 PASS；P0b 50/50 PASS；0 failure/error/skip。

- [x] **Step 4：记录未跟踪起点且确认 P0c-a 文件尚不存在**

```powershell
$new = @(
  "uwb_osr/forecast_scores.py",
  "tests/test_forecast_scores.py",
  "tools/w4_score_gate.py"
)
foreach ($path in $new) {
  if (Test-Path -LiteralPath $path) { throw "unexpected pre-existing P0c-a file: $path" }
}
git status --short
git diff --cached --name-only
```

Expected: 只看到既有六个 P0a/P0b 未跟踪文件；staged 列表为空；三个 P0c-a 文件不存在。

---

### Task 1：冻结常量、ScoreKey、只读记录、RNG 与 provenance

**Files:**
- Create: `tests/test_forecast_scores.py`
- Create: `uwb_osr/forecast_scores.py`

- [x] **Step 1：先写合同/RNG 的失败测试**

用 `apply_patch` 新建 `tests/test_forecast_scores.py`，起始内容固定为：

```python
from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import struct
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import numpy as np

from uwb_osr.causal_data import ProtocolError
import uwb_osr.forecast_scores as fs


SHA = "0" * 64


def one_cell_observation(
    likelihoods: dict[int, float],
    *,
    horizon: int = 5,
    provenance: str = SHA,
) -> fs.ObservedRangeMass:
    mass = np.zeros((3, horizon, fs.N_BINS), dtype=np.float64)
    for index, value in likelihoods.items():
        mass[0, 0, index] = value
    mask = np.zeros((3, horizon), dtype=np.bool_)
    mask[0, 0] = True
    return fs.ObservedRangeMass(mass, mask, fs.BIN_M, provenance)


def forecast_from_bins(
    bins: np.ndarray,
    *,
    horizon: int = 5,
    provenance: str = SHA,
) -> fs.ForecastRangeSamples:
    values = np.zeros((len(bins), 3, horizon), dtype=np.float64)
    values[:, 0, 0] = np.asarray(bins, dtype=np.float64) * fs.BIN_M
    mask = np.zeros((3, horizon), dtype=np.bool_)
    mask[0, 0] = True
    return fs.ForecastRangeSamples(values, mask, provenance)


class ContractAndSeedTest(unittest.TestCase):
    def test_constants_are_exact(self):
        self.assertEqual(fs.PROTOCOL_VERSION, "P0c-a-v1")
        self.assertEqual(fs.N_VIEWS, 3)
        self.assertEqual(fs.N_BINS, 189)
        self.assertEqual(fs.RANGE_WINDOW_M, 1.2)
        self.assertEqual(fs.BIN_M, 1.2 / 189)
        self.assertEqual(fs.HORIZONS, (5, 10, 20))
        self.assertEqual(fs.PREFIXES, (20, 30, 45, 60))
        self.assertEqual(fs.K_GATE, 20)
        self.assertEqual(fs.EPSILON, 1e-12)
        self.assertEqual(fs.PSEUDO_SUBJECTS, 8)
        self.assertEqual(fs.SEQUENCES_PER_SUBJECT, 240)
        self.assertEqual(fs.DT, 0.05)

    def test_public_record_fields_are_exact(self):
        self.assertEqual(
            tuple(field.name for field in fields(fs.ScoreKey)),
            ("pseudo_subject", "sequence", "prefix_t", "horizon"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(fs.ObservedRangeMass)),
            ("mass", "valid_mask", "bin_m", "provenance"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(fs.ForecastRangeSamples)),
            ("range_m", "valid_mask", "forecast_provenance"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(fs.SyntheticTruth)),
            ("range_m", "mode", "provenance"),
        )

    def test_score_key_is_frozen_and_fail_closed(self):
        key = fs.ScoreKey(0, 0, 20, 5)
        with self.assertRaises(FrozenInstanceError):
            key.sequence = 1
        for values in (
            (-1, 0, 20, 5),
            (8, 0, 20, 5),
            (0, -1, 20, 5),
            (0, 240, 20, 5),
            (0, 0, 21, 5),
            (0, 0, 20, 6),
        ):
            with self.assertRaises(ProtocolError):
                fs.ScoreKey(*values)

    def test_seed_bytes_match_independent_manual_serialization(self):
        key = fs.ScoreKey(3, 17, 45, 10)
        purpose = "forecast_base"
        sample_index = 7
        payload = bytearray(fs.PROTOCOL_VERSION.encode("utf-8"))
        for value in (3, 17, 45, 10):
            payload.extend(struct.pack("<q", value))
        encoded = purpose.encode("utf-8")
        payload.extend(struct.pack("<Q", len(encoded)))
        payload.extend(encoded)
        payload.extend(struct.pack("<q", sample_index))
        expected = int.from_bytes(
            hashlib.sha256(payload).digest()[:8], "little", signed=False
        )
        self.assertEqual(fs.score_seed(key, purpose, sample_index), expected)

    def test_seed_purpose_and_sample_streams_are_separate(self):
        key = fs.ScoreKey(2, 4, 30, 20)
        seeds = {
            fs.score_seed(key, purpose, sample)
            for purpose in ("truth", "forecast_base", "permutation")
            for sample in (0, 1)
        }
        self.assertEqual(len(seeds), 6)
        first = fs.score_rng(key, "truth", 0).standard_normal(8)
        second = fs.score_rng(key, "truth", 0).standard_normal(8)
        np.testing.assert_array_equal(first, second)

    def test_records_own_readonly_float64_and_bool_arrays(self):
        mass = np.zeros((3, 5, fs.N_BINS), dtype=np.float64)
        mask = np.zeros((3, 5), dtype=np.bool_)
        mass[0, 0, 0] = 1.0
        mask[0, 0] = True
        observation = fs.ObservedRangeMass(mass, mask, fs.BIN_M, SHA)
        forecast_values = np.zeros((2, 3, 5), dtype=np.float64)
        forecast = fs.ForecastRangeSamples(forecast_values, mask, SHA)
        truth = fs.SyntheticTruth(
            np.zeros((3, 5), dtype=np.float64), -1, SHA
        )
        for array in (
            observation.mass,
            observation.valid_mask,
            forecast.range_m,
            forecast.valid_mask,
            truth.range_m,
        ):
            self.assertTrue(array.flags.owndata)
            self.assertFalse(array.flags.writeable)
        mass[0, 0, 0] = 0.0
        forecast_values[0, 0, 0] = 9.0
        self.assertEqual(observation.mass[0, 0, 0], 1.0)
        self.assertEqual(forecast.range_m[0, 0, 0], 0.0)

    def test_observation_rejects_every_registered_contract_error(self):
        mass = np.zeros((3, 5, fs.N_BINS), dtype=np.float64)
        mask = np.zeros((3, 5), dtype=np.bool_)
        mass[0, 0, 0] = 1.0
        mask[0, 0] = True
        bad_inputs = []
        bad_inputs.append((mass.astype(np.float32), mask, fs.BIN_M, SHA))
        bad_inputs.append((mass, mask.astype(np.uint8), fs.BIN_M, SHA))
        bad_inputs.append((mass[:, :, :-1], mask, fs.BIN_M, SHA))
        bad_inputs.append((mass, np.zeros_like(mask), fs.BIN_M, SHA))
        unnormalized = mass.copy()
        unnormalized[0, 0, 0] = 0.5
        bad_inputs.append((unnormalized, mask, fs.BIN_M, SHA))
        negative = mass.copy()
        negative[0, 0, 1] = -0.1
        negative[0, 0, 0] = 1.1
        bad_inputs.append((negative, mask, fs.BIN_M, SHA))
        nonfinite = mass.copy()
        nonfinite[0, 0, 0] = np.nan
        bad_inputs.append((nonfinite, mask, fs.BIN_M, SHA))
        invalid_nonzero = mass.copy()
        invalid_nonzero[1, 0, 0] = 1.0
        bad_inputs.append((invalid_nonzero, mask, fs.BIN_M, SHA))
        bad_inputs.append((mass, mask, fs.BIN_M * 2.0, SHA))
        bad_inputs.append((mass, mask, fs.BIN_M, "not-a-sha"))
        for values in bad_inputs:
            with self.subTest(values=values[2:]):
                with self.assertRaises(ProtocolError):
                    fs.ObservedRangeMass(*values)

    def test_forecast_and_truth_reject_dtype_shape_value_and_provenance_errors(self):
        mask = np.ones((3, 5), dtype=np.bool_)
        good = np.zeros((2, 3, 5), dtype=np.float64)
        for values, bad_mask, provenance in (
            (good.astype(np.float32), mask, SHA),
            (good[:, :2], mask, SHA),
            (good, mask.astype(np.uint8), SHA),
            (good, mask[:, :4], SHA),
            (good, mask, "bad"),
        ):
            with self.assertRaises(ProtocolError):
                fs.ForecastRangeSamples(values, bad_mask, provenance)
        for bad in (-1.0, np.nan, np.inf):
            values = good.copy()
            values[0, 0, 0] = bad
            with self.assertRaises(ProtocolError):
                fs.ForecastRangeSamples(values, mask, SHA)
        with self.assertRaises(ProtocolError):
            fs.SyntheticTruth(
                np.zeros((3, 6), dtype=np.float64), -1, SHA
            )
        with self.assertRaises(ProtocolError):
            fs.SyntheticTruth(
                np.zeros((3, 5), dtype=np.float32), -1, SHA
            )
        with self.assertRaises(ProtocolError):
            fs.SyntheticTruth(
                np.zeros((3, 5), dtype=np.float64), 0, "bad"
            )
```

- [x] **Step 2：运行并观察导入失败**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -v
```

Expected: FAIL，根因为 `ModuleNotFoundError: No module named 'uwb_osr.forecast_scores'`；不能先出现别的测试错误。

- [x] **Step 3：实现常量、严格只读记录和 SHA256→PCG64 子流**

用 `apply_patch` 新建 `uwb_osr/forecast_scores.py`：

```python
"""P0c-a-v1 synthetic-only future-trajectory score contracts."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Optional, Sequence, Union

import numpy as np

from uwb_osr.causal_baselines import DT
from uwb_osr.causal_data import ProtocolError


PROTOCOL_VERSION = "P0c-a-v1"
N_VIEWS = 3
N_BINS = 189
RANGE_WINDOW_M = 1.2
BIN_M = RANGE_WINDOW_M / N_BINS
HORIZONS = (5, 10, 20)
PREFIXES = (20, 30, 45, 60)
K_GATE = 20
EPSILON = 1e-12
PSEUDO_SUBJECTS = 8
SEQUENCES_PER_SUBJECT = 240
PREFIX_SCALE = {20: 1.25, 30: 1.10, 45: 1.00, 60: 0.90}
VIEW_CORRELATION = np.array(
    [[1.0, 0.45, 0.25], [0.45, 1.0, 0.35], [0.25, 0.35, 1.0]],
    dtype=np.float64,
)
VIEW_CORRELATION.setflags(write=False)
BOUNDARY = "P0c-a score semantics PASS != trajectory predictability PASS"


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _readonly_exact(
    value: np.ndarray,
    dtype: Union[np.dtype, type],
    name: str,
) -> np.ndarray:
    array = np.asarray(value)
    expected = np.dtype(dtype)
    if array.dtype != expected:
        raise ProtocolError(f"{name} must have dtype {expected}")
    out = np.array(array, dtype=expected, copy=True, order="C")
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class ScoreKey:
    __slots__ = ("pseudo_subject", "sequence", "prefix_t", "horizon")

    pseudo_subject: int
    sequence: int
    prefix_t: int
    horizon: int

    def __post_init__(self) -> None:
        values = (
            self.pseudo_subject,
            self.sequence,
            self.prefix_t,
            self.horizon,
        )
        if any(type(value) is not int for value in values):
            raise ProtocolError("ScoreKey fields must be plain integers")
        if self.pseudo_subject not in range(PSEUDO_SUBJECTS):
            raise ProtocolError("unregistered pseudo_subject")
        if self.sequence not in range(SEQUENCES_PER_SUBJECT):
            raise ProtocolError("unregistered sequence")
        if self.prefix_t not in PREFIXES:
            raise ProtocolError("unregistered prefix_t")
        if self.horizon not in HORIZONS:
            raise ProtocolError("unregistered horizon")


def _key_bytes(key: ScoreKey, purpose: str, sample_index: int) -> bytes:
    if not isinstance(key, ScoreKey):
        raise ProtocolError("key must be ScoreKey")
    if not isinstance(purpose, str) or not purpose:
        raise ProtocolError("purpose must be non-empty text")
    if type(sample_index) is not int:
        raise ProtocolError("sample_index must be a plain integer")
    payload = bytearray(PROTOCOL_VERSION.encode("utf-8"))
    for value in (
        key.pseudo_subject,
        key.sequence,
        key.prefix_t,
        key.horizon,
    ):
        payload.extend(struct.pack("<q", value))
    purpose_bytes = purpose.encode("utf-8")
    payload.extend(struct.pack("<Q", len(purpose_bytes)))
    payload.extend(purpose_bytes)
    payload.extend(struct.pack("<q", sample_index))
    return bytes(payload)


def score_seed(key: ScoreKey, purpose: str, sample_index: int = 0) -> int:
    digest = hashlib.sha256(_key_bytes(key, purpose, sample_index)).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def score_rng(
    key: ScoreKey,
    purpose: str,
    sample_index: int = 0,
) -> np.random.Generator:
    return np.random.Generator(
        np.random.PCG64(score_seed(key, purpose, sample_index))
    )


def case_provenance(key: ScoreKey) -> str:
    return hashlib.sha256(_key_bytes(key, "case_provenance", 0)).hexdigest()


@dataclass(frozen=True)
class ObservedRangeMass:
    __slots__ = ("mass", "valid_mask", "bin_m", "provenance")

    mass: np.ndarray
    valid_mask: np.ndarray
    bin_m: float
    provenance: str

    def __post_init__(self) -> None:
        mass = _readonly_exact(self.mass, np.float64, "mass")
        mask = _readonly_exact(self.valid_mask, np.bool_, "valid_mask")
        if mass.ndim != 3 or mass.shape[0] != N_VIEWS:
            raise ProtocolError("mass must have shape (3,H,189)")
        horizon = mass.shape[1]
        if horizon not in HORIZONS or mass.shape[2] != N_BINS:
            raise ProtocolError("mass has an unregistered H or bin count")
        if mask.shape != (N_VIEWS, horizon):
            raise ProtocolError("valid_mask must have shape (3,H)")
        if not mask.any():
            raise ProtocolError("at least one observation cell must be valid")
        if not np.isfinite(mass).all() or np.any(mass < 0.0):
            raise ProtocolError("mass must be finite and nonnegative")
        if np.any(mass[~mask] != 0.0):
            raise ProtocolError("invalid observation cells must be all zero")
        sums = np.sum(mass[mask], axis=-1)
        if not np.allclose(sums, 1.0, atol=1e-12, rtol=0.0):
            raise ProtocolError("valid observation mass must sum to one")
        if type(self.bin_m) not in (float, int) or not np.isfinite(self.bin_m):
            raise ProtocolError("bin_m must be finite")
        if float(self.bin_m) != BIN_M:
            raise ProtocolError("bin_m differs from the registered grid")
        if not _is_sha256(self.provenance):
            raise ProtocolError("provenance must be a SHA-256 hex string")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "bin_m", float(self.bin_m))


@dataclass(frozen=True)
class ForecastRangeSamples:
    __slots__ = ("range_m", "valid_mask", "forecast_provenance")

    range_m: np.ndarray
    valid_mask: np.ndarray
    forecast_provenance: str

    def __post_init__(self) -> None:
        ranges = _readonly_exact(self.range_m, np.float64, "range_m")
        mask = _readonly_exact(self.valid_mask, np.bool_, "valid_mask")
        if ranges.ndim != 3 or ranges.shape[0] < 1:
            raise ProtocolError("range_m must have shape (K,3,H), K>=1")
        if ranges.shape[1] != N_VIEWS or ranges.shape[2] not in HORIZONS:
            raise ProtocolError("range_m has an unregistered view/H shape")
        if mask.shape != ranges.shape[1:]:
            raise ProtocolError("forecast valid_mask must have shape (3,H)")
        if not mask.any():
            raise ProtocolError("at least one forecast cell must be valid")
        if not np.isfinite(ranges).all() or np.any(ranges < 0.0):
            raise ProtocolError("forecast ranges must be finite and nonnegative")
        if not _is_sha256(self.forecast_provenance):
            raise ProtocolError(
                "forecast_provenance must be a SHA-256 hex string"
            )
        object.__setattr__(self, "range_m", ranges)
        object.__setattr__(self, "valid_mask", mask)


@dataclass(frozen=True)
class SyntheticTruth:
    __slots__ = ("range_m", "mode", "provenance")

    range_m: np.ndarray
    mode: int
    provenance: str

    def __post_init__(self) -> None:
        ranges = _readonly_exact(self.range_m, np.float64, "truth range_m")
        if ranges.shape[0] != N_VIEWS or ranges.ndim != 2:
            raise ProtocolError("truth range_m must have shape (3,H)")
        if ranges.shape[1] not in HORIZONS:
            raise ProtocolError("truth has an unregistered horizon")
        if not np.isfinite(ranges).all() or np.any(ranges < 0.0):
            raise ProtocolError("truth ranges must be finite and nonnegative")
        if type(self.mode) is not int or self.mode not in (-1, 1):
            raise ProtocolError("truth mode must be -1 or +1")
        if not _is_sha256(self.provenance):
            raise ProtocolError("truth provenance must be SHA-256")
        object.__setattr__(self, "range_m", ranges)
```

这里把 observation、forecast 和 truth 的 provenance 明确解释为**同一 ScoreKey 的配对身份**；错误臂名称不进入 scorer，也不改变 provenance。完整合成配置另在 Task 5 生成独立 `config_provenance`。

- [x] **Step 4：运行合同测试并得到绿灯**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -v
```

Expected: 当前 `ContractAndSeedTest` 全部 PASS，0 failure/error/skip。

- [x] **Step 5：记录只读状态，不暂存**

```powershell
git status --short -- uwb_osr/forecast_scores.py tests/test_forecast_scores.py
git diff --cached --name-only
```

Expected: 两个新文件显示 `??`；staged 输出为空。

---

### Task 2：实现 P0b 位置样本到三视角距离的最小桥

**Files:**
- Modify: `tests/test_forecast_scores.py`
- Modify: `uwb_osr/forecast_scores.py`

- [x] **Step 1：追加几何桥失败测试**

在测试文件追加：

```python
class GeometryBridgeTest(unittest.TestCase):
    def setUp(self):
        self.radar = np.array(
            [[-0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.0, 0.0]],
            dtype=np.float64,
        )
        self.mask = np.ones((3, 5), dtype=np.bool_)

    def test_known_points_preserve_k_view_h_axes(self):
        position = np.array(
            [
                [[0.0, 0.0, 0.5]] * 5,
                [[0.1, -0.2, 0.4]] * 5,
            ],
            dtype=np.float64,
        )
        out = fs.project_position_samples(position, self.radar, self.mask, SHA)
        expected = np.linalg.norm(
            position[:, None, :, :] - self.radar[None, :, None, :], axis=-1
        )
        self.assertEqual(out.range_m.shape, (2, 3, 5))
        np.testing.assert_allclose(out.range_m, expected, atol=0.0, rtol=0.0)
        self.assertEqual(out.forecast_provenance, SHA)
        self.assertFalse(out.range_m.flags.writeable)

    def test_bridge_rejects_shape_dtype_nonfinite_and_mask_errors(self):
        good = np.zeros((2, 5, 3), dtype=np.float64)
        cases = (
            (good.astype(np.float32), self.radar, self.mask, SHA),
            (good[:, :, :2], self.radar, self.mask, SHA),
            (good, self.radar.astype(np.float32), self.mask, SHA),
            (good, self.radar[:2], self.mask, SHA),
            (good, self.radar, self.mask.astype(np.uint8), SHA),
            (good, self.radar, self.mask[:, :4], SHA),
            (good, self.radar, self.mask, "bad"),
        )
        for values in cases:
            with self.subTest(shapes=tuple(np.shape(v) for v in values[:3])):
                with self.assertRaises(ProtocolError):
                    fs.project_position_samples(*values)
        nonfinite = good.copy()
        nonfinite[0, 0, 0] = np.nan
        with self.assertRaises(ProtocolError):
            fs.project_position_samples(nonfinite, self.radar, self.mask, SHA)
```

- [x] **Step 2：运行并确认只因接口缺失而失败**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k GeometryBridgeTest -v
```

Expected: FAIL，错误为 `AttributeError` 指向 `project_position_samples`。

- [x] **Step 3：追加纯几何实现**

在生产模块 `SyntheticTruth` 后追加：

```python
def project_position_samples(
    position_m: np.ndarray,
    radar_positions_m: np.ndarray,
    valid_mask: np.ndarray,
    provenance: str,
) -> ForecastRangeSamples:
    positions = np.asarray(position_m)
    radars = np.asarray(radar_positions_m)
    mask = np.asarray(valid_mask)
    if positions.dtype != np.dtype(np.float64):
        raise ProtocolError("position_m must be float64")
    if radars.dtype != np.dtype(np.float64):
        raise ProtocolError("radar_positions_m must be float64")
    if mask.dtype != np.dtype(np.bool_):
        raise ProtocolError("valid_mask must be bool")
    if positions.ndim != 3 or positions.shape[0] < 1:
        raise ProtocolError("position_m must have shape (K,H,3)")
    if positions.shape[1] not in HORIZONS or positions.shape[2] != 3:
        raise ProtocolError("position_m has an unregistered H/coordinate shape")
    if radars.shape != (N_VIEWS, 3):
        raise ProtocolError("radar_positions_m must have shape (3,3)")
    if mask.shape != (N_VIEWS, positions.shape[1]):
        raise ProtocolError("valid_mask must have shape (3,H)")
    if not np.isfinite(positions).all() or not np.isfinite(radars).all():
        raise ProtocolError("geometry bridge inputs must be finite")
    if not _is_sha256(provenance):
        raise ProtocolError("geometry provenance must be SHA-256")
    ranges = np.linalg.norm(
        positions[:, None, :, :] - radars[None, :, None, :], axis=-1
    )
    return ForecastRangeSamples(ranges, mask, provenance)
```

- [x] **Step 4：重跑聚焦与累计测试**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k GeometryBridgeTest -v
if ($LASTEXITCODE -ne 0) { throw "geometry tests failed" }
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -v
```

Expected: GeometryBridgeTest 和此前合同/RNG 测试全部 PASS；0 skip。

---
### Task 3：实现 FRLS/FRNLL 的连续网格似然与 log-after-average

**Files:**
- Modify: `tests/test_forecast_scores.py`
- Modify: `uwb_osr/forecast_scores.py`

- [x] **Step 1：先追加手算与边界失败测试**

在测试文件追加：

```python
class FrnllTest(unittest.TestCase):
    def test_log_is_taken_after_sample_likelihood_average(self):
        observation = one_cell_observation({0: 0.25, 1: 0.75})
        forecast = forecast_from_bins(np.array([0.0, 1.0]))
        likelihood = fs.range_likelihood(observation, forecast)
        self.assertEqual(likelihood.shape, (2, 3, 5))
        self.assertAlmostEqual(likelihood[0, 0, 0], 0.25, places=15)
        self.assertAlmostEqual(likelihood[1, 0, 0], 0.75, places=15)
        frls, frnll = fs.frls_frnll(observation, forecast)
        expected = float(np.log(0.5))
        forbidden = float(np.mean(np.log([0.25, 0.75])))
        self.assertAlmostEqual(frls, expected, places=15)
        self.assertAlmostEqual(frnll, -expected, places=15)
        self.assertNotAlmostEqual(frls, forbidden, places=6)

    def test_linear_interpolation_and_bin_188_tail_are_exact(self):
        interpolated = one_cell_observation({10: 0.2, 11: 0.8})
        value = fs.range_likelihood(
            interpolated, forecast_from_bins(np.array([10.25]))
        )[0, 0, 0]
        self.assertAlmostEqual(value, 0.35, places=15)

        upper = one_cell_observation({188: 1.0})
        likelihood = fs.range_likelihood(
            upper, forecast_from_bins(np.array([188.0, 188.5, 189.0]))
        )[:, 0, 0]
        np.testing.assert_array_equal(likelihood, np.array([1.0, 1.0, 0.0]))

    def test_out_of_grid_forecast_is_kept_and_receives_epsilon(self):
        observation = one_cell_observation({188: 1.0})
        forecast = forecast_from_bins(np.array([250.0]))
        frls, frnll = fs.frls_frnll(observation, forecast)
        self.assertEqual(frls, math.log(fs.EPSILON))
        self.assertEqual(frnll, -math.log(fs.EPSILON))
        self.assertTrue(forecast.valid_mask[0, 0])

    def test_pair_must_match_horizon_mask_and_case_provenance(self):
        observation = one_cell_observation({0: 1.0})
        wrong_mask = np.zeros((3, 5), dtype=np.bool_)
        wrong_mask[0, 1] = True
        wrong_mask_forecast = fs.ForecastRangeSamples(
            np.zeros((2, 3, 5), dtype=np.float64), wrong_mask, SHA
        )
        wrong_provenance = forecast_from_bins(
            np.array([0.0]), provenance="1" * 64
        )
        wrong_horizon = fs.ForecastRangeSamples(
            np.zeros((2, 3, 10), dtype=np.float64),
            np.ones((3, 10), dtype=np.bool_),
            SHA,
        )
        for forecast in (
            wrong_mask_forecast,
            wrong_provenance,
            wrong_horizon,
        ):
            with self.assertRaises(ProtocolError):
                fs.frls_frnll(observation, forecast)

    def test_public_scorer_signature_has_no_model_or_label_switch(self):
        parameters = inspect.signature(fs.frls_frnll).parameters
        self.assertEqual(tuple(parameters), ("observation", "forecast"))
```

在测试文件 imports 中补入：

```python
import math
```

- [x] **Step 2：运行聚焦测试并确认函数缺失**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k FrnllTest -v
```

Expected: FAIL，缺少 `range_likelihood` / `frls_frnll`；已有合同测试不回退。

- [x] **Step 3：实现唯一配对校验、线性插值和 FRLS/FRNLL**

在生产模块追加：

```python
def _validate_score_pair(
    observation: ObservedRangeMass,
    forecast: ForecastRangeSamples,
) -> None:
    if not isinstance(observation, ObservedRangeMass):
        raise ProtocolError("observation must be ObservedRangeMass")
    if not isinstance(forecast, ForecastRangeSamples):
        raise ProtocolError("forecast must be ForecastRangeSamples")
    if observation.mass.shape[1] != forecast.range_m.shape[2]:
        raise ProtocolError("observation and forecast horizons differ")
    if not np.array_equal(observation.valid_mask, forecast.valid_mask):
        raise ProtocolError("observation and forecast masks differ")
    if observation.provenance != forecast.forecast_provenance:
        raise ProtocolError("observation and forecast provenance differ")


def range_likelihood(
    observation: ObservedRangeMass,
    forecast: ForecastRangeSamples,
) -> np.ndarray:
    _validate_score_pair(observation, forecast)
    likelihood = np.zeros_like(forecast.range_m, dtype=np.float64)
    for view, step in np.argwhere(observation.valid_mask):
        scaled = forecast.range_m[:, view, step] / BIN_M
        inside = scaled <= (N_BINS - 1)
        if np.any(inside):
            coordinates = scaled[inside]
            lower = np.floor(coordinates).astype(np.int64)
            upper = np.minimum(lower + 1, N_BINS - 1)
            fraction = coordinates - lower
            cell = observation.mass[view, step]
            likelihood[inside, view, step] = (
                (1.0 - fraction) * cell[lower]
                + fraction * cell[upper]
            )
        last_bin_tail = (scaled > (N_BINS - 1)) & (scaled < N_BINS)
        likelihood[last_bin_tail, view, step] = observation.mass[
            view, step, N_BINS - 1
        ]
    likelihood.setflags(write=False)
    return likelihood


def frls_frnll(
    observation: ObservedRangeMass,
    forecast: ForecastRangeSamples,
) -> tuple[float, float]:
    likelihood = range_likelihood(observation, forecast)
    sample_average = np.mean(likelihood, axis=0)
    cell_log = np.log(
        np.maximum(EPSILON, sample_average[observation.valid_mask])
    )
    frls = float(np.mean(cell_log))
    if not np.isfinite(frls):
        raise ProtocolError("FRLS is non-finite")
    return frls, -frls
```

- [x] **Step 4：重跑聚焦和累计测试**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k FrnllTest -v
if ($LASTEXITCODE -ne 0) { throw "FRNLL tests failed" }
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -v
```

Expected: 手算值、188/189 边界、epsilon 惩罚和配对失败语义全部 PASS；0 skip。

---

### Task 4：实现期望 range CRPS、Energy/Variogram 与 coverage–sharpness

**Files:**
- Modify: `tests/test_forecast_scores.py`
- Modify: `uwb_osr/forecast_scores.py`

- [x] **Step 1：追加四类评分的精确手算测试**

在测试文件追加：

```python
class OtherScoresTest(unittest.TestCase):
    def test_expected_range_crps_matches_double_sum(self):
        observation = one_cell_observation({0: 0.5, 2: 0.5})
        forecast = forecast_from_bins(np.array([0.0, 2.0]))
        expected = 0.5 * fs.BIN_M
        self.assertAlmostEqual(
            fs.expected_range_crps(observation, forecast),
            expected,
            places=15,
        )

    def test_energy_score_known_vector_and_view_major_order(self):
        mask = np.zeros((3, 5), dtype=np.bool_)
        mask[0, :2] = True
        samples = np.zeros((1, 3, 5), dtype=np.float64)
        truth_values = np.zeros((3, 5), dtype=np.float64)
        truth_values[0, :2] = (3.0, 4.0)
        forecast = fs.ForecastRangeSamples(samples, mask, SHA)
        truth = fs.SyntheticTruth(truth_values, -1, SHA)
        self.assertEqual(fs.energy_score(forecast, truth), 5.0)

    def test_variogram_score_known_two_dimension_case(self):
        mask = np.zeros((3, 5), dtype=np.bool_)
        mask[0, :2] = True
        samples = np.zeros((1, 3, 5), dtype=np.float64)
        truth_values = np.zeros((3, 5), dtype=np.float64)
        truth_values[0, :2] = (0.0, 4.0)
        forecast = fs.ForecastRangeSamples(samples, mask, SHA)
        truth = fs.SyntheticTruth(truth_values, 1, SHA)
        self.assertEqual(fs.variogram_score(forecast, truth), 4.0)

    def test_coverage_and_width_use_linear_central_quantiles(self):
        mask = np.zeros((3, 5), dtype=np.bool_)
        mask[0, 0] = True
        samples = np.zeros((4, 3, 5), dtype=np.float64)
        samples[:, 0, 0] = (0.0, 1.0, 2.0, 3.0)
        truth_values = np.zeros((3, 5), dtype=np.float64)
        truth_values[0, 0] = 1.5
        forecast = fs.ForecastRangeSamples(samples, mask, SHA)
        truth = fs.SyntheticTruth(truth_values, -1, SHA)
        result = fs.coverage_sharpness(forecast, truth)
        np.testing.assert_array_equal(result.levels, np.array([0.5, 0.8, 0.9]))
        self.assertEqual(result.coverage[0], 1.0)
        self.assertEqual(result.width_m[0], 1.5)
        for array in (result.levels, result.coverage, result.width_m):
            self.assertFalse(array.flags.writeable)

    def test_joint_scores_require_matching_truth_and_two_valid_dimensions(self):
        mask = np.zeros((3, 5), dtype=np.bool_)
        mask[0, 0] = True
        forecast = fs.ForecastRangeSamples(
            np.zeros((2, 3, 5), dtype=np.float64), mask, SHA
        )
        wrong_truth = fs.SyntheticTruth(
            np.zeros((3, 5), dtype=np.float64), -1, "1" * 64
        )
        with self.assertRaises(ProtocolError):
            fs.energy_score(forecast, wrong_truth)
        with self.assertRaises(ProtocolError):
            fs.variogram_score(
                forecast,
                fs.SyntheticTruth(
                    np.zeros((3, 5), dtype=np.float64), -1, SHA
                ),
            )
```

- [x] **Step 2：运行并确认四个接口缺失**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k OtherScoresTest -v
```

Expected: FAIL，指向尚未实现的 CRPS/ES/VS/coverage 接口。

- [x] **Step 3：实现四类纯评分和只读覆盖记录**

在生产模块追加：

```python
@dataclass(frozen=True)
class CoverageSharpness:
    __slots__ = ("levels", "coverage", "width_m")

    levels: np.ndarray
    coverage: np.ndarray
    width_m: np.ndarray

    def __post_init__(self) -> None:
        levels = _readonly_exact(self.levels, np.float64, "levels")
        coverage = _readonly_exact(self.coverage, np.float64, "coverage")
        width = _readonly_exact(self.width_m, np.float64, "width_m")
        if levels.shape != (3,) or coverage.shape != (3,) or width.shape != (3,):
            raise ProtocolError("coverage-sharpness arrays must have shape (3,)")
        if not np.array_equal(levels, np.array([0.5, 0.8, 0.9])):
            raise ProtocolError("coverage levels must be 50/80/90 percent")
        if not np.isfinite(coverage).all() or np.any((coverage < 0) | (coverage > 1)):
            raise ProtocolError("coverage must be finite in [0,1]")
        if not np.isfinite(width).all() or np.any(width < 0):
            raise ProtocolError("sharpness width must be finite and nonnegative")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "width_m", width)


def _validate_truth_pair(
    forecast: ForecastRangeSamples,
    truth: SyntheticTruth,
) -> np.ndarray:
    if not isinstance(forecast, ForecastRangeSamples):
        raise ProtocolError("forecast must be ForecastRangeSamples")
    if not isinstance(truth, SyntheticTruth):
        raise ProtocolError("joint truth is required")
    if forecast.range_m.shape[1:] != truth.range_m.shape:
        raise ProtocolError("forecast and truth shapes differ")
    if forecast.forecast_provenance != truth.provenance:
        raise ProtocolError("forecast and truth provenance differ")
    return forecast.valid_mask


def expected_range_crps(
    observation: ObservedRangeMass,
    forecast: ForecastRangeSamples,
) -> float:
    _validate_score_pair(observation, forecast)
    grid = np.arange(N_BINS, dtype=np.float64) * BIN_M
    values = []
    for view, step in np.argwhere(observation.valid_mask):
        samples = forecast.range_m[:, view, step]
        target_term = np.mean(
            np.sum(
                observation.mass[view, step][None, :]
                * np.abs(samples[:, None] - grid[None, :]),
                axis=1,
            )
        )
        sample_term = 0.5 * np.mean(
            np.abs(samples[:, None] - samples[None, :])
        )
        values.append(float(target_term - sample_term))
    result = float(np.mean(values))
    if not np.isfinite(result):
        raise ProtocolError("range CRPS is non-finite")
    return result


def energy_score(
    forecast: ForecastRangeSamples,
    truth: SyntheticTruth,
) -> float:
    mask = _validate_truth_pair(forecast, truth)
    samples = forecast.range_m[:, mask]
    target = truth.range_m[mask]
    first = np.mean(np.linalg.norm(samples - target[None, :], axis=1))
    second = 0.5 * np.mean(
        np.linalg.norm(samples[:, None, :] - samples[None, :, :], axis=2)
    )
    result = float(first - second)
    if not np.isfinite(result):
        raise ProtocolError("Energy Score is non-finite")
    return result


def variogram_score(
    forecast: ForecastRangeSamples,
    truth: SyntheticTruth,
    p: float = 0.5,
) -> float:
    if p != 0.5:
        raise ProtocolError("P0c-a variogram exponent is frozen at 0.5")
    mask = _validate_truth_pair(forecast, truth)
    samples = forecast.range_m[:, mask]
    target = truth.range_m[mask]
    dimensions = target.size
    if dimensions < 2:
        raise ProtocolError("Variogram Score requires at least two dimensions")
    left, right = np.triu_indices(dimensions, k=1)
    target_difference = np.abs(target[left] - target[right]) ** p
    sample_difference = np.mean(
        np.abs(samples[:, left] - samples[:, right]) ** p,
        axis=0,
    )
    result = float(np.mean((target_difference - sample_difference) ** 2))
    if not np.isfinite(result):
        raise ProtocolError("Variogram Score is non-finite")
    return result


def coverage_sharpness(
    forecast: ForecastRangeSamples,
    truth: SyntheticTruth,
) -> CoverageSharpness:
    mask = _validate_truth_pair(forecast, truth)
    levels = np.array([0.5, 0.8, 0.9], dtype=np.float64)
    coverage = np.empty(3, dtype=np.float64)
    width = np.empty(3, dtype=np.float64)
    samples = forecast.range_m[:, mask]
    target = truth.range_m[mask]
    for index, level in enumerate(levels):
        tail = (1.0 - level) / 2.0
        lower, upper = np.quantile(
            samples,
            [tail, 1.0 - tail],
            axis=0,
            method="linear",
        )
        coverage[index] = np.mean((target >= lower) & (target <= upper))
        width[index] = np.mean(upper - lower)
    return CoverageSharpness(levels, coverage, width)
```

- [x] **Step 4：重跑聚焦和累计测试**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k OtherScoresTest -v
if ($LASTEXITCODE -ne 0) { throw "other score tests failed" }
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -v
```

Expected: CRPS 双重求和、ES/VS 已知向量、linear quantile 和联合真值失败语义全部 PASS；0 skip。

---

### Task 5：实现冻结联合真分布、正确预测样本与 oracle-mass 替身

**Files:**
- Modify: `tests/test_forecast_scores.py`
- Modify: `uwb_osr/forecast_scores.py`

- [x] **Step 1：先追加合成总体结构测试**

在测试文件追加：

```python
class SyntheticPopulationTest(unittest.TestCase):
    def test_config_provenance_and_case_are_byte_deterministic(self):
        key = fs.ScoreKey(3, 17, 45, 20)
        first = fs.build_synthetic_case(key)
        second = fs.build_synthetic_case(key)
        self.assertEqual(fs.synthetic_config_provenance(), first.config_provenance)
        self.assertEqual(first.config_provenance, second.config_provenance)
        self.assertEqual(len(first.config_provenance), 64)
        for left, right in (
            (first.observation.mass, second.observation.mass),
            (first.truth.range_m, second.truth.range_m),
            (first.correct.range_m, second.correct.range_m),
            (first.base_samples_m, second.base_samples_m),
            (first.mode_minus_m, second.mode_minus_m),
            (first.mode_plus_m, second.mode_plus_m),
        ):
            self.assertEqual(left.tobytes(), right.tobytes())

    def test_four_prefixes_share_conditioning_mean_and_truth_branch(self):
        cases = [
            fs.build_synthetic_case(fs.ScoreKey(1, 9, prefix, 10))
            for prefix in fs.PREFIXES
        ]
        for case in cases[1:]:
            np.testing.assert_array_equal(
                case.conditional_mean_m, cases[0].conditional_mean_m
            )
            self.assertEqual(case.truth.mode, cases[0].truth.mode)

    def test_covariance_is_frozen_view_kronecker_time_ar1(self):
        key = fs.ScoreKey(2, 0, 30, 5)
        cholesky = fs._cholesky_bins(
            key.pseudo_subject, key.prefix_t, key.horizon
        )
        time = 0.85 ** np.abs(
            np.arange(5)[:, None] - np.arange(5)[None, :]
        )
        sigma = (1.70 + 0.10 * 2) * 1.10
        expected = sigma**2 * np.kron(fs.VIEW_CORRELATION, time)
        np.testing.assert_allclose(
            cholesky @ cholesky.T, expected, atol=1e-12, rtol=0.0
        )

    def test_correct_forecast_has_ten_samples_from_each_mode(self):
        case = fs.build_synthetic_case(fs.ScoreKey(0, 0, 20, 5))
        np.testing.assert_array_equal(
            case.correct.range_m[:10], case.mode_minus_m[:10]
        )
        np.testing.assert_array_equal(
            case.correct.range_m[10:], case.mode_plus_m[10:]
        )
        difference = (case.mode_plus_m - case.mode_minus_m) / fs.BIN_M
        np.testing.assert_allclose(difference[:, 0], 8.0, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(difference[:, 1:], 0.0, atol=1e-12, rtol=0.0)

    def test_truth_stream_is_distinct_and_observation_mass_is_normalized(self):
        case = fs.build_synthetic_case(fs.ScoreKey(7, 239, 60, 20))
        self.assertFalse(
            np.any(
                np.all(
                    case.correct.range_m == case.truth.range_m[None, :, :],
                    axis=(1, 2),
                )
            )
        )
        sums = np.sum(case.observation.mass, axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-12, rtol=0.0)
        self.assertTrue(case.observation.valid_mask.all())
        self.assertGreaterEqual(
            float(np.min(case.observation.mass)), 0.02 / fs.N_BINS
        )

    def test_generator_outputs_are_nonnegative_readonly_and_k20(self):
        case = fs.build_synthetic_case(fs.ScoreKey(4, 88, 45, 10))
        self.assertEqual(case.correct.range_m.shape, (20, 3, 10))
        for array in (
            case.correct.range_m,
            case.truth.range_m,
            case.observation.mass,
            case.base_samples_m,
            case.mode_minus_m,
            case.mode_plus_m,
            case.conditional_mean_m,
        ):
            self.assertFalse(array.flags.writeable)
            self.assertTrue(np.isfinite(array).all())
            self.assertTrue(np.all(array >= 0.0))
```

- [x] **Step 2：运行并确认生成接口缺失**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k SyntheticPopulationTest -v
```

Expected: FAIL，缺少 `build_synthetic_case`、`synthetic_config_provenance` 和 `_cholesky_bins`。

- [x] **Step 3：实现配置摘要、合成 case 记录与协方差缓存**

在生产模块追加：

```python
@dataclass(frozen=True)
class SyntheticCase:
    __slots__ = (
        "key",
        "observation",
        "truth",
        "correct",
        "base_samples_m",
        "mode_minus_m",
        "mode_plus_m",
        "conditional_mean_m",
        "config_provenance",
    )

    key: ScoreKey
    observation: ObservedRangeMass
    truth: SyntheticTruth
    correct: ForecastRangeSamples
    base_samples_m: np.ndarray
    mode_minus_m: np.ndarray
    mode_plus_m: np.ndarray
    conditional_mean_m: np.ndarray
    config_provenance: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, ScoreKey):
            raise ProtocolError("synthetic case key is invalid")
        if not isinstance(self.observation, ObservedRangeMass):
            raise ProtocolError("synthetic observation is invalid")
        if not isinstance(self.truth, SyntheticTruth):
            raise ProtocolError("synthetic truth is invalid")
        if not isinstance(self.correct, ForecastRangeSamples):
            raise ProtocolError("correct forecast is invalid")
        expected = (K_GATE, N_VIEWS, self.key.horizon)
        arrays = []
        for name in ("base_samples_m", "mode_minus_m", "mode_plus_m"):
            array = _readonly_exact(getattr(self, name), np.float64, name)
            if array.shape != expected or not np.isfinite(array).all():
                raise ProtocolError(f"{name} has the wrong shape or values")
            if np.any(array < 0.0):
                raise ProtocolError(f"{name} contains negative ranges")
            arrays.append((name, array))
        mean = _readonly_exact(
            self.conditional_mean_m, np.float64, "conditional_mean_m"
        )
        if mean.shape != (N_VIEWS, self.key.horizon):
            raise ProtocolError("conditional_mean_m has the wrong shape")
        provenance = case_provenance(self.key)
        if not (
            self.observation.provenance
            == self.truth.provenance
            == self.correct.forecast_provenance
            == provenance
        ):
            raise ProtocolError("synthetic case pair provenance differs")
        if self.correct.range_m.shape[0] != K_GATE:
            raise ProtocolError("P0c-a correct forecast requires K=20")
        if not _is_sha256(self.config_provenance):
            raise ProtocolError("config_provenance must be SHA-256")
        for name, array in arrays:
            object.__setattr__(self, name, array)
        object.__setattr__(self, "conditional_mean_m", mean)


def _configuration_payload() -> dict[str, object]:
    return {
        "epsilon": EPSILON,
        "horizons": list(HORIZONS),
        "k": K_GATE,
        "n_bins": N_BINS,
        "prefix_scale": [[key, PREFIX_SCALE[key]] for key in PREFIXES],
        "prefixes": list(PREFIXES),
        "protocol": PROTOCOL_VERSION,
        "pseudo_subjects": PSEUDO_SUBJECTS,
        "range_window_m": RANGE_WINDOW_M,
        "sequences_per_subject": SEQUENCES_PER_SUBJECT,
        "time_correlation": 0.85,
        "truth_mass_outlier": 0.02,
        "truth_mass_sigma_bins": 2.0,
        "view_correlation": VIEW_CORRELATION.tolist(),
    }


def synthetic_config_provenance() -> str:
    payload = json.dumps(
        _configuration_payload(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(PROTOCOL_VERSION.encode("utf-8"))
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)
    return digest.hexdigest()


def _base_key(key: ScoreKey) -> ScoreKey:
    return ScoreKey(key.pseudo_subject, key.sequence, 20, 20)


def _reflected_slope(anchor: np.ndarray, slope: np.ndarray) -> np.ndarray:
    endpoint = anchor + 20.0 * slope
    reflected = endpoint.copy()
    reflected = np.where(endpoint < 24.0, 48.0 - endpoint, reflected)
    reflected = np.where(endpoint > 164.0, 328.0 - endpoint, reflected)
    return (reflected - anchor) / 20.0


@lru_cache(maxsize=None)
def _cholesky_bins(
    pseudo_subject: int,
    prefix_t: int,
    horizon: int,
) -> np.ndarray:
    if pseudo_subject not in range(PSEUDO_SUBJECTS):
        raise ProtocolError("unregistered covariance pseudo_subject")
    if prefix_t not in PREFIXES or horizon not in HORIZONS:
        raise ProtocolError("unregistered covariance prefix/H")
    time_index = np.arange(horizon)
    time_correlation = 0.85 ** np.abs(
        time_index[:, None] - time_index[None, :]
    )
    sigma = (1.70 + 0.10 * pseudo_subject) * PREFIX_SCALE[prefix_t]
    covariance = sigma**2 * np.kron(VIEW_CORRELATION, time_correlation)
    cholesky = np.linalg.cholesky(covariance)
    cholesky.setflags(write=False)
    return cholesky


def _conditional_mean_bins(key: ScoreKey) -> np.ndarray:
    base = _base_key(key)
    anchor = score_rng(base, "base_anchor").uniform(48.0, 140.0, N_VIEWS)
    slope = score_rng(base, "base_slope").uniform(-0.6, 0.6, N_VIEWS)
    slope = _reflected_slope(anchor, slope)
    future_step = np.arange(1, key.horizon + 1, dtype=np.float64)
    return anchor[:, None] + slope[:, None] * future_step[None, :]


def _oracle_mass(truth_bins: np.ndarray) -> np.ndarray:
    grid = np.arange(N_BINS, dtype=np.float64)
    distance = grid[None, None, :] - truth_bins[:, :, None]
    gaussian = np.exp(-0.5 * (distance / 2.0) ** 2)
    gaussian /= np.sum(gaussian, axis=-1, keepdims=True)
    return 0.98 * gaussian + 0.02 / N_BINS
```

- [x] **Step 4：实现正确 10/10 双峰样本、独立 truth 与离散质量**

继续在生产模块追加：

```python
def build_synthetic_case(key: ScoreKey) -> SyntheticCase:
    if not isinstance(key, ScoreKey):
        raise ProtocolError("synthetic generator requires ScoreKey")
    provenance = case_provenance(key)
    mean_bins = _conditional_mean_bins(key)
    cholesky = _cholesky_bins(
        key.pseudo_subject, key.prefix_t, key.horizon
    )
    forecast_noise = score_rng(key, "forecast_base").standard_normal(
        (K_GATE, N_VIEWS * key.horizon)
    )
    forecast_noise = (forecast_noise @ cholesky.T).reshape(
        K_GATE, N_VIEWS, key.horizon
    )
    base_bins = mean_bins[None, :, :] + forecast_noise
    minus_bins = base_bins.copy()
    plus_bins = base_bins.copy()
    minus_bins[:, 0, :] -= 4.0
    plus_bins[:, 0, :] += 4.0
    correct_bins = np.empty_like(base_bins)
    correct_bins[:10] = minus_bins[:10]
    correct_bins[10:] = plus_bins[10:]

    branch = -1 if score_rng(_base_key(key), "truth_mode").integers(2) == 0 else 1
    truth_noise = score_rng(key, "truth").standard_normal(
        N_VIEWS * key.horizon
    )
    truth_bins = mean_bins + (truth_noise @ cholesky.T).reshape(
        N_VIEWS, key.horizon
    )
    truth_bins[0] += 4.0 * branch
    if np.any(base_bins < 0.0) or np.any(truth_bins < 0.0):
        raise ProtocolError("frozen synthetic generator left the range domain")

    mask = np.ones((N_VIEWS, key.horizon), dtype=np.bool_)
    observation = ObservedRangeMass(
        _oracle_mass(truth_bins), mask, BIN_M, provenance
    )
    truth = SyntheticTruth(truth_bins * BIN_M, branch, provenance)
    correct = ForecastRangeSamples(correct_bins * BIN_M, mask, provenance)
    return SyntheticCase(
        key=key,
        observation=observation,
        truth=truth,
        correct=correct,
        base_samples_m=base_bins * BIN_M,
        mode_minus_m=minus_bins * BIN_M,
        mode_plus_m=plus_bins * BIN_M,
        conditional_mean_m=mean_bins * BIN_M,
        config_provenance=synthetic_config_provenance(),
    )
```

- [x] **Step 5：重跑聚焦和累计测试**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k SyntheticPopulationTest -v
if ($LASTEXITCODE -ne 0) { throw "synthetic population tests failed" }
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -v
```

Expected: 50/50 模式、AR(1)⊗view 协方差、四前缀共享条件、独立 truth 子流、2% uniform mass 和确定性全部 PASS；0 skip。

---

### Task 6：实现六个冻结错误/保护臂与速度匹配历史打乱

**Files:**
- Modify: `tests/test_forecast_scores.py`
- Modify: `uwb_osr/forecast_scores.py`

- [x] **Step 1：先追加每个臂只改变注册因素的测试**

在测试文件追加：

```python
class CorruptionArmsTest(unittest.TestCase):
    def setUp(self):
        self.case = fs.build_synthetic_case(fs.ScoreKey(0, 11, 30, 5))

    def test_mean_shift_changes_only_top_by_three_bins(self):
        shifted = fs.inject_mean_shift(self.case)
        difference = (shifted.range_m - self.case.correct.range_m) / fs.BIN_M
        np.testing.assert_allclose(difference[:, 0], 3.0, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(difference[:, 1:], 0.0, atol=1e-12, rtol=0.0)
        np.testing.assert_array_equal(shifted.valid_mask, self.case.correct.valid_mask)
        self.assertEqual(
            shifted.forecast_provenance,
            self.case.correct.forecast_provenance,
        )

    def test_dependency_break_preserves_every_marginal_multiset(self):
        broken = fs.inject_dependency_break(self.case)
        for view in range(3):
            for step in range(5):
                np.testing.assert_array_equal(
                    np.sort(broken.range_m[:, view, step]),
                    np.sort(self.case.correct.range_m[:, view, step]),
                )
        self.assertFalse(
            np.array_equal(broken.range_m, self.case.correct.range_m)
        )
        self.assertAlmostEqual(
            fs.frls_frnll(self.case.observation, broken)[1],
            fs.frls_frnll(self.case.observation, self.case.correct)[1],
            delta=1e-12,
        )
        self.assertAlmostEqual(
            fs.expected_range_crps(self.case.observation, broken),
            fs.expected_range_crps(
                self.case.observation, self.case.correct
            ),
            delta=1e-12,
        )

    def test_missing_mode_uses_all_twenty_minus_samples(self):
        missing = fs.inject_missing_mode(self.case)
        np.testing.assert_array_equal(missing.range_m, self.case.mode_minus_m)

    def test_extra_mode_is_eight_eight_plus_four_false_samples(self):
        extra = fs.inject_extra_mode(self.case)
        np.testing.assert_array_equal(extra.range_m[:8], self.case.mode_minus_m[:8])
        np.testing.assert_array_equal(extra.range_m[8:16], self.case.mode_plus_m[10:18])
        source_index = np.array([8, 9, 18, 19])
        false_offset = (extra.range_m[16:] - self.case.base_samples_m[source_index]) / fs.BIN_M
        np.testing.assert_allclose(false_offset[:, 0], 12.0, atol=1e-12, rtol=0.0)
        np.testing.assert_allclose(false_offset[:, 1:], 0.0, atol=1e-12, rtol=0.0)

    def test_overwide_keeps_mean_and_doubles_every_central_width(self):
        wide = fs.inject_overwide(self.case)
        np.testing.assert_allclose(
            np.mean(wide.range_m, axis=0),
            np.mean(self.case.correct.range_m, axis=0),
            atol=1e-15,
            rtol=0.0,
        )
        good = fs.coverage_sharpness(self.case.correct, self.case.truth)
        bad = fs.coverage_sharpness(wide, self.case.truth)
        np.testing.assert_allclose(
            bad.width_m, 2.0 * good.width_m, atol=1e-15, rtol=0.0
        )

    def test_history_shuffle_has_no_fixed_point_and_stays_in_speed_stratum(self):
        cases = tuple(
            fs.build_synthetic_case(fs.ScoreKey(0, sequence, 20, 5))
            for sequence in range(fs.SEQUENCES_PER_SUBJECT)
        )
        result = fs.history_shuffle_block(cases)
        self.assertEqual(len(result.forecasts), fs.SEQUENCES_PER_SUBJECT)
        self.assertEqual(
            sorted(result.donor_sequence),
            list(range(fs.SEQUENCES_PER_SUBJECT)),
        )
        for sequence, donor in enumerate(result.donor_sequence):
            self.assertNotEqual(sequence, donor)
            self.assertEqual(result.stratum[sequence], result.stratum[donor])
            np.testing.assert_array_equal(
                result.forecasts[sequence].range_m,
                cases[donor].correct.range_m,
            )
            self.assertEqual(
                result.forecasts[sequence].forecast_provenance,
                cases[sequence].observation.provenance,
            )

    def test_history_shuffle_rejects_incomplete_or_duplicate_block(self):
        cases = tuple(
            fs.build_synthetic_case(fs.ScoreKey(0, sequence, 20, 5))
            for sequence in range(fs.SEQUENCES_PER_SUBJECT)
        )
        with self.assertRaises(ProtocolError):
            fs.history_shuffle_block(cases[:-1])
        duplicate = cases[:-1] + (cases[0],)
        with self.assertRaises(ProtocolError):
            fs.history_shuffle_block(duplicate)
```

- [x] **Step 2：运行并确认错误臂接口缺失**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k CorruptionArmsTest -v
```

Expected: FAIL，缺少各 `inject_*` 与 `history_shuffle_block` 接口。

- [x] **Step 3：实现五个逐 key 确定性变换**

在生产模块追加：

```python
ARMS = (
    "mean_shift",
    "dependence_break",
    "missing_mode",
    "extra_mode",
    "history_shuffle",
    "overwide",
)


def _rewrap_forecast(
    case: SyntheticCase,
    range_m: np.ndarray,
) -> ForecastRangeSamples:
    if not isinstance(case, SyntheticCase):
        raise ProtocolError("corruption requires SyntheticCase")
    return ForecastRangeSamples(
        np.asarray(range_m, dtype=np.float64),
        case.correct.valid_mask,
        case.observation.provenance,
    )


def inject_mean_shift(case: SyntheticCase) -> ForecastRangeSamples:
    shifted = case.correct.range_m.copy()
    shifted[:, 0, :] += 3.0 * BIN_M
    return _rewrap_forecast(case, shifted)


def inject_dependency_break(case: SyntheticCase) -> ForecastRangeSamples:
    source = case.correct.range_m
    broken = np.empty_like(source)
    identity = tuple(range(source.shape[0]))
    used: set[tuple[int, ...]] = set()
    for view in range(N_VIEWS):
        for step in range(case.key.horizon):
            flat = view * case.key.horizon + step
            selected: Optional[tuple[int, ...]] = None
            for attempt in range(100):
                permutation = tuple(
                    int(value)
                    for value in score_rng(
                        case.key,
                        f"permutation:{flat}",
                        attempt,
                    ).permutation(source.shape[0])
                )
                if permutation != identity and permutation not in used:
                    selected = permutation
                    break
            if selected is None:
                raise ProtocolError("could not construct unique dependency permutation")
            used.add(selected)
            broken[:, view, step] = source[
                np.asarray(selected, dtype=np.int64), view, step
            ]
    return _rewrap_forecast(case, broken)


def inject_missing_mode(case: SyntheticCase) -> ForecastRangeSamples:
    return _rewrap_forecast(case, case.mode_minus_m)


def inject_extra_mode(case: SyntheticCase) -> ForecastRangeSamples:
    false_index = np.array([8, 9, 18, 19], dtype=np.int64)
    false_mode = case.base_samples_m[false_index].copy()
    false_mode[:, 0, :] += 12.0 * BIN_M
    samples = np.concatenate(
        [case.mode_minus_m[:8], case.mode_plus_m[10:18], false_mode],
        axis=0,
    )
    return _rewrap_forecast(case, samples)


def inject_overwide(case: SyntheticCase) -> ForecastRangeSamples:
    samples = case.correct.range_m
    center = np.mean(samples, axis=0, keepdims=True)
    widened = center + 2.0 * (samples - center)
    if np.any(widened < 0.0):
        raise ProtocolError("overwide arm left the nonnegative range domain")
    return _rewrap_forecast(case, widened)
```

- [x] **Step 4：实现 240 序列、5 等频速度层、无固定点循环置换**

继续在生产模块追加：

```python
@dataclass(frozen=True)
class HistoryShuffleResult:
    __slots__ = ("forecasts", "donor_sequence", "stratum")

    forecasts: tuple[ForecastRangeSamples, ...]
    donor_sequence: tuple[int, ...]
    stratum: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.forecasts) != SEQUENCES_PER_SUBJECT:
            raise ProtocolError("history shuffle forecast count differs")
        if len(self.donor_sequence) != SEQUENCES_PER_SUBJECT:
            raise ProtocolError("history shuffle donor count differs")
        if len(self.stratum) != SEQUENCES_PER_SUBJECT:
            raise ProtocolError("history shuffle stratum count differs")


def _mean_range_speed(forecast: ForecastRangeSamples) -> float:
    difference = np.diff(forecast.range_m, axis=2) / DT
    result = float(np.sqrt(np.mean(difference**2)))
    if not np.isfinite(result):
        raise ProtocolError("history speed is non-finite")
    return result


def history_shuffle_block(
    cases: Sequence[SyntheticCase],
) -> HistoryShuffleResult:
    block = tuple(cases)
    if len(block) != SEQUENCES_PER_SUBJECT:
        raise ProtocolError("history shuffle requires exactly 240 sequences")
    if any(not isinstance(case, SyntheticCase) for case in block):
        raise ProtocolError("history shuffle requires SyntheticCase values")
    first = block[0].key
    if any(
        case.key.pseudo_subject != first.pseudo_subject
        or case.key.prefix_t != first.prefix_t
        or case.key.horizon != first.horizon
        for case in block
    ):
        raise ProtocolError("history shuffle block keys are not comparable")
    by_sequence = {case.key.sequence: case for case in block}
    if set(by_sequence) != set(range(SEQUENCES_PER_SUBJECT)):
        raise ProtocolError("history shuffle sequences are missing or duplicated")
    canonical = tuple(by_sequence[index] for index in range(SEQUENCES_PER_SUBJECT))
    speeds = tuple(_mean_range_speed(case.correct) for case in canonical)
    ranked = sorted(
        range(SEQUENCES_PER_SUBJECT),
        key=lambda index: (
            speeds[index],
            score_seed(canonical[index].key, "history_speed_tie"),
        ),
    )
    strata = [-1] * SEQUENCES_PER_SUBJECT
    for rank, sequence in enumerate(ranked):
        strata[sequence] = rank // (SEQUENCES_PER_SUBJECT // 5)
    donor = [-1] * SEQUENCES_PER_SUBJECT
    for stratum in range(5):
        members = [
            sequence
            for sequence in range(SEQUENCES_PER_SUBJECT)
            if strata[sequence] == stratum
        ]
        members.sort(
            key=lambda sequence: score_seed(
                canonical[sequence].key, "history_order"
            )
        )
        if len(members) < 2:
            raise ProtocolError("history stratum cannot be cyclically shifted")
        for index, recipient in enumerate(members):
            donor[recipient] = members[(index + 1) % len(members)]
    if sorted(donor) != list(range(SEQUENCES_PER_SUBJECT)):
        raise ProtocolError("history shuffle is not one-to-one")
    if any(sequence == donor[sequence] for sequence in range(SEQUENCES_PER_SUBJECT)):
        raise ProtocolError("history shuffle contains a fixed point")
    forecasts = tuple(
        _rewrap_forecast(
            canonical[sequence],
            canonical[donor[sequence]].correct.range_m,
        )
        for sequence in range(SEQUENCES_PER_SUBJECT)
    )
    return HistoryShuffleResult(forecasts, tuple(donor), tuple(strata))
```

- [x] **Step 5：重跑错误臂、评分不变性和累计测试**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k CorruptionArmsTest -v
if ($LASTEXITCODE -ne 0) { throw "corruption arm tests failed" }
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -v
```

Expected: 六臂结构全部 PASS；依赖臂每个边缘样本集合逐位不变且 FRNLL/CRPS 差 ≤1e-12；历史臂无 fixed point；0 skip。

---

### Task 7：实现 prefix→序列→伪被试聚合与 2^8 精确配对检验

**Files:**
- Modify: `tests/test_forecast_scores.py`
- Modify: `uwb_osr/forecast_scores.py`

- [x] **Step 1：先追加聚合顺序、精确 p 与区间反演测试**

在测试文件追加：

```python
class PairedInferenceTest(unittest.TestCase):
    def full_delta_mapping(self) -> dict[fs.ScoreKey, float]:
        return {
            fs.ScoreKey(subject, sequence, prefix, 5): float(subject + 1)
            for subject in range(fs.PSEUDO_SUBJECTS)
            for sequence in range(fs.SEQUENCES_PER_SUBJECT)
            for prefix in fs.PREFIXES
        }

    def test_aggregation_first_averages_four_prefixes_then_240_sequences(self):
        subject_delta = fs.aggregate_subject_deltas(
            self.full_delta_mapping(),
            horizon=5,
            arm="mean_shift",
            metric="frnll",
        )
        np.testing.assert_array_equal(
            subject_delta, np.arange(1.0, 9.0, dtype=np.float64)
        )
        self.assertFalse(subject_delta.flags.writeable)

    def test_all_positive_equal_deltas_have_exact_minimum_one_sided_p(self):
        evidence = fs.exact_paired_evidence(
            np.ones(8, dtype=np.float64),
            horizon=5,
            arm="mean_shift",
            metric="frnll",
        )
        self.assertEqual(evidence.positive_directions, 8)
        self.assertEqual(evidence.p_one_sided, 1.0 / 256.0)
        self.assertEqual(evidence.mean_delta, 1.0)
        self.assertEqual(evidence.ci95, (1.0, 1.0))
        self.assertTrue(evidence.passed)

    def test_inverted_interval_is_shift_equivariant(self):
        delta = np.array([0.2, 0.4, 0.5, 0.8, 1.0, 1.1, 1.4, 1.7])
        first = fs.invert_sign_flip_interval(delta, alpha=0.05)
        shifted = fs.invert_sign_flip_interval(delta + 3.25, alpha=0.05)
        np.testing.assert_allclose(
            shifted,
            np.asarray(first) + 3.25,
            atol=1e-12,
            rtol=0.0,
        )

    def test_incomplete_unregistered_or_nonfinite_deltas_fail_closed(self):
        full = self.full_delta_mapping()
        missing = dict(full)
        missing.pop(next(iter(missing)))
        with self.assertRaises(ProtocolError):
            fs.aggregate_subject_deltas(
                missing, horizon=5, arm="mean_shift", metric="frnll"
            )
        with self.assertRaises(ProtocolError):
            fs.aggregate_subject_deltas(
                full, horizon=5, arm="overwide", metric="frnll"
            )
        with self.assertRaises(ProtocolError):
            fs.exact_paired_evidence(
                np.array([1.0] * 7 + [np.nan]),
                horizon=5,
                arm="mean_shift",
                metric="frnll",
            )

    def test_each_horizon_has_exactly_eight_registered_decision_cells(self):
        self.assertEqual(
            fs.RESPONSIBILITY_CELLS,
            (
                ("mean_shift", "frnll"),
                ("mean_shift", "range_crps"),
                ("dependence_break", "variogram"),
                ("missing_mode", "frnll"),
                ("missing_mode", "range_crps"),
                ("extra_mode", "frnll"),
                ("extra_mode", "range_crps"),
                ("history_shuffle", "frnll"),
            ),
        )
```

- [x] **Step 2：运行并确认统计接口缺失**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k PairedInferenceTest -v
```

Expected: FAIL，缺少聚合、精确检验和区间反演接口。

- [x] **Step 3：实现注册责任矩阵与唯一聚合顺序**

在生产模块追加：

```python
RESPONSIBILITY_CELLS = (
    ("mean_shift", "frnll"),
    ("mean_shift", "range_crps"),
    ("dependence_break", "variogram"),
    ("missing_mode", "frnll"),
    ("missing_mode", "range_crps"),
    ("extra_mode", "frnll"),
    ("extra_mode", "range_crps"),
    ("history_shuffle", "frnll"),
)


def _validate_registered_cell(horizon: int, arm: str, metric: str) -> None:
    if horizon not in HORIZONS:
        raise ProtocolError("unregistered decision horizon")
    if (arm, metric) not in RESPONSIBILITY_CELLS:
        raise ProtocolError("unregistered arm/metric responsibility cell")


def aggregate_subject_deltas(
    keyed_delta: Mapping[ScoreKey, float],
    *,
    horizon: int,
    arm: str,
    metric: str,
) -> np.ndarray:
    _validate_registered_cell(horizon, arm, metric)
    if not isinstance(keyed_delta, Mapping):
        raise ProtocolError("keyed_delta must be a mapping")
    expected_count = (
        PSEUDO_SUBJECTS * SEQUENCES_PER_SUBJECT * len(PREFIXES)
    )
    if len(keyed_delta) != expected_count:
        raise ProtocolError("paired delta mapping is incomplete")
    cube = np.empty(
        (PSEUDO_SUBJECTS, SEQUENCES_PER_SUBJECT, len(PREFIXES)),
        dtype=np.float64,
    )
    seen: set[ScoreKey] = set()
    prefix_index = {prefix: index for index, prefix in enumerate(PREFIXES)}
    for key, value in keyed_delta.items():
        if not isinstance(key, ScoreKey) or key.horizon != horizon:
            raise ProtocolError("paired delta key is invalid")
        number = float(value)
        if not np.isfinite(number):
            raise ProtocolError("paired delta must be finite")
        if key in seen:
            raise ProtocolError("duplicate paired delta key")
        seen.add(key)
        cube[key.pseudo_subject, key.sequence, prefix_index[key.prefix_t]] = number
    if len(seen) != expected_count:
        raise ProtocolError("paired delta key coverage differs")
    sequence_delta = np.mean(cube, axis=2)
    subject_delta = np.mean(sequence_delta, axis=1)
    subject_delta.setflags(write=False)
    return subject_delta
```

- [x] **Step 4：实现单侧精确 p 与同族双侧检验的区间反演**

继续在生产模块追加：

```python
@dataclass(frozen=True)
class PairedEvidence:
    __slots__ = (
        "horizon",
        "arm",
        "metric",
        "subject_delta",
        "mean_delta",
        "p_one_sided",
        "ci95",
        "positive_directions",
        "passed",
    )

    horizon: int
    arm: str
    metric: str
    subject_delta: np.ndarray
    mean_delta: float
    p_one_sided: float
    ci95: tuple[float, float]
    positive_directions: int
    passed: bool

    def __post_init__(self) -> None:
        _validate_registered_cell(self.horizon, self.arm, self.metric)
        delta = _readonly_exact(
            self.subject_delta, np.float64, "subject_delta"
        )
        if delta.shape != (PSEUDO_SUBJECTS,):
            raise ProtocolError("subject_delta must have shape (8,)")
        if not np.isfinite(delta).all():
            raise ProtocolError("subject_delta must be finite")
        object.__setattr__(self, "subject_delta", delta)


@lru_cache(maxsize=None)
def _sign_matrix(size: int) -> np.ndarray:
    signs = np.asarray(
        tuple(itertools.product((-1.0, 1.0), repeat=size)),
        dtype=np.float64,
    )
    signs.setflags(write=False)
    return signs


def _two_sided_sign_flip_p(delta: np.ndarray, shift: float) -> float:
    centered = delta - shift
    signs = _sign_matrix(centered.size)
    observed = abs(float(np.mean(centered)))
    permuted = np.abs(np.mean(signs * centered[None, :], axis=1))
    threshold = np.nextafter(observed, -np.inf)
    return float(np.count_nonzero(permuted >= threshold) / signs.shape[0])


def invert_sign_flip_interval(
    subject_delta: np.ndarray,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    delta = np.asarray(subject_delta, dtype=np.float64)
    if delta.shape != (PSEUDO_SUBJECTS,) or not np.isfinite(delta).all():
        raise ProtocolError("interval inversion requires eight finite deltas")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ProtocolError("alpha must be in (0,1)")
    signs = _sign_matrix(delta.size)
    observed_mean = float(np.mean(delta))
    signed_mean = np.mean(signs * delta[None, :], axis=1)
    sign_mean = np.mean(signs, axis=1)
    roots = []
    first_denominator = 1.0 - sign_mean
    second_denominator = 1.0 + sign_mean
    for numerator, denominator in (
        (observed_mean - signed_mean, first_denominator),
        (observed_mean + signed_mean, second_denominator),
    ):
        valid = np.abs(denominator) > 0.0
        roots.extend((numerator[valid] / denominator[valid]).tolist())
    roots_array = np.unique(np.asarray(roots, dtype=np.float64))
    if roots_array.size == 0 or not np.isfinite(roots_array).all():
        raise ProtocolError("sign-flip inversion has no finite breakpoints")

    accepted_segments: list[tuple[float, float]] = []
    for root in roots_array:
        if _two_sided_sign_flip_p(delta, float(root)) >= alpha:
            accepted_segments.append((float(root), float(root)))
    for lower, upper in zip(roots_array[:-1], roots_array[1:]):
        midpoint = float(lower + (upper - lower) / 2.0)
        if _two_sided_sign_flip_p(delta, midpoint) >= alpha:
            accepted_segments.append((float(lower), float(upper)))
    span = max(1.0, float(np.ptp(roots_array)))
    for outer in (float(roots_array[0] - span), float(roots_array[-1] + span)):
        if _two_sided_sign_flip_p(delta, outer) >= alpha:
            raise ProtocolError("sign-flip interval is unexpectedly unbounded")
    if not accepted_segments:
        raise ProtocolError("sign-flip interval acceptance set is empty")
    accepted_segments.sort()
    merged: list[list[float]] = []
    for lower, upper in accepted_segments:
        if not merged or lower > np.nextafter(merged[-1][1], np.inf):
            merged.append([lower, upper])
        else:
            merged[-1][1] = max(merged[-1][1], upper)
    if len(merged) != 1:
        raise ProtocolError("sign-flip acceptance set is disconnected")
    return float(merged[0][0]), float(merged[0][1])


def exact_paired_evidence(
    subject_delta: np.ndarray,
    *,
    horizon: int,
    arm: str,
    metric: str,
) -> PairedEvidence:
    _validate_registered_cell(horizon, arm, metric)
    delta = np.asarray(subject_delta, dtype=np.float64)
    if delta.shape != (PSEUDO_SUBJECTS,) or not np.isfinite(delta).all():
        raise ProtocolError("paired evidence requires eight finite deltas")
    signs = _sign_matrix(delta.size)
    observed = float(np.mean(delta))
    permuted = np.mean(signs * delta[None, :], axis=1)
    threshold = np.nextafter(observed, -np.inf)
    p_one_sided = float(
        np.count_nonzero(permuted >= threshold) / signs.shape[0]
    )
    interval = invert_sign_flip_interval(delta, alpha=0.05)
    positive = int(np.count_nonzero(delta > 0.0))
    passed = bool(observed > 0.0 and p_one_sided < 0.05 and interval[0] > 0.0)
    return PairedEvidence(
        horizon=horizon,
        arm=arm,
        metric=metric,
        subject_delta=delta,
        mean_delta=observed,
        p_one_sided=p_one_sided,
        ci95=interval,
        positive_directions=positive,
        passed=passed,
    )
```

区间算法枚举所有符号翻转均值与观测均值绝对值发生相等的有限断点，在每个断点和相邻开区间内评估同一个双侧精确检验，再取唯一连通接受集的闭包。它不是正态近似，也不是 bootstrap。

- [x] **Step 5：重跑统计与累计测试**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k PairedInferenceTest -v
if ($LASTEXITCODE -ne 0) { throw "paired inference tests failed" }
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -v
```

Expected: 聚合顺序、`1/256` 单侧 p、区间平移等变性、24 个注册责任格和 fail-closed 全部 PASS；0 skip。

---

### Task 8：组装三个 H 的完整判别力门与逐格 manifest

**Files:**
- Modify: `tests/test_forecast_scores.py`
- Modify: `uwb_osr/forecast_scores.py`

- [x] **Step 1：先追加完整责任矩阵、边缘不变与过宽保护测试**

在测试文件追加：

```python
class FullSyntheticGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = fs.run_synthetic_selftest()

    def test_manifest_has_all_twenty_four_decision_cells(self):
        cells = self.manifest["decision_cells"]
        self.assertEqual(len(cells), len(fs.HORIZONS) * 8)
        observed = {
            (cell["horizon"], cell["arm"], cell["metric"])
            for cell in cells
        }
        expected = {
            (horizon, arm, metric)
            for horizon in fs.HORIZONS
            for arm, metric in fs.RESPONSIBILITY_CELLS
        }
        self.assertEqual(observed, expected)
        for cell in cells:
            self.assertTrue(cell["passed"])
            self.assertGreater(cell["mean_delta"], 0.0)
            self.assertLess(cell["p_one_sided"], 0.05)
            self.assertGreater(cell["ci95"][0], 0.0)
            self.assertGreaterEqual(cell["positive_directions"], 1)
            self.assertLessEqual(cell["positive_directions"], 8)

    def test_dependency_marginals_are_invariant_for_every_horizon(self):
        cells = self.manifest["invariance_cells"]
        self.assertEqual(len(cells), len(fs.HORIZONS) * 2)
        for cell in cells:
            self.assertIn(cell["metric"], ("frnll", "range_crps"))
            self.assertLessEqual(cell["max_abs_difference"], 1e-12)
            self.assertTrue(cell["passed"])

    def test_overwide_is_rejected_by_width_without_coverage_dominance(self):
        cells = self.manifest["overwide_cells"]
        self.assertEqual(len(cells), len(fs.HORIZONS))
        for cell in cells:
            self.assertGreaterEqual(cell["min_subject_width_ratio"], 1.5)
            self.assertFalse(cell["coverage_used_for_dominance"])
            self.assertFalse(cell["overwide_dominates"])
            self.assertTrue(cell["passed"])

    def test_manifest_boundary_and_frozen_scale_are_explicit(self):
        self.assertEqual(self.manifest["gate"], "P0c-a")
        self.assertEqual(self.manifest["status"], "PASS")
        self.assertTrue(self.manifest["synthetic_only"])
        self.assertEqual(self.manifest["real_signal_reads"], 0)
        self.assertEqual(self.manifest["verdict_boundary"], fs.BOUNDARY)
        config = self.manifest["config"]
        self.assertEqual(config["pseudo_subjects"], 8)
        self.assertEqual(config["sequences_per_subject"], 240)
        self.assertEqual(config["prefixes"], [20, 30, 45, 60])
        self.assertEqual(config["horizons"], [5, 10, 20])
        self.assertEqual(config["k"], 20)
        self.assertEqual(config["n_bins"], 189)
        self.assertEqual(config["epsilon"], 1e-12)

    def test_in_process_manifest_is_deterministic_and_has_no_partial_language(self):
        second = fs.run_synthetic_selftest()
        self.assertEqual(self.manifest, second)
        encoded = json.dumps(
            self.manifest, sort_keys=True, separators=(",", ":")
        )
        self.assertNotIn("SKIPPED", encoded)
        self.assertNotIn("PARTIAL PASS", encoded)
        self.assertNotIn("trajectory prediction PASS", encoded)
        self.assertNotIn("CV wins", encoded)
```

- [x] **Step 2：运行并确认总门接口缺失**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k FullSyntheticGateTest -v
```

Expected: FAIL，缺少 `run_synthetic_selftest`；此前单元测试仍保持通过。

- [x] **Step 3：把完整冻结常量纳入配置 provenance 并添加门配置校验**

将生产模块中的 `_configuration_payload` 替换为：

```python
def _configuration_payload() -> dict[str, object]:
    return {
        "anchor_bins": [48.0, 140.0],
        "dt": DT,
        "endpoint_reflection_bins": [24.0, 164.0],
        "arms": {
            "dependence_break": "unique_per_dimension_permutation",
            "extra_mode": {"false_weight": 0.20, "top_shift_bins": 12.0},
            "history_shuffle": {"speed_strata": 5, "cycle_shift": 1},
            "mean_shift_bins": 3.0,
            "missing_mode": -1,
            "overwide_multiplier": 2.0,
        },
        "epsilon": EPSILON,
        "horizons": list(HORIZONS),
        "k": K_GATE,
        "mode_top_shift_bins": [-4.0, 4.0],
        "n_bins": N_BINS,
        "prefix_scale": [[key, PREFIX_SCALE[key]] for key in PREFIXES],
        "prefixes": list(PREFIXES),
        "protocol": PROTOCOL_VERSION,
        "pseudo_subject_sigma_bins": [1.70, 0.10],
        "pseudo_subjects": PSEUDO_SUBJECTS,
        "range_window_m": RANGE_WINDOW_M,
        "sequences_per_subject": SEQUENCES_PER_SUBJECT,
        "slope_bins_per_frame": [-0.6, 0.6],
        "time_correlation": 0.85,
        "truth_mass_outlier": 0.02,
        "truth_mass_sigma_bins": 2.0,
        "variogram_p": 0.5,
        "view_correlation": VIEW_CORRELATION.tolist(),
    }


def _validate_gate_configuration() -> None:
    if HORIZONS != (5, 10, 20):
        raise ProtocolError("P0c-a horizons differ from the frozen registry")
    if PREFIXES != (20, 30, 45, 60):
        raise ProtocolError("P0c-a prefixes differ from the frozen registry")
    if K_GATE != 20:
        raise ProtocolError("P0c-a gate requires K=20")
    if PSEUDO_SUBJECTS != 8 or SEQUENCES_PER_SUBJECT != 240:
        raise ProtocolError("P0c-a population scale differs")
    if N_BINS != 189 or BIN_M != 1.2 / 189 or EPSILON != 1e-12:
        raise ProtocolError("P0c-a grid or epsilon differs")
    if DT != 0.05:
        raise ProtocolError("history-stratum DT differs from P0b")
    if len(RESPONSIBILITY_CELLS) != 8 or len(ARMS) != 6:
        raise ProtocolError("P0c-a responsibility registry differs")
```

- [x] **Step 4：实现每个 block 的共同随机数评分与流式 delta 收集**

在生产模块追加：

```python
def _loss_value(
    case: SyntheticCase,
    forecast: ForecastRangeSamples,
    metric: str,
) -> float:
    if metric == "frnll":
        return frls_frnll(case.observation, forecast)[1]
    if metric == "range_crps":
        return expected_range_crps(case.observation, forecast)
    if metric == "variogram":
        return variogram_score(forecast, case.truth)
    raise ProtocolError("unregistered loss metric")


def _manifest_float(value: float) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ProtocolError("manifest contains a non-finite number")
    return float(format(number, ".15g"))


def _paired_manifest(evidence: PairedEvidence) -> dict[str, object]:
    return {
        "arm": evidence.arm,
        "ci95": [
            _manifest_float(evidence.ci95[0]),
            _manifest_float(evidence.ci95[1]),
        ],
        "horizon": evidence.horizon,
        "mean_delta": _manifest_float(evidence.mean_delta),
        "metric": evidence.metric,
        "p_one_sided": _manifest_float(evidence.p_one_sided),
        "passed": evidence.passed,
        "positive_directions": evidence.positive_directions,
        "pseudo_subjects": PSEUDO_SUBJECTS,
    }


def _run_synthetic_gate() -> dict[str, object]:
    _validate_gate_configuration()
    decision_cells: list[dict[str, object]] = []
    invariance_cells: list[dict[str, object]] = []
    overwide_cells: list[dict[str, object]] = []

    for horizon in HORIZONS:
        keyed_delta: dict[tuple[str, str], dict[ScoreKey, float]] = {
            cell: {} for cell in RESPONSIBILITY_CELLS
        }
        invariant_max = {"frnll": 0.0, "range_crps": 0.0}
        good_width = np.zeros(PSEUDO_SUBJECTS, dtype=np.float64)
        wide_width = np.zeros(PSEUDO_SUBJECTS, dtype=np.float64)
        good_coverage = np.zeros(PSEUDO_SUBJECTS, dtype=np.float64)
        wide_coverage = np.zeros(PSEUDO_SUBJECTS, dtype=np.float64)
        coverage_count = np.zeros(PSEUDO_SUBJECTS, dtype=np.int64)

        for subject in range(PSEUDO_SUBJECTS):
            for prefix in PREFIXES:
                cases = tuple(
                    build_synthetic_case(
                        ScoreKey(subject, sequence, prefix, horizon)
                    )
                    for sequence in range(SEQUENCES_PER_SUBJECT)
                )
                history = history_shuffle_block(cases)
                for sequence, case in enumerate(cases):
                    good_frnll = _loss_value(case, case.correct, "frnll")
                    good_crps = _loss_value(case, case.correct, "range_crps")
                    good_variogram = _loss_value(
                        case, case.correct, "variogram"
                    )

                    mean_shift = inject_mean_shift(case)
                    keyed_delta[("mean_shift", "frnll")][case.key] = (
                        _loss_value(case, mean_shift, "frnll") - good_frnll
                    )
                    keyed_delta[("mean_shift", "range_crps")][case.key] = (
                        _loss_value(case, mean_shift, "range_crps") - good_crps
                    )

                    dependence = inject_dependency_break(case)
                    dependence_frnll = _loss_value(case, dependence, "frnll")
                    dependence_crps = _loss_value(
                        case, dependence, "range_crps"
                    )
                    invariant_max["frnll"] = max(
                        invariant_max["frnll"],
                        abs(dependence_frnll - good_frnll),
                    )
                    invariant_max["range_crps"] = max(
                        invariant_max["range_crps"],
                        abs(dependence_crps - good_crps),
                    )
                    keyed_delta[("dependence_break", "variogram")][case.key] = (
                        _loss_value(case, dependence, "variogram")
                        - good_variogram
                    )

                    missing = inject_missing_mode(case)
                    keyed_delta[("missing_mode", "frnll")][case.key] = (
                        _loss_value(case, missing, "frnll") - good_frnll
                    )
                    keyed_delta[("missing_mode", "range_crps")][case.key] = (
                        _loss_value(case, missing, "range_crps") - good_crps
                    )

                    extra = inject_extra_mode(case)
                    keyed_delta[("extra_mode", "frnll")][case.key] = (
                        _loss_value(case, extra, "frnll") - good_frnll
                    )
                    keyed_delta[("extra_mode", "range_crps")][case.key] = (
                        _loss_value(case, extra, "range_crps") - good_crps
                    )

                    keyed_delta[("history_shuffle", "frnll")][case.key] = (
                        _loss_value(
                            case, history.forecasts[sequence], "frnll"
                        )
                        - good_frnll
                    )

                    wide = inject_overwide(case)
                    good_interval = coverage_sharpness(case.correct, case.truth)
                    wide_interval = coverage_sharpness(wide, case.truth)
                    good_width[subject] += good_interval.width_m[2]
                    wide_width[subject] += wide_interval.width_m[2]
                    good_coverage[subject] += good_interval.coverage[2]
                    wide_coverage[subject] += wide_interval.coverage[2]
                    coverage_count[subject] += 1

        for arm, metric in RESPONSIBILITY_CELLS:
            subject_delta = aggregate_subject_deltas(
                keyed_delta[(arm, metric)],
                horizon=horizon,
                arm=arm,
                metric=metric,
            )
            decision_cells.append(
                _paired_manifest(
                    exact_paired_evidence(
                        subject_delta,
                        horizon=horizon,
                        arm=arm,
                        metric=metric,
                    )
                )
            )

        for metric in ("frnll", "range_crps"):
            difference = invariant_max[metric]
            invariance_cells.append(
                {
                    "arm": "dependence_break",
                    "atol": 1e-12,
                    "horizon": horizon,
                    "max_abs_difference": _manifest_float(difference),
                    "metric": metric,
                    "passed": bool(difference <= 1e-12),
                    "rtol": 0.0,
                }
            )

        if np.any(good_width <= 0.0) or np.any(coverage_count == 0):
            raise ProtocolError("overwide aggregation has an empty denominator")
        width_ratio = wide_width / good_width
        good_rate = good_coverage / coverage_count
        wide_rate = wide_coverage / coverage_count
        coverage_closer = int(
            np.count_nonzero(
                np.abs(wide_rate - 0.90) < np.abs(good_rate - 0.90)
            )
        )
        width_pass = bool(np.min(width_ratio) >= 1.5)
        overwide_cells.append(
            {
                "arm": "overwide",
                "coverage_closer_subjects": coverage_closer,
                "coverage_used_for_dominance": False,
                "horizon": horizon,
                "metric": "width90",
                "min_subject_width_ratio": _manifest_float(
                    np.min(width_ratio)
                ),
                "overwide_dominates": False,
                "passed": width_pass,
                "subject_width_ratio": [
                    _manifest_float(value) for value in width_ratio
                ],
            }
        )

    failures = []
    failures.extend(
        f"decision:{cell['horizon']}:{cell['arm']}:{cell['metric']}"
        for cell in decision_cells
        if not cell["passed"]
    )
    failures.extend(
        f"invariance:{cell['horizon']}:{cell['metric']}"
        for cell in invariance_cells
        if not cell["passed"]
    )
    failures.extend(
        f"overwide:{cell['horizon']}"
        for cell in overwide_cells
        if not cell["passed"]
    )
    if failures:
        raise ProtocolError("synthetic score gate failed: " + ",".join(failures))

    return {
        "config": {
            "epsilon": EPSILON,
            "horizons": list(HORIZONS),
            "k": K_GATE,
            "n_bins": N_BINS,
            "prefixes": list(PREFIXES),
            "pseudo_subjects": PSEUDO_SUBJECTS,
            "sequences_per_subject": SEQUENCES_PER_SUBJECT,
        },
        "config_provenance": synthetic_config_provenance(),
        "decision_cells": decision_cells,
        "gate": "P0c-a",
        "invariance_cells": invariance_cells,
        "overwide_cells": overwide_cells,
        "protocol_version": PROTOCOL_VERSION,
        "real_signal_reads": 0,
        "status": "PASS",
        "synthetic_only": True,
        "verdict_boundary": BOUNDARY,
    }
```

- [x] **Step 5：缓存进程内规范 JSON、每次返回独立 manifest**

继续追加：

```python
@lru_cache(maxsize=1)
def _synthetic_manifest_json() -> str:
    return json.dumps(
        _run_synthetic_gate(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def run_synthetic_selftest() -> dict[str, object]:
    return json.loads(_synthetic_manifest_json())
```

这样同一测试进程只运行一次完整 8×240×4×3 总体，但每次调用得到新的 dict，调用者不能污染缓存；独立 CLI 进程仍会各自重新计算，用于检验跨进程确定性。

- [x] **Step 6：运行完整门；失败时严格停在 NO-GO**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k FullSyntheticGateTest -v
```

Expected GO 路径: 24 个决定性单元、6 个边缘不变单元、3 个过宽保护单元全部 PASS。若任何单元失败，保留原始输出并停止；不得改错误幅度、随机种子、伪被试数、序列数或删除 H 来获得通过。

- [x] **Step 7：运行累计 P0c-a 测试**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -v
```

Expected: 当前全部 P0c-a 测试 PASS，0 failure/error/skip；否则本任务未完成。

---

### Task 9：新增唯一 `--selftest` CLI、跨进程确定性与静态防越界检查

**Files:**
- Modify: `tests/test_forecast_scores.py`
- Create: `tools/w4_score_gate.py`

- [x] **Step 1：先追加 CLI、失败 JSON 和静态源码边界测试**

在测试文件 imports 中补入：

```python
import contextlib
import importlib.util
import io
from unittest import mock
```

在测试文件末尾追加：

```python
class CliAndStaticBoundaryTest(unittest.TestCase):
    @staticmethod
    def script() -> Path:
        return Path(__file__).resolve().parents[1] / "tools" / "w4_score_gate.py"

    def test_cli_requires_selftest_and_rejects_every_real_or_tracking_mode(self):
        for arguments in (
            [],
            ["--track"],
            ["--analyze"],
            ["--real"],
            ["--selftest", "--track"],
        ):
            result = subprocess.run(
                [sys.executable, str(self.script()), *arguments],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)

    def test_two_cli_processes_have_byte_identical_complete_stdout(self):
        command = [sys.executable, str(self.script()), "--selftest"]
        first = subprocess.run(command, check=False, capture_output=True)
        second = subprocess.run(command, check=False, capture_output=True)
        self.assertEqual(first.returncode, 0)
        self.assertEqual(second.returncode, 0)
        self.assertEqual(first.stderr, b"")
        self.assertEqual(second.stderr, b"")
        self.assertEqual(first.stdout, second.stdout)
        payload = json.loads(first.stdout)
        self.assertEqual(payload["status"], "PASS")
        self.assertEqual(len(payload["decision_cells"]), 24)
        self.assertEqual(len(payload["invariance_cells"]), 6)
        self.assertEqual(len(payload["overwide_cells"]), 3)
        decoded = first.stdout.decode("utf-8")
        self.assertNotIn("SKIPPED", decoded)
        self.assertNotIn("PARTIAL PASS", decoded)

    def test_protocol_error_path_is_one_fail_json_and_exit_one(self):
        spec = importlib.util.spec_from_file_location(
            "w4_score_gate_under_test", self.script()
        )
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            module,
            "run_synthetic_selftest",
            side_effect=ProtocolError("forced failure"),
        ):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = module.main(["--selftest"])
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        lines = stderr.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        payload = json.loads(lines[0])
        self.assertEqual(
            payload,
            {"error": "forced failure", "gate": "P0c-a", "status": "FAIL"},
        )

    def test_production_and_cli_sources_have_no_real_data_dependency(self):
        root = Path(__file__).resolve().parents[1]
        sources = (
            (root / "uwb_osr" / "forecast_scores.py").read_text(encoding="utf-8"),
            self.script().read_text(encoding="utf-8"),
        )
        forbidden = (
            "np.load",
            "open(",
            "cache/",
            "logs/",
            "torch",
            "scipy",
            "--track",
            "--analyze",
            "w3_readout",
            "gesture_label",
            "subject_label",
        )
        for source in sources:
            for token in forbidden:
                self.assertNotIn(token, source)

    def test_public_score_signatures_accept_no_arm_model_path_or_label(self):
        public = (
            fs.frls_frnll,
            fs.expected_range_crps,
            fs.energy_score,
            fs.variogram_score,
            fs.coverage_sharpness,
        )
        forbidden_names = {"arm", "model", "path", "label", "gesture", "subject"}
        for function in public:
            names = set(inspect.signature(function).parameters)
            self.assertTrue(names.isdisjoint(forbidden_names))
```

- [x] **Step 2：运行并确认 CLI 文件缺失**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k CliAndStaticBoundaryTest -v
```

Expected: FAIL，根因为 `tools/w4_score_gate.py` 尚不存在；不能通过跳过 CLI 测试获得绿灯。

- [x] **Step 3：实现只接受必选 `--selftest` 的薄 CLI**

用 `apply_patch` 新建 `tools/w4_score_gate.py`：

```python
#!/usr/bin/env python
"""P0c-a-v1 memory-only score-discriminability gate."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uwb_osr.causal_data import ProtocolError
from uwb_osr.forecast_scores import run_synthetic_selftest


def _json(payload: dict[str, object]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="P0c-a synthetic score-semantics gate."
    )
    parser.add_argument("--selftest", action="store_true", required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    _build_parser().parse_args(argv)
    try:
        manifest = run_synthetic_selftest()
    except ProtocolError as exc:
        print(
            _json({"error": str(exc), "gate": "P0c-a", "status": "FAIL"}),
            file=sys.stderr,
        )
        return 1
    print(_json(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 4：先跑静态/错误路径，再跑两个完整 CLI 进程**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_forecast_scores.py" -k CliAndStaticBoundaryTest -v
```

Expected: 静态 token、未知参数、单行 FAIL JSON 和两次完整 stdout 字节一致全部 PASS；0 skip。若完整门为 NO-GO，CLI 必须两次都 exit=1，不得伪造成功输出。

- [x] **Step 5：直接人工可复核 manifest 形状**

```powershell
$py = "D:\D_conda\envs\pytorch\python.exe"
$line = & $py tools/w4_score_gate.py --selftest
if ($LASTEXITCODE -ne 0) { throw "P0c-a CLI failed" }
$payload = $line | ConvertFrom-Json
if ($payload.decision_cells.Count -ne 24) { throw "missing decision cells" }
if ($payload.invariance_cells.Count -ne 6) { throw "missing invariance cells" }
if ($payload.overwide_cells.Count -ne 3) { throw "missing overwide cells" }
if ($payload.verdict_boundary -ne "P0c-a score semantics PASS != trajectory predictability PASS") {
  throw "verdict boundary differs"
}
$line
```

Expected: stdout 只有一行紧凑 JSON，且包含所有逐格证据，不是只打印总 PASS。

---

### Task 10：全量回归、哈希审计、证据写回与停止交接

**Files:**
- Verify unchanged: P0a/P0b 六个文件
- Verify unchanged: `proposal_v2/17_读出失败预测_预注册.md`
- Verify unchanged: `proposal_v2/18_通道D整数bin相位修正.md`
- Verify unchanged: `tools/w3_readout_predict.py`
- Modify only after GO: P0c-a design、本文、`proposal_v2/15_块二重构_因果状态估计_未来轨迹预测_未来回波生成.md`

- [x] **Step 1：运行三个聚焦套件，保留零 skip**

```powershell
$py = "D:\D_conda\envs\pytorch\python.exe"
foreach ($pattern in @(
  "test_causal_forecast_protocol.py",
  "test_causal_baselines.py",
  "test_forecast_scores.py"
)) {
  & $py -m unittest discover -s tests -p $pattern -v
  if ($LASTEXITCODE -ne 0) { throw "focused suite failed: $pattern" }
}
```

Expected: P0a 47/47、P0b 50/50、P0c-a 文件内全部测试 PASS；三个套件均 0 failure/error/skip。

- [x] **Step 2：运行全仓测试并只容许既有 reliability 单点失败**

```powershell
$output = & "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_*.py" -v 2>&1
$code = $LASTEXITCODE
$output
if ($code -ne 0) {
  $joined = $output -join "`n"
  $oldFailure = "test_corrupted_view_downweighted (test_reliability.TestReliabilityCalib)"
  if (-not $joined.Contains($oldFailure)) {
    throw "full suite failed somewhere other than the recorded reliability test"
  }
  if ($joined -notmatch 'FAILED \(failures=1\)') {
    throw "full suite has a new failure, error, or extra failure count"
  }
}
```

Expected comparison boundary: 除既有 `test_reliability.TestReliabilityCalib.test_corrupted_view_downweighted` 外无新 failure，且 0 error。若该旧失败消失也允许全绿；任何新增失败阻塞 P0c-a 完成。

- [x] **Step 3：运行生产/CLI 静态依赖与空白检查**

```powershell
$forbidden = rg -ni "np\.load|open\(|cache/|logs/|torch|scipy|--track|--analyze|w3_readout|gesture_label|subject_label|\bO1\b|\bW1\b|\bW3\b" uwb_osr/forecast_scores.py tools/w4_score_gate.py
if ($LASTEXITCODE -eq 0) { $forbidden; throw "forbidden P0c-a dependency found" }
if ($LASTEXITCODE -gt 1) { throw "rg dependency scan failed" }
$trailing = rg -n '[ \t]+$' uwb_osr/forecast_scores.py tests/test_forecast_scores.py tools/w4_score_gate.py
if ($LASTEXITCODE -eq 0) { $trailing; throw "trailing whitespace in P0c-a files" }
if ($LASTEXITCODE -gt 1) { throw "rg whitespace scan failed" }
```

Expected: 两个扫描均无命中。

- [x] **Step 4：复核 P0a/P0b、17/18/W3 哈希与 W3 分数缺失**

```powershell
$expected = @{
  "uwb_osr/causal_data.py" = "44C7CADB41596D5044EABAE17E47E55D9342AB65FDF104664FC9A11C7BC31676"
  "tests/test_causal_forecast_protocol.py" = "DC6B024481CF8CDD8748F3048FFF3ED2607671F2EBDE9EAC62592296B8AD828E"
  "tools/w4_causal_gate.py" = "8BA5B354D4833C9EA559E74F7B45CE553AD337FA238C01CC7D94F051BBB27AE4"
  "uwb_osr/causal_baselines.py" = "2EC5CF3CCBAA651A731F82C8BF9613AB74F7A4A2A8DDA6617C3F0AF43379485D"
  "tests/test_causal_baselines.py" = "C67FBB681242F30F9DC79E82366B9995D821BBD681FB1AF9C283F728CD3B3751"
  "tools/w4_kinematic_gate.py" = "B995D067600F65F0FD1FB7FF1EB431D51B614E1B5ECD9AE74CE3CB7FEB921547"
  "D:\uwb_gestures\proposal_v2\17_读出失败预测_预注册.md" = "FE35D845BB7FEEB62050A59423E4CF1088B056440B10822B27BFBC6595E1F32F"
  "D:\uwb_gestures\proposal_v2\18_通道D整数bin相位修正.md" = "2335E804440EB7B34416543D647B42F662A0E769B4A6427EC2ECBA79E8ADF7BE"
  "D:\uwb_gestures\tools\w3_readout_predict.py" = "5E597A6CDBA426CCEBCEB6609CFBCCEC4E680487B32F981BB43DBE83664F4F90"
}
foreach ($path in $expected.Keys) {
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $path).Hash
  if ($actual -ne $expected[$path]) { throw "$path hash changed: $actual" }
}
if (Test-Path -LiteralPath "D:\uwb_gestures\logs\w3_readout_scores.npz") {
  throw "forbidden W3 score exists"
}
```

Expected: 所有哈希逐位一致，W3 分数仍不存在。

- [x] **Step 5：捕获两次 stdout、完整 JSON 证据和三个新文件哈希**

```powershell
$py = "D:\D_conda\envs\pytorch\python.exe"
@'
from pathlib import Path
import hashlib
import json
import subprocess
import sys

command = [sys.executable, str(Path("tools/w4_score_gate.py")), "--selftest"]
first = subprocess.run(command, check=False, capture_output=True)
second = subprocess.run(command, check=False, capture_output=True)
if first.returncode != 0 or second.returncode != 0:
    raise SystemExit("P0c-a CLI returned nonzero")
if first.stderr or second.stderr:
    raise SystemExit("P0c-a CLI wrote stderr")
if first.stdout != second.stdout:
    raise SystemExit("P0c-a stdout differs across processes")
payload = json.loads(first.stdout)
if len(payload["decision_cells"]) != 24:
    raise SystemExit("P0c-a manifest lacks decision cells")
if not all(cell["passed"] for cell in payload["decision_cells"]):
    raise SystemExit("P0c-a decision cell failed")
print(
    "stdout_bytes=" + str(len(first.stdout))
    + " stdout_sha256=" + hashlib.sha256(first.stdout).hexdigest()
)
'@ | & $py -
if ($LASTEXITCODE -ne 0) { throw "P0c-a deterministic evidence failed" }
Get-FileHash -Algorithm SHA256 uwb_osr/forecast_scores.py,tests/test_forecast_scores.py,tools/w4_score_gate.py
```

Expected: 两次独立进程 stdout 字节相同；打印 stdout 长度/SHA-256 和三个新文件 SHA-256。原样保存这些实际值，不预填或四舍五入。

- [x] **Step 6：仅在全部 GO 条件成立后写回三处文档**

用 `apply_patch` 做以下确定性语义修改；证据数字必须复制 Step 1/2/5 的实际输出，不允许猜测：

1. P0c-a 设计头部状态改为“P0c-a-v1 已实现并通过纯合成门；等待用户证据复核；未授权 P0c-b 或真实数据”。
2. 本计划勾选实际完成项，并追加聚焦/全仓测试数、唯一旧失败、stdout 长度/哈希、三个新文件哈希、六个旧文件与 17/18/W3 未变、W3 分数缺失、无 stage/commit。
3. 在 `15` 的 §14 做下面的**精确语义更新**：

```diff
 ### 14.1 已完成、可直接复用
 
 - P0b-v1 最小因果 F0、Last/CV(2/5/20)/CA(5/20)、训练折运动上限、验证集选择器与端到端未来替换 C0
   （仅“运动学契约 PASS”，不是轨迹可预测性结果）；
+- P0c-a-v1 纯合成评分合同与判别力门：FRNLL、期望 range CRPS、合成联合 ES/VS、coverage–sharpness、
+  六个冻结错误/保护臂及三个 H 的逐格配对判决
+  （仅“评分语义 PASS”，不是轨迹可预测性结果，也不证明真实 8 被试下 5% 功效）；
@@
 ### 14.2 尚未完成
 
-- P0b 之后真实数据所需的最小 O1/评分判别力审计与验证折参数选择；
+- P0c-b：冻结最小 O1，并只用训练/验证/校准被试估计真实数据下的评分分辨能力与 5% 功效；
@@
-- 概率评分判别力体检与随机负对照；
+- 真实数据上的概率评分分辨能力、随机负对照与联合目标边界；
@@
 ### 14.3 当前停止点与后续顺序
 
-P0a-v1 与 P0b-v1 的纯合成/C0 实现均已完成。当前停止在 **P0b 证据复核之后、任何真实参数选择和
-轨迹比较之前**；后续顺序是：
+P0a-v1、P0b-v1 与 P0c-a-v1 的纯合成/C0 实现均已完成。当前停止在 **P0c-a 评分语义证据复核之前、
+任何 O1、真实参数选择和轨迹比较之前**；P0c-a score semantics PASS 不等于 trajectory predictability PASS。
+后续顺序是：
 
-1. 先由用户复核 P0b 证据，再单独批准真实数据所需的最小 O1/评分判别力审计和验证折参数选择；
+1. 先由用户复核 P0c-a 证据，再另行设计并书面冻结 P0c-b；本次 PASS 不自动授权读取真实数据；
```

不得在 `15` 写“轨迹预测成功”“CV/CA 变好/变差”或任何真实模型结果。

- [x] **Step 7：最终 no-stage/no-commit 与工作树范围检查**

```powershell
git diff --check
git diff --cached --name-only
git status --short
```

Expected: 无 whitespace error；staged 输出为空；P0a/P0b/P0c-a 九个实现/测试/CLI 文件仍为未跟踪，文档只出现本计划授权的状态更新。不要暂存或提交。

- [x] **Step 8：明确停止，不进入真实实验**

交接只报告：

```text
P0c-a score semantics PASS != trajectory predictability PASS
Next allowed action: user review, then P0c-b written design only.
No real signal read; no O1/W1/W3 run; no --track/--analyze; no model ranking.
```

#### 2026-08-01 P0c-a-v1 实施证据

- 聚焦回归：P0a 47/47（1.303 s）、P0b 50/50（1.217 s）、P0c-a 48/48（1455.107 s）；
  三个套件均为 0 failure、0 error、0 skip。
- 全仓回归：174 项（929.410 s）；仅有既有
  `test_reliability.TestReliabilityCalib.test_corrupted_view_downweighted` 单点失败，汇总为
  `FAILED (failures=1)`，无新增 failure/error。
- 两次独立 `tools/w4_score_gate.py --selftest` stdout 逐字节一致：7264 bytes，
  SHA-256 `be35861bef7c7c04a2ed305339c4278f38f183289504f5f48395711fb23a4bc5`；
  manifest 含 24 个决定格、6 个不变格、3 个过宽保护格，且 `real_signal_reads=0`。
- 新文件 SHA-256：
  - `uwb_osr/forecast_scores.py`：`9B7CE990483B0B17CAC5C86A57B7C83DBEC9D6EA144C0796ECC59EA14BC504AB`
  - `tests/test_forecast_scores.py`：`7F01786C839C1C24DD5A747F66AE88403B0B52F0D43891215FE83F2A542E6A8C`
  - `tools/w4_score_gate.py`：`0D8B9092066AFC0B40FA8B1449C6D92F61DFAF37C50EB1C45362C23C1B5E0A07`
- P0a/P0b 六个旧文件 SHA-256 逐位未变：
  `44C7CADB…1676`、`DC6B0244…828E`、`8BA5B354…7AE4`、
  `2EC5CF3C…485D`、`C67FBB68…3751`、`B995D067…1547`。
- 17/18/W3 SHA-256 逐位未变：`FE35D845…1F32F`、`2335E804…ADF7BE`、
  `5E597A6C…F4F90`；`logs/w3_readout_scores.npz` 不存在。
- 实施过程中未执行 stage、commit 或 merge；未读取真实 signal，未运行 O1/W1/W3、
  `--track/--analyze` 或模型排名。Step 7 只读复核确认：隔离工作树索引为空且恰有 9 个预期未跟踪文件；
  主工作树索引为空，既有任务外修改/未跟踪内容未触碰。整个 `proposal_v2/` 原本未跟踪，故文档范围以
  本计划列出的三份文件直接内容复核为准。

如果 Task 8 或 Task 9 为 NO-GO，则不执行文档 PASS 更新；保存失败的 H×arm×metric 证据并回到用户复审。

---

## 完成标准

P0c-a 实现只有在以下全部有证据时才完成：

1. 设计中的常量、RNG 字节协议、只读类型、shape/dtype/mask/provenance 失败语义全部实现；
2. 几何桥准确保持 K/view/H 轴，不进入显著性判决；
3. FRLS 明确先平均 K 个 likelihood 再取 log，越界预测保留并受 epsilon 惩罚；
4. 期望 range CRPS、ES、VS(p=0.5)、50/80/90 coverage–sharpness 与手算一致；
5. 冻结总体为 8×240、四相关 prefix、H=5/10/20、K=20，truth 与 forecast RNG 分离；
6. 六个错误/保护臂只改变注册因素，依赖臂的 FRNLL/CRPS 差不超过 1e-12；
7. 聚合顺序固定为 prefix→sequence→8 pseudo-subject，精确检验枚举 2^8；
8. 24 个决定性格均满足 mean>0、单侧 p<0.05、反演 95% 区间下界>0；
9. 三个过宽格的 90% width 比至少 1.5，coverage 不参与支配判决；
10. CLI 只接受必选 `--selftest`，两次独立进程 stdout 逐字节一致且逐格可审计；
11. P0a/P0b 聚焦回归不回退，全仓无新增 failure/error；
12. P0a/P0b、17/18/W3 哈希未变，W3 分数不存在；
13. 无真实 cache/log 读取，无 O1/W1/W3、`--track/--analyze` 或模型排名；
14. 无 stage/commit/merge；
15. 文档只把 GO 解释为评分尺子语义可信，下一步仍需 P0c-b 独立书面设计。

---

## 计划自审记录

| 获批设计要求 | 本计划覆盖 |
|---|---|
| 三个只读核心对象、严格 provenance、K=20 门 | Tasks 1、3、5、8 |
| SHA256 字节子流与共同随机数 | Tasks 1、5、6 |
| 最小位置→距离桥、不参与统计门 | Task 2 |
| FRLS log-after-average、189-bin 插值/越界 | Task 3 |
| 期望 range CRPS、合成 ES/VS、coverage–sharpness | Task 4 |
| 8×240、四 prefix、三 H、双峰联合真分布 | Task 5 |
| 均值、依赖、漏模、增模、历史打乱、过宽六臂 | Task 6 |
| prefix→sequence→subject 与 exact 2^8/CI | Task 7 |
| 24 决定格、6 不变格、3 保护格全量输出 | Task 8 |
| 仅 `--selftest`、跨进程确定性、静态防越界 | Task 9 |
| 回归、哈希、文档与 P0c-b 停止点 | Task 10 |

计划交付前必须再执行以下文档自审；这些命令只读，不运行实现：

```powershell
$plan = "D:\uwb_gestures\proposal_v2\specs\2026-08-01-p0c-score-discriminability-implementation-plan.md"
rg -n "Task [0-9]+|RESPONSIBILITY_CELLS|decision_cells|invariance_cells|overwide_cells|2\^8|P0c-b|trajectory predictability" $plan
$unfinished = rg -ni "T[O]DO|T[B]D|later f[i]ll|place[h]older|省[略]|同[上]|自行实[现]" $plan
if ($LASTEXITCODE -eq 0) { $unfinished; throw "unfinished implementation marker in plan" }
if ($LASTEXITCODE -gt 1) { throw "plan marker scan failed" }
```

Expected: Task 0–10、24/6/3 证据结构、精确检验和 P0c-b 边界都有命中；未完成标记扫描无命中。计划编写阶段不得创建三个 P0c-a 实现文件。
