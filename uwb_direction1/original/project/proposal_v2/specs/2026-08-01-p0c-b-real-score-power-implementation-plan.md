# P0c-b Real-Score Power Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现获批的 P0c-b-v1：冻结最小 O1，用嵌套被试级交叉拟合、确定性 `K=20` oracle/donor 坏臂和 7/8 人精确统计，审计真实评分尺子能否检出 5% FRNLL 退化。

**Architecture:** 新增三个隔离单元：`range_likelihood.py` 只负责固定模板与 O1，`score_power.py` 只负责角色拆分、受控臂和统计功效，`p0c_b_audit.py` 只负责无标签缓存适配与全折编排。CLI 将纯合成 `--selftest` 与须另行授权的 `--audit-real` 物理分离；既有 P0a/P0b/P0c-a 九个文件只读复用且哈希不变。

**Tech Stack:** Python 3.9.23、NumPy 2.0.2、SciPy 1.13.1（仅 `scipy.stats.beta.ppf`）、标准库 `argparse/dataclasses/hashlib/itertools/json/pathlib/struct/unittest`；CPU only。

**Plan status:** 用户已批准编写 implementation plan；实现、真实数据读取和实验运行尚未获准。

---

## 执行位置与硬边界

未来实现 worktree：`D:\uwb_gestures\.worktrees\p0a-causal-prefix`。固定解释器：

```powershell
& "D:\D_conda\envs\pytorch\python.exe"
```

用户未授权 `git add`、commit、merge。每个任务以测试、`git diff --check`、SHA-256 和空的
`git diff --cached --name-only` 作为检查点。

只允许新增：

```text
uwb_osr/range_likelihood.py
uwb_osr/score_power.py
uwb_osr/p0c_b_audit.py
tests/test_range_likelihood.py
tests/test_score_power.py
tests/test_p0c_b_audit.py
tools/w4_real_score_power_gate.py
```

实现验收后只可更新 P0c-b design/plan 和 `proposal_v2/15_...md` 的状态文字。以下九个文件保持字节不变：

```text
44C7CADB41596D5044EABAE17E47E55D9342AB65FDF104664FC9A11C7BC31676  uwb_osr/causal_data.py
DC6B024481CF8CDD8748F3048FFF3ED2607671F2EBDE9EAC62592296B8AD828E  tests/test_causal_forecast_protocol.py
8BA5B354D4833C9EA559E74F7B45CE553AD337FA238C01CC7D94F051BBB27AE4  tools/w4_causal_gate.py
2EC5CF3CCBAA651A731F82C8BF9613AB74F7A4A2A8DDA6617C3F0AF43379485D  uwb_osr/causal_baselines.py
C67FBB681242F30F9DC79E82366B9995D821BBD681FB1AF9C283F728CD3B3751  tests/test_causal_baselines.py
B995D067600F65F0FD1FB7FF1EB431D51B614E1B5ECD9AE74CE3CB7FEB921547  tools/w4_kinematic_gate.py
9B7CE990483B0B17CAC5C86A57B7C83DBEC9D6EA144C0796ECC59EA14BC504AB  uwb_osr/forecast_scores.py
7F01786C839C1C24DD5A747F66AE88403B0B52F0D43891215FE83F2A542E6A8C  tests/test_forecast_scores.py
0D8B9092066AFC0B40FA8B1449C6D92F61DFAF37C50EB1C45362C23C1B5E0A07  tools/w4_score_gate.py
```

另行批准前禁止读取 `cache/`、加载 `logs/w2_pulse.npz`、运行 `--audit-real`、O1 全量、W1/W3、
Last/CV/CA、`--track`、`--analyze` 或模型排名。

---

## 文件职责

| 文件 | 单一职责 |
|---|---|
| `uwb_osr/range_likelihood.py` | 模板合同、宽度库、幅度 profile、`sigma_v/p(s)`、outlier、O1 mass |
| `uwb_osr/score_power.py` | 平衡拆分、oracle、donor、`alpha_H`、聚合、精确统计与功效 |
| `uwb_osr/p0c_b_audit.py` | label-free subject store、外层编排、manifest 与 fail-closed verdict |
| 三个测试文件 | 分别验收上述三个边界，禁止跨层补救 |
| `tools/w4_real_score_power_gate.py` | `--selftest`/`--audit-real` 互斥 CLI |

---

### Task 0：确认获批设计、哈希与零真实读取起点

**Files:**
- Read only: `D:\uwb_gestures\proposal_v2\specs\2026-08-01-p0c-b-real-score-power-design.md`
- Read only: 上述九个冻结文件
- Assert absent: 七个 P0c-b 新文件

- [ ] **Step 1：核对设计停止边界**

```powershell
rg -n "书面设计已冻结|未授权实现|不得加载|8×7=56|4.75%|0.80|整体 NO-GO" `
  "D:\uwb_gestures\proposal_v2\specs\2026-08-01-p0c-b-real-score-power-design.md"
```

Expected: 七类约束全部命中。

- [ ] **Step 2：用本页哈希表逐项验证九文件，确认七个新文件不存在、staging 为空**

```powershell
$new = @("uwb_osr/range_likelihood.py", "uwb_osr/score_power.py", "uwb_osr/p0c_b_audit.py",
  "tests/test_range_likelihood.py", "tests/test_score_power.py", "tests/test_p0c_b_audit.py",
  "tools/w4_real_score_power_gate.py")
foreach ($path in $new) { if (Test-Path -LiteralPath $path) { throw "unexpected file: $path" } }
git diff --cached --name-only
```

Expected: 新文件不存在；staged 列表为空。哈希不符立即停止。

---

### Task 1：固定模板合同与宽度 shift bank

**Files:**
- Create: `tests/test_range_likelihood.py`
- Create: `uwb_osr/range_likelihood.py`

`range_likelihood.py` 文件头固定为：

```python
from __future__ import annotations
import hashlib
import math
import struct
from dataclasses import dataclass
import numpy as np
from uwb_osr.causal_data import ProtocolError, transform_representation
from uwb_osr.forecast_scores import BIN_M, HORIZONS, ObservedRangeMass
```

- [ ] **Step 1：先写模板字段、中心峰和只读测试**

```python
class PulseContractTest(unittest.TestCase):
    def payload(self):
        env = np.array([0.1, 0.5, 1.0, 0.5, 0.1], dtype=np.float64)
        return {"env": env, "W": np.array(2), "template": env.astype(np.complex128),
                "carrier": np.array(1.0), "amp_exp": np.array(-2.0),
                "width_bins": np.array(3.0)}

    def test_contract_uses_env_and_w_only(self):
        pulse = rl.pulse_from_payload(self.payload(), "a" * 64, "b" * 64)
        self.assertEqual(pulse.W, 2)
        self.assertFalse(pulse.env.flags.writeable)

    def test_rejects_off_center_peak(self):
        payload = self.payload()
        payload["env"] = np.array([1.0, 0.5, 0.1, 0.0, 0.0])
        with self.assertRaises(ProtocolError):
            rl.pulse_from_payload(payload, "a" * 64, "b" * 64)
```

- [ ] **Step 2：运行并确认因模块不存在而失败**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest tests.test_range_likelihood.PulseContractTest -v
```

Expected: FAIL，`uwb_osr.range_likelihood` 不存在。

- [ ] **Step 3：实现 `PulseTemplate` 与严格 payload 验证**

```python
PROTOCOL_VERSION = "P0c-b-v1"
WIDTH_SCALES = np.array([1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50], dtype=np.float64)
REQUIRED_PULSE_KEYS = ("template", "env", "W", "carrier", "amp_exp", "width_bins")

@dataclass(frozen=True)
class PulseTemplate:
    env: np.ndarray
    W: int
    file_sha256: str
    generator_sha256: str
    provenance: str

def pulse_from_payload(payload, file_sha256, generator_sha256):
    if set(payload) != set(REQUIRED_PULSE_KEYS):
        raise ProtocolError("pulse members differ from the frozen contract")
    if not is_sha256(file_sha256) or not is_sha256(generator_sha256):
        raise ProtocolError("pulse hashes must be SHA-256 hex")
    W = int(np.asarray(payload["W"]).item())
    env = np.array(payload["env"], dtype=np.float64, copy=True)
    if env.ndim != 1 or env.size != 2 * W + 1 or not np.isfinite(env).all():
        raise ProtocolError("env must be finite with len(env)=2W+1")
    if np.any(env < 0) or env.max() <= 0 or int(np.argmax(env)) != W:
        raise ProtocolError("env must peak at W")
    env /= env.max()
    env.setflags(write=False)
    provenance = hashlib.sha256((PROTOCOL_VERSION + file_sha256 + generator_sha256).encode()).hexdigest()
    return PulseTemplate(env, W, file_sha256, generator_sha256, provenance)
```

同一步加入：

```python
def is_sha256(value):
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True
```

- [ ] **Step 4：写并实现固定七宽度 `(7,189,189)` shift bank**

```python
@dataclass(frozen=True)
class WidthLibrary:
    scales: np.ndarray
    shift_bank: np.ndarray
    provenance: str

def build_width_library(pulse):
    bank = np.zeros((7, 189, 189), dtype=np.float64)
    source_x = np.arange(-pulse.W, pulse.W + 1, dtype=np.float64)
    for si, scale in enumerate(WIDTH_SCALES):
        rel = np.arange(-94, 95, dtype=np.float64) / scale
        energy = np.interp(rel, source_x, pulse.env, left=0.0, right=0.0) ** 2
        energy /= np.linalg.norm(energy)
        for rho in range(189):
            src = np.arange(189) - (rho - 94)
            valid = (src >= 0) & (src < 189)
            bank[si, rho, valid] = energy[src[valid]]
    scales = WIDTH_SCALES.copy()
    scales.setflags(write=False)
    bank.setflags(write=False)
    provenance = hashlib.sha256(pulse.provenance.encode() + bank.astype("<f8").tobytes()).hexdigest()
    return WidthLibrary(scales, bank, provenance)
```

Test asserts `scales==WIDTH_SCALES`、shape `(7,189,189)`、每个 shift 范数正、数组只读。

- [ ] **Step 5：运行 PulseContractTest 并检查工作树**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest tests.test_range_likelihood.PulseContractTest -v
git diff --check
git diff --cached --name-only
```

Expected: PASS；两个 Git 检查无输出。

---

### Task 2：幅度 profile、`sigma_v` 与共享宽度先验

**Files:**
- Modify: `tests/test_range_likelihood.py`
- Modify: `uwb_osr/range_likelihood.py`

- [ ] **Step 1：写闭式 amplitude/SSE 手算测试**

```python
def test_profile_sse_recovers_nonnegative_amplitude(self):
    bank = np.zeros((1, 2, 3), dtype=np.float64)
    bank[0, 0] = [1.0, 0.0, 0.0]
    bank[0, 1] = [0.0, 1.0, 0.0]
    sse = rl.profile_sse(np.array([[2.0, 1.0, 0.0]]), bank)
    np.testing.assert_allclose(sse, np.array([[[1.0], [4.0]]]))
```

- [ ] **Step 2：实现批量闭式 profile**

```python
def profile_sse(energy, shift_bank):
    x = np.asarray(energy, dtype=np.float64)
    bank = np.asarray(shift_bank, dtype=np.float64)
    dot = np.einsum("...b,srb->...rs", x, bank, optimize=True)
    norm2 = np.sum(bank**2, axis=2).T
    amp = np.maximum(0.0, dot / norm2)
    x2 = np.sum(x**2, axis=-1)[..., None, None]
    return np.maximum(x2 - 2 * amp * dot + amp**2 * norm2, 0.0)
```

- [ ] **Step 3：写五被试等权、Jeffreys `+0.5`、边界质量 `>0.25` 失败测试**

Test synthetic subject arrays with unequal row counts; duplicating one subject's rows must not change the other four subjects'
weight. Force `prior[0]+prior[-1]>0.25` and assert `ProtocolError`。

- [ ] **Step 4：实现确定性两遍 `fit_o1`**

```python
@dataclass(frozen=True)
class O1Fit:
    sigma_v: np.ndarray
    width_prior: np.ndarray
    boundary_mass: float
    provenance: str

FRAME_BATCH = 64

def frame_chunks(frames):
    for start in range(0, frames.shape[0], FRAME_BATCH):
        yield frames[start:start + FRAME_BATCH]

def fit_o1(subject_echoes, library):
    if len(subject_echoes) != 5:
        raise ProtocolError("O1 fit requires exactly five subjects")
    sigma_subject = []
    for subject in sorted(subject_echoes):
        energy = analytic_energy(_validate_echoes(subject_echoes[subject]))
        flat = energy.reshape(-1, 189)
        best = np.concatenate([np.min(profile_sse(chunk, library.shift_bank), axis=(1, 2))
                               for chunk in frame_chunks(flat)]).reshape(energy.shape[0], 3, 90)
        sigma_subject.append(np.sqrt(np.median(best / 188.0, axis=(0, 2))))
    sigma_v = np.median(np.stack(sigma_subject), axis=0)
    if not np.isfinite(sigma_v).all() or np.any(sigma_v <= 0):
        raise ProtocolError("sigma_v must be finite and positive")
    subject_prior = []
    for subject in sorted(subject_echoes):
        energy = analytic_energy(_validate_echoes(subject_echoes[subject])).reshape(-1, 3, 189)
        counts = np.full(7, 0.5)
        for view in range(3):
            for chunk in frame_chunks(energy[:, view]):
                sse = profile_sse(chunk, library.shift_bank)
                log_evidence = logsumexp(-sse / (2 * sigma_v[view]**2), axis=1)
                counts += softmax(log_evidence, axis=1).sum(axis=0)
        subject_prior.append(counts / counts.sum())
    prior = np.mean(np.stack(subject_prior), axis=0)
    boundary = float(prior[0] + prior[-1])
    if boundary > 0.25:
        raise ProtocolError("width prior boundary mass exceeds 25 percent")
    return readonly_fit(sigma_v, prior, boundary, library.provenance)
```

同一步加入所有被后续函数复用的数值 helper，避免在任务间出现未定义名字：

```python
def analytic_energy(x):
    return transform_representation(np.asarray(x), "envelope") ** 2

def _validate_echoes(x):
    value = np.asarray(x)
    if value.ndim != 4 or value.shape[1:] != (3, 90, 189) or value.dtype != np.float32:
        raise ProtocolError("subject echoes must have shape (N,3,90,189) float32")
    if not np.isfinite(value).all():
        raise ProtocolError("subject echoes must be finite")
    return value

def logsumexp(value, axis):
    x = np.asarray(value, dtype=np.float64)
    peak = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(np.sum(np.exp(x - peak), axis=axis))

def softmax(value, axis):
    x = np.asarray(value, dtype=np.float64)
    shifted = x - np.max(x, axis=axis, keepdims=True)
    weight = np.exp(shifted)
    return weight / np.sum(weight, axis=axis, keepdims=True)

def readonly_fit(sigma_v, prior, boundary, library_provenance):
    sigma = np.array(sigma_v, dtype=np.float64, copy=True)
    width = np.array(prior, dtype=np.float64, copy=True)
    sigma.setflags(write=False)
    width.setflags(write=False)
    digest = hashlib.sha256(library_provenance.encode() + sigma.astype("<f8").tobytes()
                            + width.astype("<f8").tobytes()).hexdigest()
    return O1Fit(sigma, width, float(boundary), digest)
```

- [ ] **Step 5：运行 O1FitTest**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest tests.test_range_likelihood.O1FitTest -v
```

Expected: amplitude、subject balance、边界失败测试全部 PASS。

---

### Task 3：全局 outlier 与 O1 `ObservedRangeMass`

**Files:**
- Modify: `tests/test_range_likelihood.py`
- Modify: `uwb_osr/range_likelihood.py`

- [ ] **Step 1：写 `pi_out` rounding 与 `>0.20` 失败测试**

```python
def test_global_outlier_rounds_to_point001(self):
    q = np.array([1.0] * 997 + [20.0] * 3)
    self.assertEqual(rl.calibrate_outlier_from_q(q, "c" * 64).pi_out, 0.003)

def test_rejects_more_than_twenty_percent(self):
    with self.assertRaises(ProtocolError):
        rl.calibrate_outlier_from_q(np.array([1.0] * 79 + [20.0] * 21), "c" * 64)
```

- [ ] **Step 2：实现唯一全局 outlier 对象**

```python
@dataclass(frozen=True)
class O1Calibration:
    pi_out: float
    threshold: float
    raw_fraction: float
    provenance: str

def calibrate_outlier_from_q(q, fit_provenance):
    values = np.asarray(q, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ProtocolError("Q must be a non-empty finite vector")
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    threshold = median + 3 * mad
    raw = float(np.mean(values > threshold))
    if raw > 0.20:
        raise ProtocolError("pi_raw exceeds 0.20")
    rounded = math.floor(raw * 1000 + 0.5) / 1000
    provenance = hashlib.sha256(f"{fit_provenance}|{threshold:.17g}|{raw:.17g}|{rounded:.3f}".encode()).hexdigest()
    return O1Calibration(rounded, threshold, raw, provenance)
```

校准被试的 `Q_i` 只能由冻结 fit/library 计算，仍按 64 帧分块：

```python
def calibrate_o1(calibration_echoes, library, fit):
    energy = analytic_energy(_validate_echoes(calibration_echoes)).reshape(-1, 3, 189)
    q_rows = []
    for view in range(3):
        for chunk in frame_chunks(energy[:, view]):
            sse = profile_sse(chunk, library.shift_bank)
            q_rows.append(np.min(sse, axis=(1, 2)) / (188 * fit.sigma_v[view]**2))
    return calibrate_outlier_from_q(np.concatenate(q_rows), fit.provenance)
```

- [ ] **Step 3：写并实现 normalized O1 mass**

```python
def o1_frame_mass(frames_cr, library, fit, calibration):
    frames = np.asarray(frames_cr)
    if frames.ndim != 3 or frames.shape[0] != 3 or frames.shape[1] < 1 or frames.shape[2] != 189:
        raise ProtocolError("frames_cr must have shape (3,T,189), T>=1")
    energy = analytic_energy(frames)
    mass = np.empty_like(energy, dtype=np.float64)
    for view in range(3):
        sse = profile_sse(energy[view], library.shift_bank)
        log_q = logsumexp(-sse / (2 * fit.sigma_v[view]**2) + np.log(fit.width_prior), axis=2)
        q = softmax(log_q, axis=1)
        mass[view] = (1 - calibration.pi_out) * q + calibration.pi_out / 189
    return mass

def observed_future_mass(future_cr, library, fit, calibration, key_provenance):
    frames = np.asarray(future_cr)
    if frames.shape[1] not in (5, 10, 20):
        raise ProtocolError("future horizon must be 5, 10 or 20")
    mass = o1_frame_mass(frames, library, fit, calibration)
    provenance = o1_provenance(library, fit, calibration, frames.shape[1], key_provenance)
    return ObservedRangeMass(mass, np.ones(frames.shape[:2], dtype=np.bool_), BIN_M, provenance)
```

`o1_provenance` 的字段顺序固定，不接受字典遍历顺序：

```python
def o1_provenance(library, fit, calibration, horizon, key_provenance):
    digest = hashlib.sha256(PROTOCOL_VERSION.encode())
    digest.update(library.provenance.encode("ascii"))
    digest.update(fit.provenance.encode("ascii"))
    digest.update(calibration.provenance.encode("ascii"))
    digest.update(struct.pack("<q", int(horizon)))
    digest.update(key_provenance.encode("ascii"))
    return digest.hexdigest()
```

Test asserts `(3,H,189) float64`、质量和 `1±1e-12`、全局 floor、provenance 变化、非有限输入 fail-closed。

- [ ] **Step 4：运行整个 range-likelihood 测试文件**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest tests.test_range_likelihood -v
git diff --check
```

Expected: 全部 PASS；diff check 无输出。

---

### Task 4：平衡拆分、midpoint oracle 与 donor

**Files:**
- Create: `tests/test_score_power.py`
- Create: `uwb_osr/score_power.py`

`score_power.py` 文件头固定为：

```python
from __future__ import annotations
import hashlib
import itertools
import struct
from dataclasses import dataclass
import numpy as np
from scipy.stats import beta
from uwb_osr.causal_data import ProtocolError
from uwb_osr.forecast_scores import (BIN_M, ForecastRangeSamples, ObservedRangeMass,
                                     expected_range_crps, frls_frnll)
```

- [ ] **Step 1：写七次循环角色测试**

```python
def test_balanced_cycle(self):
    splits = sp.inner_splits(outer_test=8)
    self.assertEqual(len(splits), 7)
    self.assertEqual({s.audit for s in splits}, set(range(1, 8)))
    self.assertEqual({s.calibration for s in splits}, set(range(1, 8)))
    for split in splits:
        self.assertEqual(len(split.fit_subjects), 5)
        self.assertNotIn(8, (split.audit, split.calibration, *split.fit_subjects))
```

- [ ] **Step 2：实现不可变 `InnerSplit`**

```python
@dataclass(frozen=True)
class InnerSplit:
    outer_test: int
    audit: int
    calibration: int
    fit_subjects: tuple[int, ...]

def inner_splits(outer_test):
    development = tuple(u for u in range(1, 9) if u != outer_test)
    rows = []
    for j, audit in enumerate(development):
        calibration = development[(j + 1) % 7]
        fit = tuple(u for u in development if u not in (audit, calibration))
        rows.append(InnerSplit(outer_test, audit, calibration, fit))
    return tuple(rows)
```

- [ ] **Step 3：写并实现 midpoint inverse-CDF**

```python
def midpoint_oracle(observation):
    quantiles = (np.arange(20, dtype=np.float64) + 0.5) / 20
    cdf = np.cumsum(observation.mass, axis=2)
    ranges = np.zeros((20, 3, observation.mass.shape[1]), dtype=np.float64)
    for view, step in np.argwhere(observation.valid_mask):
        ranges[:, view, step] = np.searchsorted(cdf[view, step], quantiles, side="left") * BIN_M
    return ForecastRangeSamples(ranges, observation.valid_mask, observation.provenance)
```

Test uses two point masses at bins 10/20 and asserts first ten samples equal `10*BIN_M`、last ten equal `20*BIN_M`。

donor 的数值样本必须重包为目标键 provenance，不能伪称 donor 键就是目标键：

```python
def donor_for_target(donor_oracle, target_observation):
    if donor_oracle.range_m.shape[1:] != target_observation.mass.shape[:2]:
        raise ProtocolError("donor and target shapes differ")
    return ForecastRangeSamples(donor_oracle.range_m, target_observation.valid_mask,
                                target_observation.provenance)
```

- [ ] **Step 4：写五帧 OLS、subject-balanced scale 和 donor 不看未来测试**

```python
def history_feature(history_mass):
    med = np.argmax(np.cumsum(history_mass, axis=2) >= 0.5, axis=2) * BIN_M
    y = med[:, -5:]
    t = np.arange(5, dtype=np.float64) * 0.05
    slope = np.sum((t - t.mean()) * (y - y.mean(axis=1, keepdims=True)), axis=1) / np.sum((t - t.mean())**2)
    return np.concatenate((med[:, -1], slope))

def nearest_donors(sample_ids, features, scale):
    distance = np.sum(((features[:, None] - features[None, :]) / scale) ** 2, axis=2)
    np.fill_diagonal(distance, np.inf)
    order = np.argsort(sample_ids)
    return np.array([sample_ids[order[np.argmin(distance[i, order])]] for i in range(sample_ids.size)])
```

五名拟合被试的 matching scale 必须被试等权：

```python
def fit_matching_scale(features_by_subject):
    if len(features_by_subject) != 5:
        raise ProtocolError("matching scale requires five subjects")
    rows = []
    for subject in sorted(features_by_subject):
        feature = np.asarray(features_by_subject[subject], dtype=np.float64)
        if feature.ndim != 2 or feature.shape[1] != 6 or not np.isfinite(feature).all():
            raise ProtocolError("matching features must have shape (N,6)")
        rows.append(np.std(feature, axis=0, ddof=1))
    scale = np.median(np.stack(rows), axis=0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ProtocolError("matching scale must be finite and positive")
    scale.setflags(write=False)
    return scale
```

Test asserts同被试/同 prefix/H 的候选池由调用者预先限定、donor 不同序列、并列取较小 `sample_idx`；
`nearest_donors` 签名不得包含 future、gesture 或 score。

- [ ] **Step 5：运行拆分/oracle/donor 测试**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest tests.test_score_power.SplitOracleDonorTest -v
```

Expected: 全部 PASS。

---

### Task 5：`alpha_H`、FRNLL/CRPS 与唯一聚合顺序

**Files:**
- Modify: `tests/test_score_power.py`
- Modify: `uwb_osr/score_power.py`

- [ ] **Step 1：写网格并列取小和不可达失败测试**

```python
def test_selects_closest_then_smaller_alpha(self):
    result = sp.select_alpha_from_curve(np.array([0.0, 0.049, 0.051]),
                                        np.array([0.0, 0.1, 0.2]), 5)
    self.assertEqual((result.alpha, result.achieved), (0.1, 0.049))

def test_rejects_outside_tolerance(self):
    with self.assertRaises(ProtocolError):
        sp.select_alpha_from_curve(np.array([0.0, 0.04]), np.array([0.0, 1.0]), 10)
```

- [ ] **Step 2：实现 `AlphaCalibration` 和冻结选择**

```python
@dataclass(frozen=True)
class AlphaCalibration:
    horizon: int
    alpha: float
    achieved: float

def select_alpha_from_curve(degradation, alpha_grid, horizon):
    distance = np.abs(np.asarray(degradation) - 0.05)
    index = int(np.flatnonzero(distance == distance.min())[0])
    achieved = float(degradation[index])
    if not 0.0475 <= achieved <= 0.0525:
        raise ProtocolError("alpha cannot reach the frozen interval")
    return AlphaCalibration(horizon, float(alpha_grid[index]), achieved)
```

- [ ] **Step 3：写并实现臂评分和四 prefix 等权相对差**

```python
def blend_bad(oracle, donor, alpha):
    if oracle.forecast_provenance != donor.forecast_provenance:
        raise ProtocolError("arm provenance differs")
    ranges = (1 - alpha) * oracle.range_m + alpha * donor.range_m
    return ForecastRangeSamples(ranges, oracle.valid_mask, oracle.forecast_provenance)

def score_pair(observation, oracle, donor, alpha):
    bad = blend_bad(oracle, donor, alpha)
    return (frls_frnll(observation, oracle)[1], frls_frnll(observation, bad)[1],
            expected_range_crps(observation, oracle), expected_range_crps(observation, bad))

def subject_relative_delta(good_prefix, bad_prefix):
    good = np.asarray(good_prefix, dtype=np.float64)
    bad = np.asarray(bad_prefix, dtype=np.float64)
    if good.shape != (4,) or np.any(good <= 0) or not np.isfinite(good).all():
        raise ProtocolError("four finite positive prefix FRNLL values required")
    return float(np.mean((bad - good) / good))

def fixed_offset_sensitivity(oracle):
    ranges = oracle.range_m.copy()
    ranges[:, 0, :] += 3 * BIN_M
    return ForecastRangeSamples(ranges, oracle.valid_mask, oracle.forecast_provenance)
```

Test `good=[1,10,100,1000]`、`bad/good=[1.01,1.02,1.03,1.04]`，必须得到 `0.025`，不得得到 pooled ratio。
另测固定偏移只改变 Top 视角 `+3*BIN_M`，并证明其结果字段不进入 `final_verdict`。

- [ ] **Step 4：运行 AlphaAggregationTest**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest tests.test_score_power.AlphaAggregationTest -v
```

Expected: 全部 PASS。

---

### Task 6：通用 7/8 人 exact sign-flip、区间与 5% 功效

**Files:**
- Modify: `tests/test_score_power.py`
- Modify: `uwb_osr/score_power.py`

- [ ] **Step 1：写 7 人 exact 与 P0c-a 8 人回归等价测试**

```python
def test_seven_subject_exact_probability(self):
    evidence = sp.exact_evidence(np.arange(1.0, 8.0))
    self.assertEqual(evidence.p_one_sided, 1 / 128)
    self.assertGreater(evidence.ci95[0], 0)

def test_eight_subject_matches_p0c_a(self):
    delta = np.arange(1.0, 9.0)
    old = fs.exact_paired_evidence(delta, horizon=5, arm="mean_shift", metric="frnll")
    new = sp.exact_evidence(delta)
    self.assertEqual((new.p_one_sided, new.ci95), (old.p_one_sided, old.ci95))
```

- [ ] **Step 2：实现任意 7/8 长度 exact evidence**

```python
def readonly(value):
    out = np.array(value, dtype=np.float64, copy=True)
    out.setflags(write=False)
    return out

def sign_matrix(size):
    if size not in (7, 8):
        raise ProtocolError("sign matrix size must be seven or eight")
    signs = np.asarray(tuple(itertools.product((-1.0, 1.0), repeat=size)), dtype=np.float64)
    signs.setflags(write=False)
    return signs

def two_sided_p(delta, signs, shift):
    centered = delta - shift
    observed = abs(float(np.mean(centered)))
    permuted = np.abs(np.mean(signs * centered[None, :], axis=1))
    return float(np.mean(permuted >= np.nextafter(observed, -np.inf)))

def invert_interval(delta, signs, alpha=0.05):
    values = np.asarray(delta, dtype=np.float64)
    observed_mean = float(np.mean(values))
    signed_mean = np.mean(signs * values[None, :], axis=1)
    sign_mean = np.mean(signs, axis=1)
    roots = []
    for numerator, denominator in ((observed_mean - signed_mean, 1.0 - sign_mean),
                                   (observed_mean + signed_mean, 1.0 + sign_mean)):
        valid = np.abs(denominator) > 0.0
        roots.extend((numerator[valid] / denominator[valid]).tolist())
    roots = np.unique(np.asarray(roots, dtype=np.float64))
    if roots.size == 0 or not np.isfinite(roots).all():
        raise ProtocolError("sign-flip inversion has no finite breakpoints")
    accepted = []
    for root in roots:
        if two_sided_p(values, signs, float(root)) >= alpha:
            accepted.append((float(root), float(root)))
    for lower, upper in zip(roots[:-1], roots[1:]):
        midpoint = float(lower + (upper - lower) / 2.0)
        if two_sided_p(values, signs, midpoint) >= alpha:
            accepted.append((float(lower), float(upper)))
    span = max(1.0, float(np.ptp(roots)))
    for outer in (float(roots[0] - span), float(roots[-1] + span)):
        if two_sided_p(values, signs, outer) >= alpha:
            raise ProtocolError("sign-flip interval is unbounded")
    if not accepted:
        raise ProtocolError("sign-flip acceptance set is empty")
    accepted.sort()
    merged = []
    for lower, upper in accepted:
        if not merged or lower > np.nextafter(merged[-1][1], np.inf):
            merged.append([lower, upper])
        else:
            merged[-1][1] = max(merged[-1][1], upper)
    if len(merged) != 1:
        raise ProtocolError("sign-flip acceptance set is disconnected")
    return float(merged[0][0]), float(merged[0][1])

@dataclass(frozen=True)
class ExactEvidence:
    delta: np.ndarray
    mean_delta: float
    p_one_sided: float
    ci95: tuple[float, float]
    positive: int
    passed: bool

def exact_evidence(delta):
    values = np.asarray(delta, dtype=np.float64)
    if values.shape not in ((7,), (8,)) or not np.isfinite(values).all():
        raise ProtocolError("seven or eight finite deltas required")
    signs = sign_matrix(values.size)
    observed = float(values.mean())
    permuted = np.mean(signs * values[None, :], axis=1)
    p_one = float(np.mean(permuted >= np.nextafter(observed, -np.inf)))
    interval = invert_interval(values, signs)
    return ExactEvidence(readonly(values), observed, p_one, interval,
                         int(np.sum(values > 0)),
                         observed > 0 and p_one < 0.05 and interval[0] > 0)
```

测试还要覆盖空 acceptance、非有限输入和 disconnected interval 的 `ProtocolError`。

- [ ] **Step 3：写并实现确定性 LOO power**

```python
@dataclass(frozen=True)
class PowerEvidence:
    omitted: int
    power: float
    lower95: float
    seed: int

def power_seed(outer_test, horizon, omitted):
    payload = bytearray("P0c-b-v1-power".encode("utf-8"))
    for value in (outer_test, horizon, omitted):
        payload.extend(struct.pack("<q", int(value)))
    return int.from_bytes(hashlib.sha256(bytes(payload)).digest()[:8], "little")

def leave_one_out_power(delta, outer_test, horizon, simulations=100000):
    centered = np.asarray(delta, dtype=np.float64) - np.mean(delta)
    signs = sign_matrix(8)
    output = []
    for omitted in range(7):
        pool = np.delete(centered, omitted)
        pool -= pool.mean()
        seed = power_seed(outer_test, horizon, omitted)
        rng = np.random.Generator(np.random.PCG64(seed))
        passed = 0
        for start in range(0, simulations, 2000):
            size = min(2000, simulations - start)
            sample = 0.05 + pool[rng.integers(0, 6, size=(size, 8))]
            means = sample.mean(axis=1)
            perm = np.einsum("sn,kn->sk", sample, signs, optimize=True) / 8
            p_one = np.mean(perm >= np.nextafter(means[:, None], -np.inf), axis=1)
            p_two = np.mean(np.abs(perm) >= np.nextafter(np.abs(means[:, None]), -np.inf), axis=1)
            passed += int(np.sum((means > 0) & (p_one < 0.05) & (p_two < 0.05)))
        power = passed / simulations
        lower = 0.0 if passed == 0 else float(beta.ppf(0.05, passed, simulations - passed + 1))
        output.append(PowerEvidence(omitted, power, lower, seed))
    return tuple(output)
```

Test uses `simulations=2000` twice and asserts逐项完全相同、七个遗漏齐全、`lower95<=power`。另用 100 组合
验证批量 `p_two<0.05` 与完整区间 `ci95[0]>0` 等价。

- [ ] **Step 4：运行 ExactPowerTest**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest tests.test_score_power.ExactPowerTest -v
```

Expected: 小规模纯合成测试 PASS；不运行真实全折 `B=100000`。

---

### Task 7：无手势标签 store、外层封存与整体 verdict

**Files:**
- Create: `tests/test_p0c_b_audit.py`
- Create: `uwb_osr/p0c_b_audit.py`

`p0c_b_audit.py` 文件头固定为：

```python
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from uwb_osr.causal_data import ProtocolError
from uwb_osr.range_likelihood import (build_width_library, calibrate_o1, fit_o1,
                                      o1_frame_mass, observed_future_mass, pulse_from_payload)
from uwb_osr.score_power import (exact_evidence, fit_matching_scale, inner_splits,
                                 leave_one_out_power, midpoint_oracle)
```

- [ ] **Step 1：写 constructor 无 label 参数与外层不读测试被试测试**

```python
def test_store_constructor_has_no_label_argument(self):
    self.assertEqual(tuple(inspect.signature(audit.SubjectEchoStore).parameters), ("x_cr", "subjects"))

def test_outer_fold_never_reads_test_subject(self):
    store = RecordingStore()
    audit.run_outer_fold(store, synthetic_pulse(), outer_test=8, simulations=200)
    self.assertNotIn(8, store.read_subjects)
```

- [ ] **Step 2：实现 label-free store 和只显式加载 CR/subject 的 loader**

```python
class SubjectEchoStore:
    def __init__(self, x_cr, subjects):
        if x_cr.shape[1:] != (3, 90, 189) or x_cr.dtype != np.float32:
            raise ProtocolError("x_cr shape/dtype mismatch")
        if subjects.shape != (x_cr.shape[0],) or subjects.dtype != np.int64:
            raise ProtocolError("subjects must be aligned int64")
        self._x_cr, self._subjects = x_cr, subjects

    def subject(self, subject_id):
        index = np.flatnonzero(self._subjects == subject_id)
        if index.size == 0:
            raise ProtocolError("subject has no rows")
        return index, self._x_cr[index]

def load_label_free_cache(cache_dir):
    x = np.load(Path(cache_dir) / "X_clutterremoved.npy", mmap_mode="r")
    subjects = np.load(Path(cache_dir) / "subj_clutterremoved.npy", mmap_mode="r")
    return SubjectEchoStore(x, subjects)
```

- [ ] **Step 3：写并实现 12 格/CRPS/power/任一折-H fail-closed verdict**

```python
def final_verdict(outer_bundles):
    if set(outer_bundles) != set(range(1, 9)):
        raise ProtocolError("all eight outer folds required")
    for outer in range(1, 9):
        for horizon in (5, 10, 20):
            cell = outer_bundles[outer]["horizons"][horizon]
            checks = (
                cell["exact"]["passed"],
                all(value > 0 for value in cell["frnll_cells"].values()),
                all(value > 0 for value in cell["crps_cells"].values()),
                min(item["lower95"] for item in cell["power"]) >= 0.80,
            )
            if not all(checks):
                return {"passed": False, "failed_outer_fold": outer, "failed_horizon": horizon}
    return {"passed": True, "failed_outer_fold": None, "failed_horizon": None}
```

Synthetic bundle test first PASS, then依次破坏一个 FRNLL 格、CRPS 格、exact、power、H 和 outer fold，均必须
整体 FAIL；Energy/Variogram 字段无论取值均不得改变 verdict。

- [ ] **Step 4：实现模板真实 loader，但本任务只做纯合成测试**

```python
def load_frozen_pulse(path, generator_path):
    file_sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    generator_sha = hashlib.sha256(Path(generator_path).read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as payload:
        mapping = {key: payload[key] for key in payload.files}
    return pulse_from_payload(mapping, file_sha, generator_sha)
```

- [ ] **Step 5：运行 audit 测试**

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest tests.test_p0c_b_audit -v
```

Expected: 封存、标签隔离、12 格、CRPS、整体 NO-GO 和 manifest schema 全部 PASS，无真实读取。

---

### Task 8：CLI 模式隔离、selftest 与静态禁区

**Files:**
- Create: `tools/w4_real_score_power_gate.py`
- Modify: `tests/test_p0c_b_audit.py`

CLI 文件头固定为：

```python
from __future__ import annotations
import argparse
import json
from pathlib import Path
from uwb_osr.causal_data import ProtocolError
from uwb_osr.p0c_b_audit import run_real_audit, run_synthetic_selftest
```

- [ ] **Step 1：写 CLI 互斥和缺路径拒绝测试**

```python
def test_selftest_needs_no_paths(self):
    code, payload = run_cli("--selftest")
    self.assertEqual((code, payload["mode"], payload["real_signal_reads"]),
                     (0, "synthetic-selftest", 0))

def test_real_mode_requires_all_paths(self):
    code, payload = run_cli("--audit-real")
    self.assertNotEqual(code, 0)
    self.assertEqual(payload["status"], "FAIL")
```

- [ ] **Step 2：实现显式互斥 parser**

```python
def build_parser():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--selftest", action="store_true")
    mode.add_argument("--audit-real", action="store_true")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--pulse", type=Path)
    parser.add_argument("--pulse-generator", type=Path)
    parser.add_argument("--out-dir", type=Path)
    return parser

def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.selftest:
        payload = run_synthetic_selftest()
    else:
        if any(value is None for value in (args.cache, args.pulse, args.pulse_generator, args.out_dir)):
            raise ProtocolError("real mode requires four explicit paths")
        payload = run_real_audit(args.cache, args.pulse, args.pulse_generator, args.out_dir)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0
```

- [ ] **Step 3：写生产源码禁区测试**

```python
def test_production_sources_have_no_forbidden_dependency(self):
    paths = ("uwb_osr/range_likelihood.py", "uwb_osr/score_power.py",
             "uwb_osr/p0c_b_audit.py", "tools/w4_real_score_power_gate.py")
    source = "\n".join(Path(path).read_text(encoding="utf-8") for path in paths)
    for forbidden in ("y_clutterremoved", "gesture_eval", "traj_w1", "w3_",
                      "causal_baselines", "--track", "--analyze"):
        self.assertNotIn(forbidden, source)
```

同一测试构造任意 O0、fixed-offset、Energy 和 Variogram 数值，逐项改写后 `final_verdict` 必须逐字节不变。

- [ ] **Step 4：两次纯合成 selftest 逐字节比较**

```powershell
$a = & "D:\D_conda\envs\pytorch\python.exe" tools\w4_real_score_power_gate.py --selftest
if ($LASTEXITCODE -ne 0) { throw "first selftest failed" }
$b = & "D:\D_conda\envs\pytorch\python.exe" tools\w4_real_score_power_gate.py --selftest
if ($LASTEXITCODE -ne 0) { throw "second selftest failed" }
if ($a -cne $b) { throw "selftest differs" }
$a
```

Expected: 单行确定性 JSON，含 `synthetic_only=true`、`real_signal_reads=0` 和三段式边界句。

---

### Task 9：全回归、静态自审与停止点

**Files:**
- Test: 七个 P0c-b 新文件
- Read only hash check: 九个冻结文件
- Modify only after verification: P0c-b design/plan 与 `proposal_v2/15_...md` 状态文字

- [ ] **Step 1：运行三个 P0c-b 测试文件**

```powershell
foreach ($name in @("test_range_likelihood.py", "test_score_power.py", "test_p0c_b_audit.py")) {
  & "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p $name -v
  if ($LASTEXITCODE -ne 0) { throw "P0c-b test failed: $name" }
}
```

Expected: 全部 PASS，0 failure/error/skip。

- [ ] **Step 2：运行 P0a/P0b/P0c-a 回归**

```powershell
foreach ($name in @("test_causal_forecast_protocol.py", "test_causal_baselines.py", "test_forecast_scores.py")) {
  & "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p $name -v
  if ($LASTEXITCODE -ne 0) { throw "regression failed: $name" }
}
```

Expected: 三个既有测试文件全部 PASS，0 failure/error/skip。

- [ ] **Step 3：复核哈希、占位符、diff 与 staging**

```powershell
rg -n "TBD|TODO|FIXME|implement later|SKIPPED|PARTIAL PASS" `
  uwb_osr/range_likelihood.py uwb_osr/score_power.py uwb_osr/p0c_b_audit.py `
  tests/test_range_likelihood.py tests/test_score_power.py tests/test_p0c_b_audit.py `
  tools/w4_real_score_power_gate.py
if ($LASTEXITCODE -eq 0) { throw "placeholder or partial pass found" }
git diff --check
git diff --cached --name-only
```

Expected: 全部无输出；九个冻结 SHA-256 仍与 Task 0 相同。

- [ ] **Step 4：只记录 contract complete，不写真实 PASS**

三个状态文档只加入：

```markdown
P0c-b-v1 implementation contract complete：模板/O1/嵌套拆分/受控臂/7–8 人统计与 CLI 的纯合成测试通过；
未读取真实信号、未运行 `--audit-real`，因此 P0c-b real-score power 仍为 NOT RUN，轨迹可预测性仍未知。
```

- [ ] **Step 5：输出七文件 SHA、状态并停止**

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath @(
  "uwb_osr/range_likelihood.py", "uwb_osr/score_power.py", "uwb_osr/p0c_b_audit.py",
  "tests/test_range_likelihood.py", "tests/test_score_power.py", "tests/test_p0c_b_audit.py",
  "tools/w4_real_score_power_gate.py")
git status --short
git diff --cached --name-only
```

Expected: 七个 SHA-256；staged 为空。等待用户复核，不自动运行 `--audit-real`。

---

## 设计覆盖索引

| 设计要求 | 任务 |
|---|---|
| 固定模板 SHA/元数据、七宽度、幅度 profile | 0–2 |
| 五拟合被试 `sigma_v/p(s)`、一个全局 outlier、O1 mass | 2–3 |
| 7 次平衡循环、外层封存、不池化 56 | 4、7 |
| midpoint `K=20`、同被试 donor、五帧速度 | 4 |
| 三个 `alpha_H`、四 prefix 等权、4.75%–5.25% | 5 |
| 7 人 exact p/CI、12 格、CRPS | 5–7 |
| 中心化严格 5%、`n=8`、七次 LOO、CP 下界 | 6 |
| Energy/Variogram、O0/O2 不进门；任一折/H 整体 NO-GO | 7–8 |
| Top `+3*BIN_M` 固定偏移只作非决定敏感性 | 5、8 |
| selftest 与真实读取分离；不提交、不自动实跑 | 8–9 |
