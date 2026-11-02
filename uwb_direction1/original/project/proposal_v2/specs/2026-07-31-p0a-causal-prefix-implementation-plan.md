# P0a 因果前缀数据接口实施计划

> **For Codex:** 实施时必须使用 `superpowers:test-driven-development`，逐任务先看到预期
> 失败再写最小代码；完成前必须使用 `superpowers:verification-before-completion`。若在后续
> 独立执行轮次运行本计划，先使用 `superpowers:executing-plans`。

**目标**：实现 P0a-v1 的因果前缀数据接口、训练折固定归一化、输入/目标隔离、纯合成 C0 自检和
实际缓存只读审计。  
**架构**：新建独立的 `uwb_osr/causal_data.py`，不复用现有完整 90 帧
`MVDataset` 或 W1 smoother；测试与缓存审计分别由 `unittest` 和轻量 CLI 驱动。  
**技术栈**：Python 3、NumPy、标准库 `dataclasses/hashlib/struct/pathlib/argparse/unittest`；
P0a 不使用 PyTorch、SciPy 或 GPU。  
**上位设计**：`2026-07-31-p0a-causal-prefix-design.md`（P0a-v1，已由用户复核冻结）  
**状态**：用户已批准并执行；Task 0–7 验证门通过，Task 8 证据已回写，用户实施证据复核已通过；P0a 完成  

---

## 0. 全程边界

本计划只允许修改：

- 新建 `uwb_osr/causal_data.py`；
- 新建 `tests/test_causal_forecast_protocol.py`；
- 新建 `tools/w4_causal_gate.py`；
- 验证成功后更新 15 和 P0a 设计文件的实施状态。

不得修改：

- `uwb_osr/data.py`；
- `uwb_osr/trajectory.py`；
- `uwb_osr/forward.py`；
- 块一代码；
- `17`、`18`、`tools/w3_readout_predict.py`。

不得运行：

- `w3_readout_predict.py --track`；
- `w3_readout_predict.py --analyze`；
- 任何训练、F0、Last/CV/CA、W1 重算或 FRLS 评测。

本计划不提交 Git。当前研究资产存在既有未跟踪/未提交内容，实施必须保留它们；不得 reset、checkout、
clean 或把无关变化纳入 P0a。

---

## Task 0：实施前安全快照

**文件**：不修改。

### Step 1：确认工作树和权威文件

从 `D:\uwb_gestures` 执行：

```powershell
git status --short
Get-FileHash -Algorithm SHA256 proposal_v2\17_读出失败预测_预注册.md,proposal_v2\18_通道D整数bin相位修正.md,tools\w3_readout_predict.py
Test-Path logs\w3_readout_scores.npz
```

预期基线：

- 17：`fe35d845bb7feeb62050a59423e4cf1088b056440b10822b27bfbc6595e1f32f`；
- 18：`2335e804440eb7b34416543d647b42f662a0e769b4a6427ec2ecba79e8adf7be`；
- W3：`5e597a6cdba426ccebceb6609cfbcccec4e680487b32f981bb43dbe83664f4f90`；
- 分数文件：不存在。

若任一项已改变，停止并先解释变化来源，不覆盖新状态。

### Step 2：确认解释器

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -c "import sys,numpy; print(sys.executable); print(numpy.__version__)"
```

该环境只作为仓库统一解释器；P0a 代码不得 import torch。

### Step 3：保存现有 Python 测试基线

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_*.py" -v
```

若存在实施前失败，记录具体测试和错误；P0a 不能把既有失败冒充回归，也不能顺手修改无关模块。

---

## Task 1：协议常量、键与 LOSO

**文件**

- Create：`tests/test_causal_forecast_protocol.py`
- Create：`uwb_osr/causal_data.py`

### Step 1：先写失败测试

测试类 `ProtocolPrimitiveTest` 至少包含：

1. `test_registered_constants`：
   - `VIEW_ORDER == ("Left","Top","Right")`；
   - `PREFIXES == (20,30,45,60)`；
   - `HORIZONS == (5,10,20)`；
   - `N_FRAMES == 90`、`N_BINS == 189`、`FPS == 20.0`。
2. `test_key_grid_order`：
   - 两条序列产生 24 个键；
   - 前 12 个键全部属于 `sample_idx=0`；
   - 顺序严格为 sample → prefix → horizon。
3. `test_key_validation_rejects_bad_values`：
   - 负 sample；
   - 未注册 $t/H$；
   - $t+H>90$；
   - 非整数；
   均抛 `ProtocolError`。
4. `test_fold_rotation_exact`：八折 val/cal/train 被试与设计 §4 完全一致。
5. `test_split_indices_disjoint_complete`：合成 subjects 上四组索引两两不交、并集完整。
6. `test_split_does_not_use_gesture_labels`：同一 subjects、两份任意置换标签得到相同切分。

建议导入接口：

```python
from uwb_osr.causal_data import (
    FPS, HORIZONS, N_BINS, N_FRAMES, PREFIXES, VIEW_ORDER,
    FoldAssignment, ForecastKey, ProtocolError,
    build_fold_assignment, iter_forecast_keys, split_indices,
)
```

### Step 2：运行并确认 RED

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_forecast_protocol.py" -k ProtocolPrimitiveTest -v
```

预期失败：`ModuleNotFoundError: uwb_osr.causal_data`。若测试意外通过，先调查是否存在同名
旧模块，不继续写代码。

### Step 3：写最小实现

在 `uwb_osr/causal_data.py` 新增：

```python
PROTOCOL_VERSION = "P0a-v1"
VIEW_ORDER = ("Left", "Top", "Right")
N_VIEWS = 3
N_FRAMES = 90
N_BINS = 189
FPS = 20.0
PREFIXES = (20, 30, 45, 60)
HORIZONS = (5, 10, 20)

class ProtocolError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class ForecastKey:
    sample_idx: int
    prefix_t: int
    horizon: int

@dataclass(frozen=True, slots=True)
class FoldAssignment:
    test_subject: int
    val_subject: int
    cal_subject: int
    train_subjects: tuple[int, ...]
```

实现：

- `validate_key(key, n_samples=None)`；
- `iter_forecast_keys(n_samples)`：生成器，不建立 115116 个数组副本；
- `forecast_key_count(n_samples)`；
- `build_fold_assignment(test_subject)`；
- `split_indices(subjects, fold)`。

所有入口先校验 dtype/shape/取值。subjects 只允许一维整数数组且值域 1–8；切分结果使用
`np.flatnonzero`，按原索引升序。

### Step 4：运行 GREEN

重复 Step 2，预期该测试类全部 PASS。

---

## Task 2：slow-time 局部表征

**文件**

- Modify：`tests/test_causal_forecast_protocol.py`
- Modify：`uwb_osr/causal_data.py`

### Step 1：先写失败测试

测试类 `RepresentationTest`：

1. `test_signed_is_float64_copy`；
2. `test_envelope_matches_manual_analytic_transform`；
3. `test_log_power_definition`：严格等于
   `np.log1p(envelope ** 2)`；
4. `test_representation_is_slow_time_local`：
   - 构造两条 shape `(3,90,189)` 序列；
   - 前 45 帧完全相同，后 45 帧完全不同；
   - 三种表征的前 45 帧逐位相同；
5. `test_bad_representation_and_shape_fail_closed`。

### Step 2：运行 RED

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_forecast_protocol.py" -k RepresentationTest -v
```

预期失败：缺少 `transform_representation`。

### Step 3：写最小实现

```python
def transform_representation(x: np.ndarray, representation: str) -> np.ndarray:
    ...
```

要求：

- 输入至少二维，最后一轴必须是 189；
- 首先显式转换为 float64；
- analytic/Hilbert 只沿 `axis=-1`；
- `signed` 返回 float64 实值副本；
- `envelope` 返回解析信号绝对值；
- `log_power` 返回 `np.log1p(envelope ** 2)`；
- 不在 slow-time 轴求均值、标准差或 FFT；
- 不 import `uwb_osr.data` 或 `uwb_osr.trajectory`。

Hilbert 权重按奇偶长度通用实现，但协议入口仍要求 189 bins。

### Step 4：运行 GREEN

重复 Step 2，预期全部 PASS。

---

## Task 3：训练被试等权 FoldStats

**文件**

- Modify：`tests/test_causal_forecast_protocol.py`
- Modify：`uwb_osr/causal_data.py`

### Step 1：先写失败测试

测试类 `FoldStatsTest`：

1. `test_subject_balanced_median_mad_exact`：
   - 使用 8 名合成被试、每人不同数量序列；
   - signed 表征；
   - 手工计算“序列 median/MAD → 被试 median → 5 被试 median”；
   - 与实现逐位比较。
2. `test_nontrain_changes_do_not_change_stats`：
   - 只改 val/cal/test 为极大值；
   - center、scale、provenance 必须逐位不变。
3. `test_representation_has_separate_stats`：signed/envelope/log_power 的 provenance 不同。
4. `test_stats_are_float64_read_only`。
5. `test_degenerate_or_nonfinite_stats_fail`：零 MAD、NaN、Inf 均硬失败。
6. `test_provenance_is_deterministic_and_binary`：
   - 同输入同哈希；
   - 改 fold、表征、center 或 scale 任一字节，哈希改变。
7. `test_chunk_size_does_not_change_stats`：chunk 1 与 chunk 4 逐位相同。

为保证测试快，精确公式测试可使用 shape `(N,3,4,189)` 的内部数组。
`fit_fold_stats` 允许任意正 slow-time 长度，因为公式本身与长度无关；生产
`CausalPrefixStore` 和真实缓存审计分别独立强制 90 帧。不得为测试增加公开的
`n_frames` 绕过参数。

### Step 2：运行 RED

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_forecast_protocol.py" -k FoldStatsTest -v
```

### Step 3：写最小实现

```python
@dataclass(frozen=True, slots=True)
class FoldStats:
    fold: FoldAssignment
    representation: str
    center: np.ndarray
    scale: np.ndarray
    source_shape: tuple[int, ...]
    source_dtype: str
    provenance: str

def fit_fold_stats(
    x: np.ndarray,
    subjects: np.ndarray,
    fold: FoldAssignment,
    representation: str = "envelope",
    *,
    chunk_size: int = 16,
) -> FoldStats:
    ...
```

实现算法：

1. 只取 `subjects in fold.train_subjects` 的索引；
2. 每 chunk 调用 `transform_representation`；
3. 每条序列、每视角计算 median 与 MAD，保存 shape `(N_train,3)` 小数组；
4. 每名训练被试内部取中位数；
5. 5 名训练被试之间取中位数；
6. 加 epsilon 前断言 raw MAD 有限且严格大于 0；
7. `scale = 1.4826 * raw_mad + 1e-6`；
8. center/scale 强制 little-endian float64、复制并设 `writeable=False`；
9. 生成设计 §6.2 冻结的 SHA-256 provenance。

provenance 使用长度前缀编码文本、固定字节序整数与
`center.astype("<f8").tobytes()` / `scale.astype("<f8").tobytes()`；禁止哈希
浮点十进制字符串。

再新增：

```python
def normalize_prefix(z: np.ndarray, stats: FoldStats) -> np.ndarray:
    ...

def normalize_running_prefix(z: np.ndarray) -> np.ndarray:
    ...
```

两者输出 float32、独立只读数组。running-prefix 只在传入 prefix 上统计，不接收完整序列或 $t$。

### Step 4：运行 GREEN

重复 Step 2。

---

## Task 4：输入与目标物理隔离

**文件**

- Modify：`tests/test_causal_forecast_protocol.py`
- Modify：`uwb_osr/causal_data.py`

### Step 1：先写失败测试

测试类 `StoreIsolationTest`：

1. `test_forecast_input_has_no_future_or_label_field`；
2. `test_forecast_target_is_separate_type`；
3. `test_prefix_and_future_slices_exact`：哨兵帧号验证无 off-by-one；
4. `test_future_replacement_keeps_input_bytes`；
5. `test_future_replacement_changes_target_only`；
6. `test_label_permutation_keeps_input`；
7. `test_arrays_are_read_only`；
8. `test_store_rejects_wrong_shape_dtype_or_key`；
9. `test_geometry_comes_from_config_view_order`。

### Step 2：运行 RED

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_forecast_protocol.py" -k StoreIsolationTest -v
```

### Step 3：写最小实现

```python
@dataclass(frozen=True, slots=True)
class ForecastInput:
    key: ForecastKey
    prefix: np.ndarray
    radar_positions: np.ndarray
    valid_view_mask: np.ndarray
    norm_provenance: str

@dataclass(frozen=True, slots=True)
class ForecastTarget:
    key: ForecastKey
    future_cr: np.ndarray
    future_raw: np.ndarray | None
    subject: int
    gesture_eval: int

class CausalPrefixStore:
    def __init__(self, x_cr, stats: FoldStats): ...
    def get(self, key: ForecastKey) -> ForecastInput: ...

class FutureTargetStore:
    def __init__(self, x_cr, y, subjects, x_raw=None): ...
    def get(self, key: ForecastKey) -> ForecastTarget: ...
```

关键实现顺序：

```python
raw_prefix = np.asarray(x_cr[key.sample_idx, :, :key.prefix_t, :])
z_prefix = transform_representation(raw_prefix, stats.representation)
prefix = normalize_prefix(z_prefix, stats)
```

禁止先变换完整 90 帧再切 prefix。`CausalPrefixStore` 不接收 y、future store 或 X_raw。

geometry 只从 `config.VIEW_ORDER` 与 `config.VIEW_POSITIONS` 构造并补 z=0，
不 import `trajectory.py`，避免把 SciPy/W1 依赖带进 P0a。

### Step 4：运行 GREEN

重复 Step 2。

---

## Task 5：随机键与端到端 C0 辅助

**文件**

- Modify：`tests/test_causal_forecast_protocol.py`
- Modify：`uwb_osr/causal_data.py`

### Step 1：先写失败测试

测试类 `RNGContractTest`：

1. `test_same_registered_fields_same_seed`；
2. `test_each_registered_field_changes_seed`；
3. `test_future_content_cannot_change_seed`；
4. `test_same_key_generates_identical_samples`；
5. `test_future_replacement_end_to_end_reference_predictor`：
   使用只消费 `ForecastInput` 与 RNG 的参考预测器，两份不同未来得到逐位相同输出。

### Step 2：运行 RED

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_forecast_protocol.py" -k RNGContractTest -v
```

### Step 3：写最小实现

```python
def forecast_seed(
    outer_fold: int,
    key: ForecastKey,
    model_seed: int,
    *,
    protocol_version: str = PROTOCOL_VERSION,
) -> int:
    ...

def forecast_rng(...) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(forecast_seed(...)))
```

哈希字段顺序固定为：

```text
(protocol_version, outer_fold, sample_idx, prefix_t, horizon, model_seed)
```

用 SHA-256 前 8 字节 little-endian 无符号整数作为 PCG64 seed。API 不接受 future、label、batch index
或对象地址。

### Step 4：运行 GREEN

重复 Step 2。

---

## Task 6：缓存完整性审计与 CLI

**文件**

- Modify：`tests/test_causal_forecast_protocol.py`
- Modify：`uwb_osr/causal_data.py`
- Create：`tools/w4_causal_gate.py`

### Step 1：先写失败测试

测试类 `CacheAuditTest` 使用临时小型 `.npy` 夹具，不读取真实缓存。实现中定义
内部不可导出的 `CacheSpec`；测试调用内部 `_audit_cache_with_spec` 配小规格，
公开 `audit_cache` 永远使用冻结的 `P0A_CACHE_SPEC`，CLI 不提供覆盖入口：

1. `test_cache_audit_accepts_valid_fixture`；
2. `test_cache_audit_uses_memmap_for_x`；
3. `test_cache_audit_rejects_missing_file`；
4. `test_cache_audit_rejects_shape_or_dtype`；
5. `test_cache_audit_rejects_raw_cr_metadata_mismatch`；
6. `test_cache_audit_rejects_wrong_subject_gesture_counts`；
7. `test_manifest_does_not_claim_physical_event_pairing`；
8. `test_synthetic_selftest_does_not_read_cache`；
9. `test_cli_modes_are_mutually_exclusive`；
10. `test_cli_failure_has_no_skipped_status`；
11. `test_default_cache_spec_is_read_only`。

临时文件由 `tempfile` 管理；不得写仓库 cache/logs。

### Step 2：运行 RED

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_forecast_protocol.py" -k CacheAuditTest -v
```

### Step 3：实现审计函数

在 `causal_data.py` 新增：

```python
def audit_cache(cache_dir: Path) -> dict[str, object]:
    ...

def run_synthetic_selftest() -> dict[str, object]:
    ...
```

内部结构：

```python
@dataclass(frozen=True, slots=True)
class _CacheSpec:
    shape: tuple[int, int, int, int]
    x_dtype: str
    meta_dtype: str
    expected_count_table: np.ndarray

_EXPECTED_COUNT_TABLE = np.full((8, 12), 100, dtype=np.int64)
_EXPECTED_COUNT_TABLE[5, 3] = 93
_EXPECTED_COUNT_TABLE.setflags(write=False)

P0A_CACHE_SPEC = _CacheSpec(
    shape=(9593, 3, 90, 189),
    x_dtype="float32",
    meta_dtype="int64",
    expected_count_table=_EXPECTED_COUNT_TABLE,
)

def _audit_cache_with_spec(cache_dir: Path, spec: _CacheSpec) -> dict[str, object]:
    ...
```

`_CacheSpec`、`_audit_cache_with_spec` 不从模块 `__all__` 导出；只有测试可
直接访问。`audit_cache(cache_dir)` 必须无条件调用冻结的 `P0A_CACHE_SPEC`。

`audit_cache` 必须：

- 六个文件缺一即 `ProtocolError`；
- X 用 `np.load(..., mmap_mode="r")`；
- 核对冻结 shape/dtype；
- 小型 y/subj 才允许完整载入；
- 核对 Raw/CR y/subj；
- 核对 HV6-G4=93 且只有这一格不为 100；
- 核对八折样本数；
- 核对 115116 个键的计数公式和边界，不物化全部键；
- manifest 使用
  `"raw_cr_index_metadata_equal": true` 和
  `"physical_event_pairing_scope": "not audited in P0a"`；
- 不读取 X 的信号主体，不计算轨迹或分数。

`run_synthetic_selftest` 使用内存合成数据覆盖键、LOSO、future replacement、nontrain stats、
label blindness、RNG；返回无时间戳、键排序固定的 manifest。

### Step 4：实现 CLI

`tools/w4_causal_gate.py`：

- 从 `Path(__file__).resolve().parents[1]` 定位 repo；
- 使用 mutually exclusive required group：`--selftest` 或 `--audit-cache`；
- `--cache` 默认 repo `cache`，只供 audit；
- PASS 在 stdout 输出排序、确定性的 JSON；
- `ProtocolError` 在 stderr 输出
  `{"gate":"P0a","status":"FAIL",...}` 并 exit 1；
- 不出现 `SKIPPED`；
- 不写文件。

### Step 5：运行 GREEN

重复 Step 2，然后运行：

```powershell
& "D:\D_conda\envs\pytorch\python.exe" tools\w4_causal_gate.py --selftest
```

此时只允许 synthetic PASS；尚不运行 `--audit-cache`。

---

## Task 7：P0a 验证门

**文件**：不先修改文档。

### Step 1：运行完整 P0a 单元测试

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_forecast_protocol.py" -v
```

要求：全部 PASS，0 failure、0 error、0 skip。

### Step 2：证明 selftest 可重复

连续两次运行：

```powershell
& "D:\D_conda\envs\pytorch\python.exe" tools\w4_causal_gate.py --selftest
& "D:\D_conda\envs\pytorch\python.exe" tools\w4_causal_gate.py --selftest
```

捕获 stdout 并逐字节比较；必须完全一致，且只含 PASS manifest，不含时间戳、路径抖动或
`SKIPPED`。

### Step 3：静态边界检查

```powershell
rg -n "MVDataset|viterbi_track|fit_trajectory|torch|scipy" uwb_osr\causal_data.py tests\test_causal_forecast_protocol.py tools\w4_causal_gate.py
```

生产模块与 CLI 中预期无匹配。测试说明文字若有匹配，必须确认只是禁止项断言。

### Step 4：首次真实缓存只读审计

只有 Steps 1–3 全过后才运行：

```powershell
& "D:\D_conda\envs\pytorch\python.exe" tools\w4_causal_gate.py --audit-cache --cache "D:\uwb_gestures\cache"
```

要求：

- exit 0；
- shape/dtype、9593、HV6-G4=93、八折计数和 115116 全部吻合；
- 输出明确写“physical event pairing not audited in P0a”；
- 无 score/trajectory/cache 派生文件。

### Step 5：回归现有 Python 测试

重复 Task 0 Step 3。与实施前基线比较；任何新增失败必须修复 P0a，而不是改无关测试。

### Step 6：重新核对禁止边界

重新计算 17、18、W3 哈希并执行：

```powershell
Test-Path logs\w3_readout_scores.npz
git status --short
```

要求：

- 三个哈希保持 Task 0 基线；
- 分数文件仍不存在；
- 只有计划允许的 P0a 文件和既有用户变化；
- 没有 Git commit/stage。

---

## Task 8：证据回写与停止点

**文件**

- Modify：`proposal_v2/15_块二重构_因果状态估计_未来轨迹预测_未来回波生成.md`
- Modify：`proposal_v2/specs/2026-07-31-p0a-causal-prefix-design.md`

### Step 1：仅在 Task 7 全过后更新状态

记录：

- 实现文件 SHA-256；
- 单元测试数量、PASS/FAIL/SKIP；
- 两次 selftest 输出是否逐字节一致；
- cache audit 冻结计数；
- 执行命令；
- 17/18/W3 哈希与 scores 不存在；
- 明确“P0a C0 PASS 不是轨迹预测 PASS”。

若 Task 7 任一项失败，只记录“P0a 实施中/FAIL”和具体失败，不得写 PASS。

### Step 2：验证文档一致性

```powershell
rg -n "P0a|C0|115116|CA-2|CV\(2/5/20\)|CA\(5/20\)" proposal_v2\15_块二重构_因果状态估计_未来轨迹预测_未来回波生成.md proposal_v2\specs\2026-07-31-p0a-causal-prefix-design.md proposal_v2\specs\2026-07-31-p0a-causal-prefix-implementation-plan.md
```

要求三份文件口径一致。

### Step 3：在 P0a 停止

向用户交付：

- 新建代码与测试；
- C0/selftest/cache audit 的确切证据；
- 尚未执行的 P0b；
- 下一步 P0b 需要另行冻结的 F0/Last/CV/CA 条款。

用户已于 2026-07-31 复核通过 P0a 结果。由此解除的仅是 **P0b 设计门**：可以开始 P0b 的设计与
预注册；在 P0b 书面设计再次获得用户批准前：

- 不实现 P0b；
- 不训练模型；
- 不运行正式轨迹预测；
- 不运行 W3；
- 不冻结提交 18。

---

## 计划完成定义

本计划的“完成”只指：

1. P0a-v1 接口按设计实现；
2. 所有纯合成 C0 测试通过且无 skip；
3. selftest 两次逐字节相同；
4. 实际缓存只读审计通过；
5. 禁止边界保持；
6. 文档准确回写；
7. 停在 P0b 设计之前。

它不包含 2.1/2.2 成立、轨迹可预测、模型胜过 CV 或论文结论。
