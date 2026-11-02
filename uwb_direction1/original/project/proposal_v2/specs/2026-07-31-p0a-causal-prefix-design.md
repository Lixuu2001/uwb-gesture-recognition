# P0a 因果前缀数据接口与 C0 泄漏审计设计

**日期**：2026-07-31  
**版本**：P0a-v1  
**状态**：设计与实施计划均已获用户批准；P0a-v1 已实现，Task 7 验证门与用户实施证据复核均已通过  
**上位方案**：`../15_块二重构_因果状态估计_未来轨迹预测_未来回波生成.md`  
**实施计划**：`2026-07-31-p0a-causal-prefix-implementation-plan.md`（已批准、执行并通过证据复核；P0b 另行设计）  
**范围**：只冻结因果数据接口、LOSO、归一化、输入/目标隔离和 C0 测试；不实现状态估计或轨迹模型  

---

## 1. 目标和非目标

P0a 只回答一个问题：

> 后续状态估计与未来轨迹预测代码，是否能够在结构上保证只看到 $X_{1:t}$，并且不能通过归一化、
> 标签、切分、随机种子或评测目标间接看到 $X_{t+1:90}$？

P0a 的交付物：

1. 严格的外层 LOSO 被试划分；
2. 惰性的“前缀—未来”键；
3. 只由训练被试拟合的固定归一化；
4. 模型输入与评测目标的独立对象和独立读取路径；
5. 纯合成单元测试与实际缓存只读完整性审计；
6. 后续每个状态估计器/预测器必须复用的端到端 C0 契约。

P0a 明确不做：

- 不生成或重算 W1 离线伪轨迹；
- 不实现 F0/F1/F2；
- 不实现 Last/CV/CA；
- 不训练神经网络；
- 不计算 FRLS/FRNLL、ADE/FDE 或任何模型排名；
- 不运行 `w3_readout_predict.py --track` / `--analyze`；
- 不冻结或提交 18。

因此，P0a 通过只表示“因果数据地基可用”，不表示轨迹可预测，也不表示研究点 2.1/2.2 PASS。

---

## 2. 已核实的当前数据事实

只读核查结果：

| 文件 | shape | dtype |
|---|---:|---|
| `X_clutterremoved.npy` | $(9593,3,90,189)$ | `float32` |
| `X_raw.npy` | $(9593,3,90,189)$ | `float32` |
| 两套 `y` | $(9593,)$ | `int64` |
| 两套 `subj` | $(9593,)$ | `int64` |

Raw 与 ClutterRemoved 的 `y`、`subj` 当前逐项一致。缓存构建顺序固定为
participant → gesture → view，视角顺序为 Left、Top、Right。

这只能证明两个缓存的**索引协议一致**；仅凭重复出现的被试/手势标签，不能独立证明每一行 Raw 与
ClutterRemoved 必然来自同一次物理重复。P0a 主输入和 C0 只使用 ClutterRemoved，因此该边界不阻塞
P0a；以后若联合使用同一行 Raw/ClutterRemoved 做完整 CIR 闭环，必须另做源文件行序或信号关系审计，
不能把本次标签一致性检查写成物理事件配对证据。

样本计数：

- 8 名被试、12 个手势；
- 除 HV6-G4 为 93 条外，其余“被试×手势”单元均为 100 条；
- HV6 合计 1193 条，其余被试各 1200 条；
- G4 合计 793 条，其余手势各 800 条。

接口不得假设每名被试固定 1200 条，也不得按固定行区间推断被试。

现有资产的边界：

- `uwb_osr/data.py::MVDataset` 使用单条完整 90 帧计算 per-view mean/std，不能用于因果主线；
- 现有 Doppler 预处理先在完整时间轴减均值并按完整时间轴标准化，也不能用于因果主线；
- `uwb_osr/trajectory.py::viterbi_track` 是完整输入段上的动态规划回溯；现有完整 90 帧结果
  不能成为时刻 $t$ 的状态；
- `logs/traj_w1.npz` 只含 1920 条样本的汇总读数，没有逐帧三维轨迹。

---

## 3. 固定常量与索引语义

固定常量：

```text
VIEW_ORDER = ("Left", "Top", "Right")
N_VIEWS = 3
N_FRAMES = 90
N_BINS = 189
FPS = 20.0
PREFIXES = (20, 30, 45, 60)
HORIZONS = (5, 10, 20)
```

对一个 shape 为 `(3,90,189)` 的样本，$X_{1:t}$ 与 $X_{t+1:t+H}$ 冻结为：

```python
prefix = x[:, :t, :]
future = x[:, t:t + H, :]
```

这里 $t$ 是“已观测帧数”，不是零起始帧号。四个 $t$ 与三个 $H$ 的 12 个组合均满足
$t+H\le90$。

评测键总数：

$$
9593\times4\times3=115116.
$$

只建立惰性 `ForecastKey`，不预展开回波数组。固定遍历顺序为：

1. `sample_idx` 升序；
2. `prefix_t` 按 20、30、45、60；
3. `horizon` 按 5、10、20。

训练阶段以后可以每 epoch、每序列随机抽一个合法 $(t,H)$，但随机抽样器属于训练适配层，不改变
P0a 的完整键空间。

---

## 4. 外层 LOSO

测试被试为 $u\in\{1,\ldots,8\}$ 时：

$$
u_{\mathrm{val}}=u\bmod8+1,\qquad
u_{\mathrm{cal}}=(u+1)\bmod8+1.
$$

其余 5 名被试训练。实际样本数冻结为完整性断言：

| test | val | cal | train subjects | train n | val n | cal n | test n |
|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 2 | 3 | 4,5,6,7,8 | 5993 | 1200 | 1200 | 1200 |
| 2 | 3 | 4 | 1,5,6,7,8 | 5993 | 1200 | 1200 | 1200 |
| 3 | 4 | 5 | 1,2,6,7,8 | 5993 | 1200 | 1200 | 1200 |
| 4 | 5 | 6 | 1,2,3,7,8 | 6000 | 1200 | 1193 | 1200 |
| 5 | 6 | 7 | 1,2,3,4,8 | 6000 | 1193 | 1200 | 1200 |
| 6 | 7 | 8 | 1,2,3,4,5 | 6000 | 1200 | 1200 | 1193 |
| 7 | 8 | 1 | 2,3,4,5,6 | 5993 | 1200 | 1200 | 1200 |
| 8 | 1 | 2 | 3,4,5,6,7 | 5993 | 1200 | 1200 | 1200 |

每折必须断言：

- train、val、cal、test 的被试集合两两不交；
- 四组索引并集恰好覆盖 9593 条；
- 每个索引恰好出现一次；
- 不使用手势标签构造外层划分；
- test 不参与归一化、表征标定、阈值、早停或随机过程参数估计。

---

## 5. 数据对象与权限边界

以下不可变数据对象已按冻结字段实现；后续模型不得扩展 `ForecastInput` 以加入标签或未来信息。

```python
@dataclass(frozen=True)
class ForecastKey:
    sample_idx: int
    prefix_t: int
    horizon: int

@dataclass(frozen=True)
class FoldAssignment:
    test_subject: int
    val_subject: int
    cal_subject: int
    train_subjects: tuple[int, ...]

@dataclass(frozen=True)
class FoldStats:
    fold: FoldAssignment
    representation: str
    center: np.ndarray       # (3,)
    scale: np.ndarray        # (3,)
    source_shape: tuple[int, ...]
    source_dtype: str
    provenance: str

@dataclass(frozen=True)
class ForecastInput:
    key: ForecastKey
    prefix: np.ndarray       # (3,t,189), read-only
    radar_positions: np.ndarray
    valid_view_mask: np.ndarray
    norm_provenance: str

@dataclass(frozen=True)
class ForecastTarget:
    key: ForecastKey
    future_cr: np.ndarray    # (3,H,189), evaluator-only
    future_raw: np.ndarray | None
    subject: int
    gesture_eval: int
```

结构要求：

1. `CausalPrefixStore` 只能生成 `ForecastInput`；
2. `FutureTargetStore` 单独生成 `ForecastTarget`；
3. 推理器签名只允许 `predict(ForecastInput, rng_key)`；
4. 评测器在预测完成后，才按 `ForecastKey` 连接目标；
5. 训练适配层以后可在训练被试上连接监督目标，但测试推理路径不得持有 target store；
6. 所有返回的 NumPy 数组设为只读或在边界处复制，避免预测器原地修改 mmap 视图。

`gesture_eval` 只用于：

- G12 vs G1–G11 的活动门训练目标；
- G12 排除运动 ADE/FDE；
- 预注册结果分层。

12 类标签不能进入 `ForecastInput`、归一化或随机键。

---

## 6. 表征与因果归一化

### 6.1 表征

主表征为 ClutterRemoved 的逐帧解析包络：

$$
A_{i,v,t,m}
=\left|\mathcal H_{\mathrm{range}}
\left(X^{\mathrm{CR}}_{i,v,t,:}\right)_m\right|.
$$

Hilbert/analytic 变换只沿最后一个 range-bin 轴进行，每一帧独立，不能跨 slow-time 求均值。

预注册表征名：

- `signed`：原始实值 ClutterRemoved；
- `envelope`：解析包络，P0a 默认；
- `log_power`：`np.log1p(envelope ** 2)`。

每种表征必须独立拟合 `FoldStats`，不能共用中心/尺度。

表征计算统一先把单条输入转为 `float64`，解析信号、median/MAD 和 FoldStats 均用
`float64`；归一化完成后才把模型输入显式转为 `float32`。测试比较必须同时覆盖
float64 统计量与 float32 输出。

### 6.2 训练折固定统计量

对训练序列 $i$、视角 $v$：

$$
m_{i,v}=\operatorname{median}_{t,m}Z_{i,v,t,m},
$$

$$
d_{i,v}=\operatorname{median}_{t,m}
\left|Z_{i,v,t,m}-m_{i,v}\right|.
$$

先在每名训练被试内部，对其全部序列的 $m_{i,v}$、$d_{i,v}$ 分别取中位数；再在 5 名训练被试之间
取中位数，得到 $c_v$、$\tilde d_v$：

$$
s_v=1.4826\,\tilde d_v+10^{-6},\qquad
\widehat Z_{i,v,t,m}=\frac{Z_{i,v,t,m}-c_v}{s_v}.
$$

这样每名训练被试等权，并且不需要把数亿个幅度值同时载入内存。

允许训练折使用训练序列的完整 90 帧来拟合固定统计量，因为这些序列属于训练数据；禁止使用
validation、calibration、test 的任意帧。测试前缀变换只能读取已冻结的 $c_v,s_v$。

`FoldStats` 必须记录：

- test/val/cal/train 被试；
- 表征名；
- `center` 与 `scale`；
- 源 shape、dtype；
- 统计协议版本 `P0a-v1`；
- 可复现 provenance 字符串。

若中心/尺度非有限、shape 错误或尺度不为正，硬失败，不静默回退到逐样本标准化。
在加 $10^{-6}$ 前若 $\tilde d_v\le0$，同样硬失败；epsilon 只防浮点除零，不得把退化视角伪装成
有效统计量。

`provenance` 冻结为 SHA-256：依次哈希协议版本 UTF-8、表征名 UTF-8、按固定顺序编码的
test/val/cal/train 被试号、`center.astype("<f8").tobytes()`、
`scale.astype("<f8").tobytes()`、源 shape 和 dtype。实现不得对浮点数先转十进制文本后哈希。

### 6.3 running-prefix 敏感性

running-prefix 只作为附加敏感性对照。对当前样本、当前视角，只允许在
`prefix[:, :t, :]` 内计算 median/MAD；不得读取 `x[:, t:, :]`。主结果仍使用
训练折固定统计量。

---

## 7. 随机性契约

确定性模型不使用随机键。随机预测器的随机流必须只由以下字段确定：

```text
(protocol_version, outer_fold, sample_idx, prefix_t, horizon, model_seed)
```

禁止把未来数组、未来摘要、Python 对象地址、遍历顺序或 batch 内位置加入种子。相同键和模型种子必须
产生相同随机样本；替换未来后随机样本也必须逐位相同。

---

## 8. C0 测试矩阵

已新增 `tests/test_causal_forecast_protocol.py`，使用仓库现有 `unittest` 风格。

### 8.1 纯合成单元测试

| 测试名 | 必须证明 |
|---|---|
| `test_prefix_slice_exact` | 哨兵数组上 `:t` 与 `t:t+H` 无偏一帧错误、无重叠 |
| `test_all_registered_windows_valid` | 12 个 $(t,H)$ 均合法且顺序冻结 |
| `test_lazy_key_count_and_order` | $N\times12$ 个键，不复制回波，固定排序 |
| `test_loso_subject_sets_exact` | 八折 train/val/cal/test 被试集合与 §4 完全一致 |
| `test_loso_indices_disjoint_and_complete` | 索引两两不交，并集完整，每行只出现一次 |
| `test_future_replacement_keeps_input_bytes` | 保持前缀、替换全部未来后，`ForecastInput` 逐位相同 |
| `test_fixed_stats_ignore_nontrain_data` | 改写 val/cal/test 不改变训练统计量 |
| `test_fixed_normalization_ignores_test_future` | 改写测试未来不改变归一化前缀 |
| `test_running_prefix_ignores_future` | running-prefix 对照也不读取未来 |
| `test_label_permutation_keeps_inputs` | 打乱 12 类标签不改变输入、键、统计量 |
| `test_input_target_are_separate_types` | predictor 只收到 `ForecastInput`，对象中不存在未来/标签字段 |
| `test_rng_key_excludes_target` | 替换未来不改变随机键与随机输出 |
| `test_representation_is_slow_time_local` | 改写未来帧不影响任一历史帧表征 |
| `test_invalid_protocol_fails_closed` | shape/dtype/被试/t/H/统计量错误均抛出协议异常，无 SKIP |
| `test_deterministic_repeated_run` | 相同输入、协议和种子输出逐位一致 |

纯合成测试不得读取真实缓存，确保失败可定位且运行快速。

### 8.2 实际缓存只读审计

纯合成测试全部通过后，才允许执行只读缓存审计：

1. 核对六个 `.npy` 的存在、shape、dtype；
2. 核对 Raw/ClutterRemoved 的 `y` 与 `subj` 逐项一致；
3. 核对被试/手势计数与 §2 一致；
4. 核对八折样本数与 §4 一致；
5. 核对 115116 个惰性键可枚举且没有越界；
6. 不读取整份 X 到内存，只用 mmap 和必要的小数组；
7. 不写模型分数、轨迹或缓存派生文件。

缓存计数与当前冻结事实不一致时必须硬失败，并要求先调查数据版本；不得自动丢弃或补齐样本。

### 8.3 后续模型的端到端 C0

每新增一个状态估计器或未来预测器，都必须增加同一模式的回归测试：

1. 构造两条完整序列，历史前缀相同、未来完全不同；
2. 使用相同 FoldStats、模型状态和随机键；
3. 比较时刻 $t$ 状态、历史编码、预测分布参数和固定键随机样本；
4. 要求逐位一致。GPU 模型必须为 C0 提供确定性评测路径（固定 RNG、确定性算子，或等价 CPU
   路径）；不能用事后放宽数值容差代替 15 冻结的“完全不变”。

接口级 C0 通过不自动替代端到端 C0。

---

## 9. 失败策略

以下任一情况判 P0a FAIL：

- 未来替换改变任何模型可见输入；
- val/cal/test 改变训练统计量；
- 标签进入输入、归一化、键或随机种子；
- LOSO 集合重叠或未完整覆盖；
- 前缀和未来重叠/越界；
- Raw/ClutterRemoved 标签或被试错位；
- 非有限归一化统计量被静默修补；
- 缓存异常被 SKIP；
- 测试通过依赖完整 90 帧 `MVDataset` 或 W1 smoother。

P0a FAIL 时，P0b 及其后的状态估计、轨迹预测和概率模型全部暂停。只修复协议与测试，不用训练结果反推
应当放宽哪条 C0。

---

## 10. 实施边界与拟新增文件

用户复核本设计后，实施计划只允许涉及：

| 文件 | 动作 |
|---|---|
| `uwb_osr/causal_data.py` | 新建：本设计的数据对象、LOSO、惰性键、表征和 FoldStats |
| `tests/test_causal_forecast_protocol.py` | 新建：§8 的纯合成测试与缓存审计测试 |
| `tools/w4_causal_gate.py` | 若实施计划证明有必要，新建轻量 CLI；`--selftest` 与 `--audit-cache` 分离 |
| `proposal_v2/15_...md` | 只在实现证据产生后更新状态，不改研究门 |

P0a 不修改：

- `uwb_osr/data.py`；
- `uwb_osr/trajectory.py`；
- `uwb_osr/forward.py`；
- 块一模型与训练脚本；
- `17`、`18`、`w3_readout_predict.py`。

`--selftest` 只运行纯合成测试；`--audit-cache` 只读真实缓存并打印协议清单，不运行
轨迹预测。二者都不得静默 SKIP 必选检查。

---

## 11. P0a 验收条件

P0a 只有同时满足下列条件才可记 PASS：

1. 本设计与 15 的 P0a、§4、C0、代码边界逐项一致；
2. 用户复核本设计；
3. 后续实现代码通过全部纯合成测试；
4. 两次纯合成测试的结果与输出逐位一致；
5. 实际缓存只读审计符合冻结 shape、dtype、计数、Raw/CR 索引元数据相等和键数；物理事件配对不在
   P0a 审计范围；
6. 没有生成模型分数、轨迹或研究结论；
7. 没有运行 W3 `--track` / `--analyze`，也没有冻结提交 18。

在第 2 项完成前，只能称“P0a 设计冻结候选”；在第 3–5 项完成前，只能称“P0a 实施中”，不得称
“C0 已通过”。

### 11.1 2026-07-31 实施证据

实施位置：隔离 worktree `D:\uwb_gestures\.worktrees\p0a-causal-prefix`，分支
`codex/p0a-causal-prefix`。未 stage、未 commit；隔离分支只有以下三个 P0a 新文件：

| 文件 | SHA-256 |
|---|---|
| `uwb_osr/causal_data.py` | `44c7cadb41596d5044eabae17e47e55d9342ab65fdf104664fc9a11c7bc31676` |
| `tests/test_causal_forecast_protocol.py` | `dc6b024481cf8cdd8748f3048fff3ed2607671f2ebde9eac62592296b8ad828e` |
| `tools/w4_causal_gate.py` | `8ba5b354d4833c9ea559e74f7b45ce553ad337fa238c01cc7d94f051bbb27ae4` |

验证证据：

- P0a 单元测试：47 项 PASS，0 failure、0 error、0 skip；独立审查后追加了“先切前缀再表征”、
  “X 主体任意访问即失败”和“RNG 固定已知向量”三项硬证据；
- 两次 `--selftest` stdout 原始字节完全相同：291 bytes，SHA-256
  `16b512d7a9f31dae53b9bf5414573f117bb56588adf966a796104a1cb1f8ad37`；
- 真实 cache 只读审计：`status=PASS`，shape `(9593,3,90,189)`、`float32`、9593 条、
  115116 个键；只有 HV6-G4 为 93，其余 95 个被试×手势格均为 100；八折计数与 §4 完全一致；
- X 只以 `mmap_mode="r"` 核对头部，manifest 为 `signal_body_read=false`；Raw/CR 的 y/subj
  逐位相等，但 `physical_event_pairing_scope="not audited in P0a"`；
- 静态禁止项 `MVDataset|viterbi_track|fit_trajectory|torch|scipy` 在 P0a 生产模块、测试和 CLI
  中均无匹配；`git diff --check` 通过；
- 全仓 Python 回归共 76 项，只有 Task 0 已登记的旧
  `test_corrupted_view_downweighted` 失败；新增 P0a 47 项全部通过，没有新增失败。

实际执行命令使用仓库现有的非 package `tests/` 结构，因此单测采用 `discover`：

```powershell
& "D:\D_conda\envs\pytorch\python.exe" -m unittest discover -s tests -p "test_causal_forecast_protocol.py" -v
& "D:\D_conda\envs\pytorch\python.exe" tools\w4_causal_gate.py --selftest
& "D:\D_conda\envs\pytorch\python.exe" tools\w4_causal_gate.py --audit-cache --cache "D:\uwb_gestures\cache"
```

禁止边界复核：

- `17_读出失败预测_预注册.md`：`fe35d845bb7feeb62050a59423e4cf1088b056440b10822b27bfbc6595e1f32f`；
- `18_通道D整数bin相位修正.md`：`2335e804440eb7b34416543d647b42f662a0e769b4a6427ec2ecba79e8adf7be`；
- `tools/w3_readout_predict.py`：`5e597a6cdba426cccebceb6609cfbccec4e680487b32f981bb43dbe83664f4f90`；
- `logs/w3_readout_scores.npz` 不存在；没有运行 W3 `--track/--analyze`，没有冻结提交 18。

**裁决边界**：P0a C0 PASS 只说明因果数据接口、切分、归一化、输入/目标隔离和随机键满足冻结契约；
它不是状态估计 PASS，不是轨迹预测 PASS，也不是研究点 2.1/2.2 PASS。P0b 尚未设计或执行。

---

## 12. 延后到 P0b 的内容

P0b 必须另行设计、预注册并获用户批准。当前只冻结数学可辨识边界：

- Last：无估计窗口；
- CV：位置序列上的速度窗口 $L\in\{2,5,20\}$；
- CA：二次拟合窗口 $L\in\{5,20\}$；
- 两帧不能估计加速度，因此禁止 CA-2；
- 每个 $H$ 不能单独选择窗口；
- 当前状态必须来自只看前缀的最小 F0，不得直接使用完整 90 帧 W1 smoother；
- 正式 FRLS/FRNLL 比较等待 O1 距离似然和评分判别力审计。

P0b 尚未冻结 F0 峰值定义、三边定位失败处理、Kalman 参数、CV/CA 回归公式、加速度约束或预测不确定度
包装；这些不能在 P0a 实现中顺手决定。

---

## 13. 文档复核清单

- [x] 15 与本文的 $t/H$、LOSO、归一化公式一致；
- [x] 15 与本文都明确区分 P0a、P0b；
- [x] 执行窗口统一为 CV(2/5/20)、CA(5/20)，CA-2 只作为禁止项出现；
- [x] P0a 未被写成轨迹预测结果；
- [x] W1 只作为以后训练折伪目标/次指标，不作为当前状态；
- [x] W3/17/18 边界未改变；
- [x] 用户已批准设计与实施计划，P0a 实现、Task 7 验证门和实施证据复核均已完成；当前只进入 P0b 设计与预注册。
