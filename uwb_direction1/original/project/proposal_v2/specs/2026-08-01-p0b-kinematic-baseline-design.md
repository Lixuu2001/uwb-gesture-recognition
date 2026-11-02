# P0b-v1：最小因果 F0 与未来轨迹运动学基线设计

**日期**：2026-08-01  
**版本**：P0b-v1  
**状态**：已按批准的实施计划完成纯合成/C0 实现与复核；P0b contract PASS，待用户审阅证据后才可进入任何真实参数选择或轨迹比较  
**上位方案**：`../15_块二重构_因果状态估计_未来轨迹预测_未来回波生成.md`  
**前置依赖**：`2026-07-31-p0a-causal-prefix-design.md` 与隔离 worktree 中已通过复核的 P0a-v1  
**研究主线**：历史回波 → 最小因果状态 → 未来轨迹预测 → 概率多模态 → 真实未来回波闭环  

---

## 0. 决策摘要与硬边界

P0b 只建立一把可信的未来轨迹预测“尺子”：

1. 用最小 F0 从历史回波得到不泄漏未来的位置/速度起点；
2. 冻结 Last、CV(2/5/20)、CA(5/20) 的精确未来轨迹公式；
3. 冻结失败语义、参数选择边界、确定性输出包装与纯合成 C0；
4. 把最佳冻结 CV 升格为复杂轨迹预测的淘汰基线。

P0b-v1 **不做**：

- 不实现或讨论 F1/F2、粒子滤波、神经状态估计器；
- 不训练 GRU、Transformer、Gaussian、MDN、flow 或 diffusion；
- 不读取真实信号缓存，不运行真实数据排名；
- 不生成 W1 离线伪轨迹，不以 W1 估计 Kalman 或运动约束；
- 不计算 ADE/FDE、FRLS/FRNLL、Energy/Variogram Score；
- 不运行 W3 `--track/--analyze`，不修改、冻结或提交 17/18；
- 不修改 P0a 的三个已复核文件或其证据哈希。

P0b PASS 只表示因果 F0 与运动学基线的公式、接口和合成实现可信，不表示未来轨迹可预测，也不表示
研究点 2.1/2.2 成立。

---

## 1. 时间、坐标与接口约定

### 1.1 数组时间约定

P0a 的 `prefix_t=t` 表示模型可见原数组索引 `0, ..., t-1`。因此：

- 当前帧索引固定为 `c=t-1`；
- 预测步 `h=1` 对应原数组索引 `t`；
- 预测步 `h=H` 对应原数组索引 `t+H-1`；
- `FPS=20.0`，`dt=0.05 s`；
- 长度为 `L` 的估计窗口固定使用 `c-L+1, ..., c`，不得偏移一帧。

本文公式中的当前位置统一写作 $p_c$，避免把数组的 `prefix_t` 与数学时刻混成同一个下标。

### 1.2 允许进入推理器的对象

F0 推理器运行时只允许接收：

1. 一个 P0a `ForecastInput`；
2. 构造推理器时冻结的训练折 `FoldStats`；
3. 已冻结的 `F0Hyperparameters`。

必须验证：

- `FoldStats.representation == "envelope"`；
- `ForecastInput.norm_provenance == FoldStats.provenance`；
- `radar_positions`、`valid_view_mask`、shape、dtype 与 P0a 契约一致。

`ForecastTarget`、未来数组、手势标签、被试编号不得进入 F0 或 Last/CV/CA 的预测调用。验证阶段的参数
选择器是独立评测组件，可以在预测完成后读取验证目标，但不能改变预测器接口。

### 1.3 冻结输出对象

`CausalStateTrace` 至少包含：

| 字段 | shape / 类型 | 含义 |
|---|---|---|
| `position_m` | `(t,3) float64` | 每帧过滤位置 |
| `velocity_mps` | `(t,3) float64` | 每帧过滤速度 |
| `state_cov` | `(t,6,6) float64` | $[p,v]$ 状态协方差 |
| `state_available` | `(t,) bool` | 该帧是否已有可传播状态 |
| `geometry_valid` | `(t,) bool` | 三路峰能否形成合法正深度点 |
| `update_accepted` | `(t,) bool` | Kalman 创新门是否接受当前点 |
| `peak_bin` | `(3,t) int64` | 每路因果前向峰；无效为 `-1` |
| `range_m` | `(3,t) float64` | 每路亚 bin 距离；无效为 `NaN` |
| `terminal_viterbi_score` | `(3,189) float64` | 前缀末端前向分数；只用于无回溯续接 |
| `tracker_initialized` | `(3,) bool` | 每路前向跟踪器是否已遇到合法帧 |
| `config_provenance` | 字符串 | 协议、统计量和参数的二进制哈希 |

`KinematicForecast` 至少包含：

| 字段 | shape / 类型 | 含义 |
|---|---|---|
| `key` | `ForecastKey` | P0a 键 |
| `model_name` | 字符串 | `Last`、`CV-2/5/20`、`CA-5/20` |
| `window` | `None/int` | Last 为 `None` |
| `anchor_position_m` | `(3,) float64` | 当前 $p_c$ |
| `displacement_m` | `(H,3) float64` | 相对 $p_c$ 的未来位移 |
| `position_m` | `(H,3) float64` | 绝对未来位置 |
| `samples_m` | `(1,H,3) float64` | 确定性 $K=1$ 包装 |
| `valid_horizon_mask` | `(H,) bool` | 预测是否有效；不允许删除步 |
| `constraint_active` | bool | 速度/加速度约束是否触发 |
| `distribution_kind` | 固定字符串 | `deterministic_delta` |
| `state_provenance` | 字符串 | 所用 F0 状态哈希 |

所有输出数组均为只读；内部计算和输出使用 `float64`。无效浮点槽位写 `NaN`，但必须同时有显式布尔
掩码，评测器不得仅凭 `NaN` 猜语义。

### 1.4 provenance 算法

两个 provenance 均使用 SHA-256，并沿用 P0a 的二进制纪律：文本先写 `<Q` little-endian 字节长度，
再写 UTF-8；整数写 `<q`；浮点标量写 `<d`；数组先转连续 little-endian dtype 再写原始 bytes。

`config_provenance` 按固定顺序哈希：

1. 文本 `P0b-v1`、P0a `norm_provenance`；
2. `FPS/N_BINS/RANGE_WINDOW_M/MAX_JUMP/JUMP_PENALTY/EMISSION_FLOOR/POWER_EPS`；
3. 选定的 $\sigma_r,\sigma_a$；
4. `radar_positions` 的 `<f8` bytes 与 `valid_view_mask` 的 `uint8` bytes。

`state_provenance` 再追加 `sample_idx,prefix_t`（**不追加 horizon**）与 `CausalStateTrace` 中除两个
provenance 字符串外全部数组的固定字段顺序 bytes。这样同一 sample/prefix 的 F0 状态及其 provenance
不随预测 H 改变。`KinematicForecast.key` 仍保存完整 ForecastKey。实现不得先把浮点数转十进制文本再哈希。

---

## 2. F0 的因果单峰距离轨迹

### 2.1 恢复物理包络

P0a 默认提供训练折固定标准化后的 envelope：

$$
z_{v,j,b}=\frac{e_{v,j,b}-c_v}{s_v}.
$$

F0 使用匹配 provenance 的 `FoldStats` 恢复：

$$
e_{v,j,b}=\max(0,\;s_v z_{v,j,b}+c_v).
$$

这里的非负裁剪只处理 `float32` 标准化往返造成的包络舍入误差，不允许用于修补距离或三边定位。

### 2.2 因果前向 Viterbi

冻结常数：

```text
N_BINS = 189
RANGE_WINDOW_M = 1.2
BIN_M = 1.2 / 189
MAX_JUMP = 20
JUMP_PENALTY = 0.15
EMISSION_FLOOR = 1e-6
POWER_EPS = 1e-30
```

每帧功率和 emission 与现有 W1 口径一致：

$$
P_{j}(b)=e_j(b)^2,
\qquad
E_j(b)=\log\left(\frac{P_j(b)}{\max_{b'}P_j(b')+10^{-30}}+10^{-6}\right).
$$

每路独立递推：

$$
S_j(b)=E_j(b)+
\max_{|d|\le20}\{S_{j-1}(b-d)-0.15|d|\}.
$$

当前帧立即输出：

$$
b_j^*=\arg\max_b S_j(b).
$$

同分时固定取最小 bin。实现禁止保存 backpointer 后从前缀末端回溯；给前缀追加任意未来帧后，已经输出
的所有历史峰必须逐位不变。现有 `uwb_osr/trajectory.py::viterbi_track` 会回溯，因此 P0b 不能调用它。

若某路 `valid_view_mask=False`，或当前帧最大功率不为正，则该路当前观测无效。内部得分状态可以按转移
继续传播，但该帧不得向三边定位提供伪距离。精确规则是：

- 首个合法帧之前保持 `tracker_initialized=False`，终态分数为 `-Inf`；
- 首个合法帧直接以该帧 emission 初始化；
- 已初始化后遇到无效帧，只执行带跳变代价的 max-transition、不加 emission；
- 无效帧对外仍写 `peak_bin=-1`、`range_m=NaN`；
- 验证选择器需要续接第一未来帧时，只能复制 `terminal_viterbi_score` 后前进一步，不能修改原状态。

### 2.3 亚 bin 距离

在当前帧功率上对 $b_j^*$ 左中右三点做抛物线插值：

$$
\delta=
\frac12\frac{P(b-1)-P(b+1)}{P(b-1)-2P(b)+P(b+1)},
\qquad \delta\in[-1,1].
$$

边界 bin 或零分母时固定 $\delta=0$。距离为：

$$
r_{v,j}=(b_j^*+\delta)\Delta r,
\qquad \Delta r=1.2/189\ \mathrm m.
$$

---

## 3. 三边定位与失败语义

### 3.1 正深度解析解

三台雷达共面，位置写为 $P_v=(X_v,Y_v,0)$。三路距离均有效时，以第 0 路为参考，解二维线性系统：

$$
2(P_{v,xy}-P_{0,xy})^Tq
=r_0^2-r_v^2+\|P_{v,xy}\|^2-\|P_{0,xy}\|^2,
\quad v\in\{1,2\},
$$

得到 $q=(x,y)$，再计算：

$$
z^2=\frac13\sum_{v=0}^2
\left[r_v^2-\|q-P_{v,xy}\|^2\right],
\qquad z=+\sqrt{z^2}.
$$

由于这是三个距离对三个位置自由度的恰定问题，只要存在正深度交点，逐帧重投影残差在数值上天然接近
零。它只能作为实现一致性检查，**不得作为定位准确或峰值正确的证据**。

### 3.2 几何有效门

满足以下全部条件才设置 `geometry_valid=True`：

1. 三路当前距离均有效、有限且在 `[0,1.2)` m；
2. 二维定位矩阵有限且 `cond(A) <= 1e8`；
3. $z^2>0$，禁止把负 $z^2$ 裁成 0；
4. $0.2<z<1.2$ m；
5. 距离 Jacobian
   $J_v=(p-P_v)^T/\|p-P_v\|$ 有限且 `cond(J^T J) <= 1e8`；
6. 数值重投影最大绝对误差不超过 `1e-9 m`。

任一条件失败即没有位置观测。不得裁剪位置、替换为 $(0,0,0.45)$、删除帧或删除预测键。

### 3.3 位置观测协方差

对候选逐视角距离标准差 $\sigma_r$，传播为：

$$
R_p=\sigma_r^2(J^TJ)^{-1}.
$$

只在条件数门通过后计算；结果必须对称、有限、正定，否则该帧观测无效。

---

## 4. 常速度 Kalman

### 4.1 状态与转移

状态为：

$$
x_j=[p_j^T,v_j^T]^T\in\mathbb R^6.
$$

冻结：

$$
F=
\begin{bmatrix}
I_3&\Delta t I_3\\
0&I_3
\end{bmatrix},
\quad
G=
\begin{bmatrix}
\tfrac12\Delta t^2 I_3\\
\Delta t I_3
\end{bmatrix},
\quad
Q_x=G(\sigma_a^2I_3)G^T,
\quad
H=[I_3\ 0].
$$

### 4.2 初始化

首个通过几何门的位置初始化：

$$
x=[p^T,0,0,0]^T,
$$

$$
P_0=
\operatorname{blockdiag}
\left(R_p,\sigma_{v0}^2I_3\right),
\qquad
\sigma_{v0}=20\Delta r/\Delta t.
$$

初始化前的帧全部 `state_available=False`。不得用未来帧、完整 W1 或人口中心点补初始化。
首个合法点的 `geometry_valid=True`、`update_accepted=True`；创新门从下一个合法位置观测开始使用。

### 4.3 预测、创新门与更新

每帧先执行标准预测。若无合法位置观测，只保存 predict-only 状态，并令
`geometry_valid=False`、`update_accepted=False`。

若有观测，计算：

$$
d^2=(y-Hx^-)^TS^{-1}(y-Hx^-),
\qquad S=HP^-H^T+R_p.
$$

冻结三维 $\chi^2$ 0.999 门限：

```text
INNOVATION_D2_MAX = 16.26623619623813
```

$d^2$ 超门限时同样 predict-only，不用裁剪 innovation。通过时使用 Joseph 形式更新协方差：

$$
P=(I-KH)P^-(I-KH)^T+KR_pK^T.
$$

每步只允许对 $P$ 做 $(P+P^T)/2$ 数值对称化；非有限或非正定必须 fail-closed，不允许靠大对角 jitter
掩盖实现错误。

### 4.4 冻结候选网格

```text
sigma_range_m / BIN_M = (1, 3, 8)
sigma_accel_mps2      = (1, 3, 10)
```

共 9 组，P0b 不在真实数据上选择赢家。

### 4.5 后续验证折选择规则

真实选择只能在每个外层折的固定验证被试上做一次：

1. 只用 P0a 四个 `prefix_t=(20,30,45,60)`；
2. 每个前缀只取 `H=5` 目标的第一未来帧，避免三次重复同一一步目标；
3. 候选预测一步位置并投影为三路距离；目标距离由验证目标第一帧追加到同一前向 Viterbi 状态后得到；
4. G12 只在评测层排除，不进入运动参数选择；G1–G11 的具体类别编号不进入预测器；
5. 逐视角绝对误差 $e$ 使用带米单位的 Huberized absolute loss，$\delta=3\Delta r$：
   $$
   L_\delta(e)=
   \begin{cases}
   e^2/(2\delta),& |e|\le\delta,\\
   |e|-\delta/2,& |e|>\delta;
   \end{cases}
   $$
6. 每个样本先对合法视角/前缀等权，再对验证样本等权平均；任一预测键不可用时，该键损失固定为
   `1.2 m`，禁止靠跳过失败键获胜；
7. 选择平均损失最小的一组；完全同分时依次取更小 $\sigma_a$、更小 $\sigma_r$；
8. 选定参数对该折所有训练、校准、测试样本和所有 $H$ 共用，禁止按被试、手势、前缀或预测时长重选。

该选择读取验证未来是合法的超参数选择，不是推理输入；测试未来始终不可见。

---

## 5. 运动学未来轨迹基线

### 5.1 共同窗口规则

对窗口长度 $L$，固定使用最后 $L$ 个**连续原始帧索引**：

$$
i=c-L+1,\ldots,c,
\qquad
\tau_i=(i-c)\Delta t.
$$

窗口内每帧必须 `state_available=True`。predict-only 帧仍是连续因果状态，可以使用；初始化前的无状态帧
不能被跳过或由更早帧补足。窗口不足时该模型整条 $H$ 预测均无效，但键和所有未来步仍必须输出。

所有模型锚定最新过滤位置 $p_c$，不使用回归截距替代起点。

### 5.2 Last position

Last 没有估计窗口：

$$
\widehat p_{c+h}=p_c,
\qquad h=1,\ldots,H.
$$

### 5.3 Constant velocity

冻结窗口：

```text
CV_WINDOWS = (2, 5, 20)
```

对每个坐标独立做带截距 OLS：

$$
p_i=\alpha+v\tau_i+\epsilon_i.
$$

只取斜率 $\widehat v_L$，预测为：

$$
\widehat p_{c+h}=p_c+h\Delta t\,\widehat v_L.
$$

CV-2 必须精确退化为最后两点有限差分。不得直接用 Kalman 内部 $v_c$ 替代 CV-L 的窗口斜率。

### 5.4 Constant acceleration

冻结窗口：

```text
CA_WINDOWS = (5, 20)
FORBIDDEN_CA_WINDOWS = (2,)
```

对每个坐标独立拟合：

$$
p_i=\alpha+v\tau_i+\tfrac12a\tau_i^2+\epsilon_i.
$$

CA-2 在注册、构造和调用三层都必须 `ProtocolError`；不得静默退化为 CV-2。

### 5.5 训练折运动约束

F0 参数在验证被试上选定后，仅用五名训练被试的因果 F0 轨迹计算约束；不读取 W1：

1. 速度只来自相邻两帧均 `update_accepted=True` 的位置差分；
2. 加速度只来自连续三帧均 `update_accepted=True` 的二阶差分；
3. 每名训练被试的速度和加速度有效样本数都必须至少为 100；
4. 对排序后的 $n$ 个范数，用 nearest-rank 索引 $\lceil0.99n\rceil-1$ 取第 99 百分位，避免 NumPy
   版本改变插值口径；
5. 五名被试各自产生一个速度和加速度百分位，再分别取排序后的第三个值，得到 $v_{99}$、$a_{99}$；
6. 任一训练被试样本不足，或阈值不为有限正数，协议停止，不设默认回退值。

CV 将 $\widehat v$ 做欧氏范数裁剪到 $v_{99}$。

CA 先把拟合 $\widehat v$、$\widehat a$ 分别裁到 $v_{99}$、$a_{99}$，再逐步积分：

$$
v_h=\operatorname{clipnorm}(v_{h-1}+\widehat a\Delta t,v_{99}),
$$

$$
p_h=p_{h-1}+\tfrac12(v_{h-1}+v_h)\Delta t.
$$

未触发速度上限时，它与常加速度公式完全一致。任何裁剪触发均记录 `constraint_active=True`，不得隐藏。

### 5.6 确定性分布包装

Last/CV/CA 固定为 $K=1$ 的退化分布：

$$
\{\widehat p_{c+1:c+H}^{(1)}\}.
$$

不得添加从测试误差拟合的高斯噪声，也不得为了提高 coverage 撒点。F0 状态协方差作为状态诊断保留，
但 P0b 不把它冒充已经校准的未来轨迹分布。

---

## 6. 窗口报告与 CV 淘汰门

### 6.1 不允许按 H 挑窗口

正式阶段必须完整报告：

- Last；
- CV-2、CV-5、CV-20；
- CA-5、CA-20；
- 四个前缀端点；
- $H=5/10/20$，即 0.25/0.5/1.0 s。

`CV*` 只能在验证被试上，使用已冻结 O1/评分口径，对 12 个 `(prefix_t,H)` 单元等权平均 FRNLL 后
选择一个 $L\in\{2,5,20\}$。缺失单元记 $+\infty$，不能跳过；同分取较短窗口。该窗口随后对所有
测试被试和所有 H 固定。CA 可用相同规则登记 `CA*`，但复杂模型的正式淘汰对手固定为 `CV*`。

### 6.2 复杂预测的两级硬淘汰门

两级门都只能在 O1 和评分判别力审计通过后运行。统一定义正 FRNLL 相对降幅：

$$
g_H=\frac{\operatorname{FRNLL}_{CV^*,H}-\operatorname{FRNLL}_{model,H}}
{\operatorname{FRNLL}_{CV^*,H}}.
$$

分母必须有限且严格为正。四个前缀端点等权；不得先池化所有窗口再让样本量大的格子支配结果。

#### 6.2.1 开发进入门：不得打开外层测试

先冻结一个简单、同预算的确定性学习预测器。对每个 H，分别在八折的验证角色和校准角色上聚合
$g_H$。至少一个 H 必须同时满足：

1. 验证角色的 $g_H\ge0.05$；
2. 校准角色的 $g_H\ge0.05$；
3. 两个角色方向一致，且所有 H 完整报告。

若没有任何 H 通过，复杂未来轨迹预测在开发阶段 NO-GO：不打开外层测试，不开发 MDN、flow、
diffusion、MoFlow 或 LED。该门是成本/路线裁决，不包装成最终统计结论。

若至少一个 H 通过，后续历史长度、Gaussian/MDN/flow 与多模态门仍只能使用 train/val/cal 决定；完成
全部模型家族和超参数冻结后，才允许一次性打开各折外层测试。

#### 6.2.2 最终证据门：冻结后单次外层测试

对每个预注册 H，最终冻结模型必须同时满足：

1. 外层测试上的
   $$
   g_H\ge0.05;
   $$
2. 以 8 个外层测试被试为聚类单位的配对 95% CI 下界严格大于 0；
3. 三个 H 全部照实报告。

测试门失败后不得回到开发阶段更换模型、窗口或 H。该门仍不是最终论文 PASS；后续联合分布、覆盖和
真实回波闭环继续按 15 的独立条款验收，但它们不能反过来挽救一个未赢 CV 的 H。

裁决：

- 只在 H=5 或 H=5/10 通过：只保留得到支持的 0.25/0.5 s 主张；
- H=5/10/20 全部不通过：研究点 2.2 的复杂未来轨迹预测 **NO-GO**，不靠更大 GRU/Transformer、
  MDN、flow、diffusion、MoFlow 或 LED 挽救；只保留 CV 工程基线；
- “真实应用不会匀速”只能成为未来新数据假设，不能挽救当前数据上的失败；
- CV 表现差时仍作为必要对照，但不自动证明任一复杂模型成功。

---

## 7. C0 与纯合成测试矩阵

### 7.1 因果与接口

| 测试 | 必须证明 |
|---|---|
| `test_f0_accepts_forecast_input_only` | 预测调用不接受 target/future/label/subject |
| `test_stats_provenance_required` | representation 或 provenance 不匹配即停止 |
| `test_prefix_extension_preserves_history_bytes` | 追加任意帧不改变已有峰与状态 |
| `test_future_replacement_preserves_all_forecasts` | 同前缀、不同未来时，六个基线逐位一致 |
| `test_state_independent_of_horizon` | 同 sample/prefix 的 F0 状态及 provenance 不随 key.H 改变 |
| `test_forecast_horizon_prefix_consistency` | H=5/10 预测分别等于 H=20 的前 5/10 步 |
| `test_existing_backtracking_viterbi_not_used` | 生产模块不导入/调用现有 `viterbi_track` |
| `test_outputs_read_only_and_complete` | 无删帧、无删键、shape/dtype/mask 冻结 |

### 7.2 峰值与几何

| 测试 | 必须证明 |
|---|---|
| `test_forward_viterbi_known_path` | 固定 emission/transition 的已知路径正确 |
| `test_forward_viterbi_tie_uses_lowest_bin` | 同分决策确定 |
| `test_subbin_parabola_exact` | 已知抛物线峰恢复正确亚 bin |
| `test_trilateration_recovers_known_point` | 合法三距离恢复正深度真点 |
| `test_negative_z2_is_invalid_not_clipped` | 不存在交点时显式失败 |
| `test_geometry_condition_gate` | 退化几何 fail-closed |
| `test_exact_fit_not_claimed_as_accuracy` | 恰定重投影残差只作数值一致性 |

### 7.3 Kalman 与失败传播

| 测试 | 必须证明 |
|---|---|
| `test_kalman_matrices_exact` | F/G/H/Q 逐项符合冻结公式 |
| `test_initial_state_and_covariance_exact` | 首个合法点初始化、之前无状态 |
| `test_invalid_measurement_is_predict_only` | 失败帧不更新、不删除，协方差增长 |
| `test_innovation_rejection_is_predict_only` | 超门限不裁剪 innovation |
| `test_joseph_covariance_stays_symmetric_pd` | 更新后有限、对称、正定 |
| `test_qr_registry_exact` | 只有冻结 9 组候选 |
| `test_selector_rejects_cal_or_test_data` | 选择器只接受冻结训练产物与验证数据，cal/test 无入口 |
| `test_selector_penalizes_unavailable_keys` | 不能靠跳过失败键获胜 |

### 7.4 未来轨迹公式

| 测试 | 必须证明 |
|---|---|
| `test_last_is_zero_displacement` | 所有 H 保持 $p_c$ |
| `test_cv_windows_recover_linear_motion` | CV-2/5/20 精确恢复直线轨迹 |
| `test_cv2_equals_last_finite_difference` | CV-2 数学退化正确 |
| `test_ca_windows_recover_quadratic_motion` | 无约束触发时 CA-5/20 精确恢复二次轨迹 |
| `test_ca2_rejected_everywhere` | CA-2 无静默分支 |
| `test_regressions_anchor_at_current_state` | 不以拟合截距替代当前点 |
| `test_windows_do_not_compress_missing_time` | 初始化前无状态不能跨越补窗 |
| `test_norm_constraints_and_flags` | 范数裁剪和 `constraint_active` 正确 |
| `test_one_cv_window_shared_across_horizons` | 选择器不能按 H 返回三个窗口 |
| `test_deterministic_distribution_k1` | `samples_m.shape==(1,H,3)` 且无随机撒点 |

### 7.5 自检输出

`--selftest` 必须：

- 只构造纯合成峰、几何、轨迹和未来替换夹具；
- 不读 `cache/`、`logs/`、W1/W3 产物；
- 全部检查必须 PASS，无 `SKIPPED`；
- 两次运行 stdout 逐字节一致且 exit=0；
- 明确打印 `P0b contract PASS != trajectory predictability PASS`。

---

## 8. 拟新增文件与隔离边界

书面规范复核和实施计划批准后，实施只新增：

| 文件 | 责任 |
|---|---|
| `uwb_osr/causal_baselines.py` | F0、状态/预测对象、Last/CV/CA、参数与窗口选择纯函数 |
| `tests/test_causal_baselines.py` | §7 的纯合成/C0 测试 |
| `tools/w4_kinematic_gate.py` | `--selftest` 与以后单独批准的参数选择入口 |

实施位置继续使用：

```text
D:\uwb_gestures\.worktrees\p0a-causal-prefix
```

P0a 的以下文件保持逐字不改：

```text
uwb_osr/causal_data.py
tests/test_causal_forecast_protocol.py
tools/w4_causal_gate.py
```

P0b 不混入现有 `uwb_osr/trajectory.py`；后者继续是完整 90 帧 W1 离线 smoother。

---

## 9. P0b 自身 Go/No-Go

P0b PASS 必须同时满足：

1. §7 全部纯合成测试通过，无 skip；
2. 两次 `--selftest` stdout 逐字节一致；
3. 前缀扩展与未来替换 C0 对六个基线全部通过；
4. 直线/二次真轨迹的公式恢复测试通过；
5. 失败帧和不可用窗口均保留完整键/时间轴；
6. 静态边界确认不调用 W1 回溯、不导入 target/label、未读取真实缓存；
7. P0a 三个文件哈希未变；
8. W3 分数文件仍不存在，未执行 `--track/--analyze`；
9. 未训练、未排名、未输出任何轨迹可预测性结论。

任一失败时只修复 P0b 协议或实现；不得通过读取真实结果调门限，也不得提前进入学习预测器。

---

## 10. P0b 之后的主线顺序

P0b 通过后：

1. 只实现未来真实回波评分必需的最小 O1/判别力审计；
2. 用冻结 F0 和 Last/CV/CA 建立真实 C1 基线表；
3. 增加一个简单、同预算的确定性学习预测器，只用 val/cal 执行开发进入门；
4. F1/F2 不得阻塞这次轨迹可预测性裁决；它们仍可作为研究点 2.1 的独立后续；
5. 只有开发门确实赢 `CV*`，才继续 Gaussian/MDN/ordinary flow 与 M0/M1；
6. 只有多模态门和非扩散概率基线都给出必要性，diffusion 才可进入；
7. 所有家族冻结后只打开一次外层测试；测试失败不得回头换模型；
8. 最终胜出轨迹必须进入真实未来距离/CIR 观测闭环，不用 W1 ADE 单独收官。

这保证工作重心始终是未来轨迹预测及其观测闭环，而不是把 P0b 扩张成另一条状态估计工程线。

---

## 11. 已批准决策记录

用户在 2026-08-01 已明确批准：

- P0b 采用纯协议与纯合成验收范围；
- F0 单峰采用因果前向 Viterbi；
- 三边定位失败采用显式无效 + Kalman predict-only，不裁剪、不删键；
- Kalman 使用小型冻结候选网格，只在验证被试选择一次；
- CV 是复杂轨迹预测的正式淘汰基线；
- 若三个 H 全部不赢 CV，则舍弃当前数据上的复杂预测方法；
- 后续技术细节按本文推荐方案收敛；
- 书面规范复核通过后可以继续实施，但仍遵守 P0b 的无真实运行边界。

本文书面版本已由用户于 2026-08-01 复核通过；后续改动必须先说明原因，不得依据真实结果反向修改
冻结公式或门限。

---

## 12. 2026-08-01 实施证据与当前停止点

实施位于隔离 worktree `D:\uwb_gestures\.worktrees\p0a-causal-prefix`，未读取真实信号、未运行 W1、
未运行 W3 `--track/--analyze`，也未做正式轨迹误差或模型排名。

| 证据 | 结果 |
|---|---|
| P0a 聚焦测试 | 47/47 PASS，0 skip |
| P0b 聚焦测试 | 50/50 PASS，0 skip |
| 仓库全套测试 | 126 项；仅保留既有 `test_corrupted_view_downweighted` 失败，0 个新增失败，0 error |
| P0b CLI 确定性 | 两次 stdout 逐字节一致；430 bytes；SHA-256 `3c8e727dc8ee677c33be14a6148c82dd17003794075299517cc1b673b3f79338` |
| 静态边界 | P0b 生产模块/CLI 未出现 W1、target/label、PyTorch、SciPy、cache/log 读取依赖 |
| W3 边界 | `logs/w3_readout_scores.npz` 不存在；17/18/W3 脚本哈希与复审记录一致 |

P0b 新增文件 SHA-256：

```text
2EC5CF3CCBAA651A731F82C8BF9613AB74F7A4A2A8DDA6617C3F0AF43379485D  uwb_osr/causal_baselines.py
C67FBB681242F30F9DC79E82366B9995D821BBD681FB1AF9C283F728CD3B3751  tests/test_causal_baselines.py
B995D067600F65F0FD1FB7FF1EB431D51B614E1B5ECD9AE74CE3CB7FEB921547  tools/w4_kinematic_gate.py
```

P0a 三文件仍分别为 `44C7CADB…1676`、`DC6B0244…828E`、`8BA5B354…AE4`。17、18 与
`w3_readout_predict.py` 仍分别为 `FE35D845…1F32F`、`2335E804…ADF7BE`、`5E597A6C…64F4F90`。

**裁决边界**：`P0b contract PASS != trajectory predictability PASS`。这里证明的是因果接口、F0 与
Last/CV/CA 尺子可用于下一阶段，尚未证明手势未来轨迹可预测，也没有产生 CV 胜负结论。

**当前停止点**：等待用户审阅上述证据。获得下一阶段明确批准后，才允许做真实数据所需的最小 O1/评分
审计、验证折参数选择与 C1 轨迹比较；若学习模型在 H=5/10/20 全部不赢冻结 CV，则按预注册舍弃当前
复杂预测路线。CV 即使自身表现差也仍保留为必要对照，不能据此自动宣称复杂模型成功。
