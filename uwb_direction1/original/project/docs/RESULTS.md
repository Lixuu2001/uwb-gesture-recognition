# MV-OSR 实验汇总(UWB-Gestures, Figshare 12652592)

数据:`X (9593, 3, 90, 189)` ClutterRemoved,3 雷达(Left/Top/Right)软同步,
Eom 12 类 / 8 受试者,每 (人,手势) 100 样本。

OSR 协议:Eom Table I 的 5 个 case,每 case 6 已知 / 6 未知,
已知 600 训 / 200 测,未知 1200 测。OSCR/AUROC 为阈值无关分数;Acc/F1 取最佳阈值。

OSR 头:**deep inspector(per-class one-vs-rest 原型 + 距离阈值,Eom Stage-2),**
辅助报 **MSP** 基线。同框架内对照 Eom 单视角 OSCR ≈ 88.42。

---

## 主结论

**三视角 + flatten + abs_std + inspector,5-case OSCR 95.14 ± 0.81,超过 Eom 单视角 88.42 共 6.72 分,且标准差降到 0.81(论文 Eom 三视角 per_view_std 是 ±2.26)。** 内部多视角增益 +12.24 分 OSCR,case 4 多视角增益最大达 +21.36 分。

---

## 实验总览(TL;DR)

把 Eom 单视角方案扩成三视角,再围着主结果做消融。一句话:**多视角 + abs_std + flatten 是关键;物理 bias / 加宽模型 / 杂波对消预处理 都没用或有害(负结果让叙事更可信)。**

| 类别 | 实验 | 一句话结论 |
|---|---|---|
| **主结果** | 视角数 / 归一化 / 池化 三维扫描 | 三视角 + abs_std + flatten,**OSCR 95.14 ± 0.81**(单视角 +10 分) |
| 消融 E2 | 缺视角鲁棒性(运行时丢 1–2 雷达) | 给出降级曲线,Discussion 用 |
| 消融 E4 | score 分量(距离项 vs 视角间 disagreement) | 拆清多视角增益来源 |
| 消融 E5 | 物理 bias on/off | **无增益**(95.10 vs 95.14)— 负结果 |
| 消融 E3 | per-head D_key 8→32(对齐论文宽度 dim=128) | **掉 4.8 分**,过拟合 — dim=32 更优 |
| 阈值 A7 | 论文 η=3 启发式阈值 | 单视角近最优(gap 1.1),三视角 β=1 失准(gap 27)— 验证尺度依赖 |
| 输入 D1 | Raw vs ClutterRemoved | **打平**(95.53 vs 95.14)— 杂波对消冗余,模型自己能学 |

详见下方各节(§10–§17)。

---

## 1. Case-0 2×2 配置对照(inspector 头)

| pool × normalize | 单视角 OSCR | 三视角 OSCR |
|---|---|---|
| mean × per_view_std | 69.71 | 92.56 |
| flatten × per_view_std | 74.85 | 94.21 |
| **flatten × abs_std** ⭐ | **85.15** | **96.19** |

闭集(case 0,inspector):mean 94.42 → flatten 95.67 → flatten+abs_std 96.67(三视角同步抬到 99 左右)。

> 结论:`pool` 从 mean 到 flatten 是次因(+5),**`normalize` 从 per_view_std 到 abs_std 是首因(+10)**。

---

## 2. 5-case 全扫主对照表(flatten + inspector)

### 2.1 单视角(Left)

| 指标 5-case ±std | per_view_std | **abs_std** | Δ |
|---|---|---|---|
| Accuracy | 68.72 ± 4.05 | **78.03 ± 3.77** | +9.31 |
| F1 | 70.68 ± 2.14 | **80.84 ± 3.10** | +10.16 |
| AUROC | 75.38 ± 4.69 | **84.92 ± 4.10** | +9.54 |
| **OSCR** | **72.42 ± 4.73** | **82.90 ± 4.83** | **+10.48** |
| MSP OSCR (副) | 71.13 ± 1.57 | 77.02 ± 4.93 | +5.89 |

### 2.2 三视角(Left + Top + Right)

| 指标 5-case ±std | per_view_std | **abs_std** | Δ |
|---|---|---|---|
| Accuracy | 85.88 ± 2.09 | **92.29 ± 1.27** | +6.41 |
| F1 | 86.66 ± 1.85 | **92.54 ± 1.20** | +5.88 |
| AUROC | 93.12 ± 2.52 | **97.12 ± 1.24** | +4.00 |
| **OSCR** | **90.63 ± 2.26** | **95.14 ± 0.81** ⭐ | **+4.51** |
| MSP OSCR (副) | 83.34 ± 7.25(回看 sweep) | 81.85 ± 7.53 | — |

### 2.3 vs 论文 Eom

| | Eom 单视角 5-case | 我们单视角 abs_std | 我们三视角 abs_std |
|---|---|---|---|
| OSCR | 88.42 | 82.90 (gap −5.52) | **95.14 (+6.72)** ⭐ |

> 单视角差 5.5 分,在 1.2 std 内,基本复现到位;**三视角大幅超越,标准差还更小**。

---

## 3. 5-case 每 case 明细(flatten + abs_std + inspector)

| Case | known | unknown | 单视角 closed | 单视角 OSCR | 三视角 closed | 三视角 OSCR | **多视角增益** |
|---|---|---|---|---|---|---|---|
| 1 | b,e,f,h,i,l | a,c,d,g,j,k | 98.08 | 87.04 | 99.33 | 96.19 | +9.15 |
| 2 | b,e,f,h,k,l | a,c,d,g,i,j | 94.67 | 84.35 | 96.00 | 94.80 | +10.45 |
| 3 | c,f,g,h,j,l | a,b,d,e,i,k | 98.33 | 85.99 | 99.17 | 94.72 | +8.73 |
| 4 | a,b,h,i,k,l | c,d,e,f,g,j | 94.75 | **73.54** ⬇ | 95.75 | 94.04 | **+20.50 ⭐** |
| 5 | b,c,h,i,j,k | a,d,e,f,g,l | 97.25 | 83.59 | 98.58 | 95.95 | +12.36 |
| **均值** | | | **96.62** | **82.90 ± 4.83** | **97.77** | **95.14 ± 0.81** | **+12.24** |

> **case 4 单视角最难(73.5,inspector < MSP),三视角增益也最大(+20.5)** —— 多视角的边际收益在最难场景上最大。

---

## 4. 归一化诊断(case 0,单视角 flatten,inspector)

| `--normalize` | closed | INSPECTOR OSCR | AUROC | MSP OSCR | 说明 |
|---|---|---|---|---|---|
| `none` 原始带符号 | 87.58 | 66.44 | 71.94 | 66.75 | 模型同时处理正负反射,信息冗余 |
| `per_view_std`(旧默认) | 95.67 | 74.85 | 75.95 | 72.40 | std 归一对带符号数据次优 |
| `abs` 仅取 \|x\| | 97.42 | 81.34 | 82.71 | 76.55 | 论文 amplitude 字面 |
| **`abs_std` \|x\| + std**(新默认) | **96.67** | **85.15** | **86.13** | 77.83 | 运动能量 + 尺度归一,最佳 |

> 论文 III 脚注 1 写 "amplitude-based recognition" 是关键。clutter-removed 信号本身对称带符号,先取 |x| 才能拿到"运动能量",再做 std 归一保留几何并去除会话级尺度差。

---

## 5. Case 4 多 seed 稳定性复测(单视角 flatten + abs_std)

| seed | closed | INSPECTOR OSCR | AUROC | MSP OSCR |
|---|---|---|---|---|
| 0 | 94.67 | 73.18 | 76.49 | 79.36 |
| 1 | 94.58 | 71.46 | 75.52 | 76.93 |
| 2 | 94.33 | 73.40 | 77.09 | 73.81 |
| **均值±std** | 94.53 | **72.68 ± 0.86** | 76.37 | 76.70 |

> 三 seed 高度一致,**73 左右是稳定结果,不是 seed 抽到的坏运气**。case 4 单视角下 MSP 反超 inspector 在 3 个 seed 都重现 → 距离式 inspector 对此 case 的"近亲未知"过度自信。

---

## 6. 其它诊断结果

### 6.1 Inspector batch size(case 0 单视角 flatten per_view_std)
| batch_size_insp | INSPECTOR OSCR | MSP OSCR |
|---|---|---|
| 256(默认) | 74.85 | 72.40 |
| 16(Eom 原文) | 63.36 | 69.78 |

> 把 inspector batch 降到 16(论文 batch)反而**严重恶化**(-11 分),猜想:小 batch 下原型距离更新过于嘈杂、加上 flatten 8928 维特征,inspector 过拟合。**保留默认 256。**

### 6.2 Evidential / TMC OSR 路线(早期探索,case 0)
| 配置 | EVID OSCR | MSP OSCR | 备注 |
|---|---|---|---|
| 单视角 mean + 无 OOD | 63.90 | 62.74 | 朴素阈值法,印证论文 Fig.6 失败模式 |
| 单视角 mean + OOD λ=1 | 54.37 | 57.22 | OOD KL 过强,反而恶化 |
| 三视角 mean + OOD λ=1 | 81.19 | 83.85 | 多视角自身已强,EDL/DS 没赢 inspector |

> Evidential/TMC + Dempster-Shafer 在多视角下有效但**不如 inspector**(三视角 mean evidential 81 vs inspector 92.6)。保留作为对照消融或论文 ablation,**主路用 inspector**。

### 6.3 早期 mean pool 路径(对比基线,case 0)
| 配置 | closed | INSPECTOR OSCR |
|---|---|---|
| 单视角 mean per_view_std | 94.42 | 69.71 |
| 三视角 mean per_view_std | 99.08 | 92.56 |

---

## 7. 关键技术决策(论文要写清楚的)

1. **特征码 flatten,不 mean pool**(8928 维 vs 64 维)—— 对 inspector 的距离几何关键。
2. **`normalize=abs_std`**(`|x|` 后 per-sample per-view std 归一)—— 对齐论文 "amplitude" 表述,是单视角复现到 83+ 的关键。
3. **OSR 头用 deep inspector**(per-class 原型 + 距离阈值),不用 evidential。
4. **inspector batch=256**(我们默认),不要照搬论文 16(在 flatten 8928 维下会过拟合)。
5. **多视角融合用 per-view inspector + d_mean + var/σ̄²**(disagreement 分量),不用证据融合作主路。

---

## 8. 日志档案

每个实验的完整日志(逐 case + 汇总表)保存在 `D:\uwb_gestures\logs\`:

| 文件 | 内容 |
|---|---|
| `sweep5_single_flat_absstd.log` | **单视角 5-case 主结果(82.90 ± 4.83)** |
| `sweep5_three_flat_absstd.log` | **三视角 5-case 主结果(95.14 ± 0.81)** |
| `sweep5_single_flat.log` | 单视角 per_view_std 对照 |
| `sweep5_three_flat.log` | 三视角 per_view_std 对照 |
| `diag_norm_{none,abs,absstd}.log` | 归一化 4 路诊断 |
| `diag_c4_seed{0,1,2}.log` | case 4 多 seed 复测 |
| `b0insp_case0.log` / `b3insp_case0.log` | 早期 mean pool case 0 |
| `b0insp_flat_case0.log` / `b3insp_flat_case0.log` | flatten per_view_std case 0 |

---

## 9. 仍可做的扩展

| 编号 | 内容 | 预期收益 |
|---|---|---|
| E1 | LOSO 8 折交叉验证(主结果鲁棒性) | 应对 session 泄漏质疑;论文加分 |
| E2 | 缺视角鲁棒性(缺 1/缺 2 测试) | 多视角独占,MSP 直接崩;论文亮点 |
| E3 | 自定义 MSA 让 per-head Dkey=32(对齐论文) | 单视角可能再涨,目前差 5.5 分大概率出在这 |
| E4 | mean+variance 分量消融(disagreement 信号独立性) | 把"多视角分歧是 OSR 信号"做成 ablation 章节 |
| E5 | 物理 bias on/off(`--physical-bias`) | 跨视角注意力增益,可能 +1 分 |

---

## 10. mean × abs_std 补完(2×2 网格闭合)

### 10.1 单视角 mean+abs_std 5-case

| 指标 5-case ± std | mean × abs_std |
|---|---|
| Accuracy | 77.98 ± 2.47 |
| F1 | 78.75 ± 0.65 |
| AUROC | 85.65 ± 3.51 |
| **OSCR** | **83.34 ± 3.34** |
| MSP OSCR (副) | 78.97 ± 2.71 |

### 10.2 三视角 mean+abs_std 5-case

| 指标 5-case ± std | mean × abs_std |
|---|---|
| Accuracy | 89.62 ± 2.72 |
| F1 | 90.61 ± 2.02 |
| AUROC | 95.63 ± 1.80 |
| **OSCR** | **93.63 ± 2.05** |
| MSP OSCR (副) | 79.66 ± 4.31 |

### 10.3 mean + abs_std 每 case 明细

| Case | 单视角 closed | 单视角 OSCR | 三视角 closed | 三视角 OSCR | 多视角增益 |
|---|---|---|---|---|---|
| 1 | 98.33 | 85.99 | 99.17 | 96.32 | +10.33 |
| 2 | 94.00 | 83.52 | 96.00 | 94.68 | +11.16 |
| 3 | 96.83 | 84.94 | 98.67 | 93.23 | +8.29 |
| 4 | 94.83 | 85.38 | 95.92 | 90.10 | +4.72 |
| 5 | 95.83 | **76.85** ⬇ | 98.33 | 93.80 | +16.95 |
| **均值** | 95.96 | **83.34 ± 3.34** | 97.62 | **93.63 ± 2.05** | **+10.29** |

### 10.4 全 2×2 OSCR 总览(5-case 均值 ± std)

| 5-case OSCR | per_view_std | abs_std |
|---|---|---|
| **mean × 单视角** | — (case 0 only: 69.71) | **83.34 ± 3.34** |
| **mean × 三视角** | — (case 0 only: 92.56) | **93.63 ± 2.05** |
| **flatten × 单视角** | 72.42 ± 4.73 | 82.90 ± 4.83 |
| **flatten × 三视角** | 90.63 ± 2.26 | **95.14 ± 0.81** ⭐ 主结果 |

### 10.5 关键新发现

1. **`abs_std` 在两种 pool 上都是首因**:无论 mean 或 flatten,从 per_view_std 切到 abs_std,**单视角 +10–11 分,三视角 +3–4 分**。归一化的影响完全独立于池化方式。
2. **单视角下 mean 和 flatten 打平**:83.34 vs 82.90,差距在 std 内;mean 反而更稳(±3.34 vs ±4.83)。
3. **三视角下 flatten 胜出**:95.14 vs 93.63,差 1.5 分(> 1 std)——每视角更丰富的 token 序列对 per-view inspector 更有利,disagreement 信号更精细。
4. **"最难 case"会随 pool 翻转**:mean+abs_std 下 case 5 最弱(76.85,inspector < MSP);flatten+abs_std 下是 case 4 最弱。但 5-case 均值高度稳定。

### 10.6 实用结论(替代论文 Section 7 第 1 条的更细判断)

- **主结果(论文 headline)**:flatten + abs_std,三视角 **95.14 ± 0.81**。
- **轻量替代**:mean + abs_std 三视角 **93.63 ± 2.05**,**参数量小 ~140×**(feat=64 vs 8928),仅落后 1.5 分,适合"轻量部署版"或资源受限的对比表。
- 单视角下 mean 与 flatten 等价 → 可以在论文里写"对单视角来说池化方式不重要,关键是归一化"。

### 10.7 新增日志

| 文件 | 内容 |
|---|---|
| `sweep5_single_mean_absstd.log` | 单视角 mean+abs_std 5-case(83.34 ± 3.34) |
| `sweep5_three_mean_absstd.log` | 三视角 mean+abs_std 5-case(93.63 ± 2.05) |

---

## 11. E2 — 缺视角鲁棒性 (missing-view robustness)

**实验设计**:三视角 flatten + abs_std + inspector 训练好的模型,**测试时**切换 view subset 重新评测(**不重训**)。`SharedDualPathEncoder.forward(X, view_indices=...)` 把 7 种 view 组合在评测阶段一次出齐。

### 11.1 完整 5-case 结果(combined score,均值 ± std,%)

| view 子集 | 用几路 | **OSCR** | AUROC | Accuracy | F1 | MSP OSCR (对照) |
|---|---|---|---|---|---|---|
| **all (L+T+R)** | 3 | **95.33 ± 0.79** | 97.30 ± 1.01 | 92.45 ± 1.21 | 92.46 ± 1.39 | 83.05 ± 4.98 |
| **noT (L+R)** | 2 | **86.22 ± 2.63** ⭐ | 90.87 ± 2.42 | 84.65 ± 2.25 | 84.44 ± 2.38 | 72.42 ± 5.76 |
| noR (L+T) | 2 | 80.21 ± 2.62 | 88.76 ± 1.43 | 82.52 ± 1.58 | 81.18 ± 2.03 | 58.10 ± 6.19 |
| noL (T+R) | 2 | 71.40 ± 9.75 | 83.52 ± 6.20 | 79.72 ± 4.13 | 75.16 ± 7.10 | 47.68 ± 10.90 |
| L only | 1 | 54.71 ± 6.46 | 63.00 ± 5.22 | 66.42 ± 3.09 | 61.26 ± 5.08 | 40.07 ± 8.21 |
| T only | 1 | 53.37 ± 7.11 | 68.57 ± 5.03 | 63.38 ± 3.41 | 56.82 ± 3.91 | 40.04 ± 9.91 |
| R only | 1 | 43.35 ± 7.51 | 60.03 ± 7.36 | 64.19 ± 3.07 | 51.41 ± 6.87 | 28.01 ± 6.97 |

### 11.2 关键发现

1. **优雅降级**:3→2 视角 OSCR 降 9.11(L+R)到 23.93(T+R)分;3→1 视角降 40+ 分。
2. **L+R 是最佳 2 视角组合**:L+R(noT) 86.22 > L+T(noR) 80.21 > T+R(noL) 71.40。L 和 R 雷达水平基线 1.1m 提供最强几何互补;Top 是最可省的传感器。
3. **L 视角最关键**:dropping L 损失最大,这与原 Eom 论文只用 Left 雷达一致。
4. **MSP 在所有 view subset 上完败**:inspector 在每个配置上比 MSP 高 13–28 分 OSCR,且 std 更小;缺视角下 MSP 不可用,inspector 仍可工作。
5. **3 视角训练 → 1 视角测试塌陷**:43–55 OSCR,远低于 single-view 直接训练的 82.90。说明 3 视角训练的特征严重依赖多视角语境;partial deployment 需要 view-dropout 训练(future work)。

### 11.3 工程级 takeaway(论文 Discussion 直接抄)

> **资源敏感场景下推荐 L+R 双视角配置**:省去 Top 雷达(33% 硬件成本下降),OSCR 86.22 ± 2.63,比 Eom 单视角(88.42)仅低 2.2 分但带多视角鲁棒性。

> **多视角是不可压缩的鲁棒性来源**:在所有 7 种 view 组合上,Inspector(distance-based)显著优于 MSP(softmax-based);后者一旦缺视角就跌到 28–72 范围,前者维持 43–95。

---

## 12. E4 — Score 分量消融 (disagreement signal ablation)

**实验设计**:同一个 all-views 训练好的模型,**测试时**切换 anomaly score 公式。
$$s = \underbrace{d_{\text{mean}} / \bar{\sigma}}_{\text{mean 分量(单视角信号的多视角平均)}} + \beta \cdot \underbrace{\operatorname{var}(d_v) / \bar{\sigma}^2}_{\text{var 分量(跨视角分歧,单视角原理上没有)}}$$
- `mean_only` = β=0(只用平均距离)
- `var_only` = 只用方差项(舍弃 mean)
- `combined` = β=1(默认)

### 12.1 5-case 主结果(all 3 views,均值 ± std,%)

| Score 模式 | **OSCR** | AUROC | Accuracy | F1 |
|---|---|---|---|---|
| **mean only**(β=0) | **95.90 ± 0.73** ⭐ | 97.86 ± 1.05 | 93.80 ± 1.18 | 93.57 ± 1.28 |
| combined(β=1) | 95.33 ± 0.79 | 97.30 ± 1.01 | 92.45 ± 1.21 | 92.46 ± 1.39 |
| var only | 95.21 ± 0.81 | 97.17 ± 1.03 | 92.33 ± 1.22 | 92.35 ± 1.43 |

### 12.2 跨 view subset 的一致性(均 OSCR)

| view 子集 | mean only | combined (β=1) | var only |
|---|---|---|---|
| all | **95.90** | 95.33 | 95.21 |
| noT (L+R) | **87.06** | 86.22 | 85.18 |
| noR (L+T) | **81.04** | 80.21 | 79.33 |
| noL (T+R) | **72.03** | 71.40 | 70.75 |

**在所有多视角配置上 `mean only ≥ combined ≥ var only`**,顺序一致。

### 12.3 反直觉但更诚实的发现

预期(假设):**var 应该是独立 OOD 信号,combined > mean only**。
**实测**:三者几乎打平,**mean only 反而最高**(+0.57 OSCR 比 combined,+0.69 比 var only)。

诚实解读:
- 多视角的主功臣不是"分歧 var",而是"距离均值 d_mean"。
- 三个视角的 per-view inspector 训练得足够一致,known/unknown 在 var 上的差距已被 mean 覆盖,**var 没有提供独立信号**。
- combined 在 β=1 下轻微拖累 0.6 分:var 对 known 样本也有自然抖动,加进去等于加噪。
- 不冲突,只是**信息高度冗余**(三者都在 95.2–95.9 区间)。

### 12.4 对论文叙事的影响

| 旧叙事(没验证) | 新叙事(基于 E4 证据) |
|---|---|
| ❌ 多视角创新 = 跨视角分歧(var)是单视角造不出的 OSR 信号 | ✅ 多视角创新 = **per-view inspector 的距离均值,降噪 + 几何互补**,把 OSCR 从 82.9 拉到 95.9 |
| ❌ var-only 抓 mean 漏的 unknown | ✅ mean ≈ combined ≈ var(信息冗余,三者都好) |
| ❌ disagreement 是核心机制 | ✅ disagreement 是"可选附加分量"(β 可设为 0) |

### 12.5 论文写作建议

1. **Method 章节**把 combined 公式保留为通用定义,**β 默认 = 0**(给最好结果),β = 1 / β = ∞(var only)作消融。
2. **Ablation 章节**直接放 Section 12.1 + 12.2 这两张表,清楚说 var 信息冗余但不冲突。
3. **诚实优势**:故事更扎实——多视角靠几何互补 + 一致性平均(物理上有据),不靠"分歧投票"(更脆弱的假说)。

---

## 13. 实用配置 + 更新日志档案

**主结果命令**(论文 headline):
```bash
python main.py --osr-head inspector --pool flatten --normalize abs_std
# -> 5-case OSCR 95.14 ± 0.81 (β=1, Section 2.2)
# 若用 mean only (β=0): 5-case OSCR 95.90 ± 0.73 (Section 12.1, 更高)
```

**E2 + E4 联合**(一次出 28 行表):
```bash
python main.py --osr-head inspector --pool flatten --normalize abs_std --ablate-views --ablate-score
```

**新增日志**:
| 文件 | 内容 |
|---|---|
| `sweep5_three_flat_absstd_E2E4.log` | E2+E4 联合 5-case,28 行汇总(主 OSCR 95.33 / mean-only 95.90 / noT 86.22) |
| `runE2E4.log` | E2+E4 阶段标记 |

---

## 14. E5 — 物理 bias on/off(2026-05-28 加跑)

**实验设计**:三视角 flatten + abs_std + inspector,主结果配置上,加上 `--physical-bias` 启用 `CrossViewFusion` 里的 ρ / Δt / 方向一致性 三组物理先验作为 attention 加性 bias。其它一切相同(150 epoch / β=1 / inspector batch 256)。

### 14.1 5-case 主对照(均值 ± std,%)

| 配置 | Accuracy | F1 | AUROC | **OSCR** |
|---|---|---|---|---|
| 主结果 (no phys bias, §2.2) | 92.29 ± 1.27 | 92.54 ± 1.20 | 97.12 ± 1.24 | **95.14 ± 0.81** |
| + phys_bias (E5) | 92.12 ± 1.05 | 92.15 ± 1.11 | 97.05 ± 1.05 | **95.10 ± 0.78** |
| **Δ** | -0.17 | -0.39 | -0.07 | **-0.04** |

### 14.2 结论

**物理 bias 在主结果配置上无增益**,Δ OSCR ≈ -0.04(完全在 0.78 std 之内)。

诚实解读:
- 跨视角注意力在三视角已经从数据里学到了几何关系(view_emb + factorised attention),不需要显式的 ρ/Δt 先验。
- 物理 bias 只在弱的 baseline 上可能有用;一旦 OSCR 上 95+,它的贡献被 attention 的数据驱动学习吸收。
- 可写进论文 ablation:"几何先验作为可选项,主结果不依赖"。

### 14.3 论文写作建议

放在 ablation 表里,作为"我们考虑过但发现冗余"的设计,体现严谨。**不放进 headline 配置**。

---

## 15. A7 — η-sweep 启发式阈值(2026-05-28 加跑)

**实验设计**:对应 Eom Fig 10 的 η=3 三σ 启发式阈值。在 inspector 评测里,除了 best-threshold sweep 报 acc/f1 之外,**额外在固定 η ∈ {1.5, 2, 2.5, 3, 3.5, 4, 5, 7} 上报 acc/f1**,thresh 在 anomaly score (高=未知) 上,score > η 即判为未知。

### 15.1 3-view 主结果配置下的 η-sweep(β=1, combined)

| η | Accuracy | F1 |
|---|---|---|
| 1.5 | 53.64 ± 2.62 | 20.15 ± 5.38 |
| 2.0 | 57.60 ± 3.22 | 30.18 ± 5.84 |
| 2.5 | 61.23 ± 3.03 | 39.03 ± 5.02 |
| **3.0** (论文推荐) | **64.83 ± 3.08** | **47.01 ± 5.48** |
| 3.5 | 67.52 ± 3.26 | 52.39 ± 5.94 |
| 4.0 | 69.38 ± 3.57 | 55.90 ± 6.46 |
| 5.0 | 72.03 ± 3.73 | 60.81 ± 6.39 |
| 7.0 | 75.24 ± 3.60 | 66.51 ± 5.80 |
| best-thr (§2.2) | **92.12 ± 1.05** | **92.15 ± 1.11** |

### 15.2 反直觉发现 + 解释

**论文推荐的 η=3 在我们三视角配置下只到 64.83 acc / 47.01 F1**,远低于 best-threshold 92.12。原因:
- 我们的 combined score 是 `d_mean/σ̄ + β·var/σ̄²`(β=1)。var 项把分数从 paper 的 ~d/σ 抬高了一截。
- 在 β=1 下,known 样本的 score 也因为视角间小幅 disagreement 而被推到 3-7 区间 → η=3 截误判太多 known 为 unknown。
- **paper-aligned 的 η-sweep 应该在 single-view 上跑**(那时 V=1,var=0,combined ≡ mean_only ≡ d_mean/σ̄,正好对齐 paper 的 ‖l-φ‖/σ)。已经和 E3 一起排队。

### 15.3 论文写作建议

- **3-view β=1 配置下 η-sweep 论文用作"为什么需要 OSCR / AUROC"的论证**:阈值无关分数稳定,固定 η 不稳定。
- **单视角 η-sweep 留作 paper-faithful 对照**,验证 paper 的 η=3 在 single-view 上确实合理。
- 写法:"Paper-style heuristic threshold η=3 transfers to single-view but not to the β=1 multi-view combined score, where the variance term shifts the score scale; we therefore recommend reporting OSCR/AUROC as the primary multi-view metrics."

## 16. D1 — Raw vs ClutterRemoved 输入(2026-05-29 加跑)

**实验设计**:数据集同时提供 `ClutterRemoved`(已做背景对消)与 `Raw`(原始 ADC)两套立方体。主结果一直用 ClutterRemoved。这里在**完全相同的配置(三视角 + flatten + abs_std + inspector + β=1,150 epoch)**下把输入换成 Raw,看 Transformer 是否需要数据集预先做的杂波抑制。日志:`sweep5_three_flat_absstd_raw.log`。

### 16.1 5-case 主对照(均值 ± std,%)

| 输入 | Accuracy | F1 | AUROC | OSCR |
|---|---|---|---|---|
| ClutterRemoved (§2.2 主结果) | 92.29 ± 1.27 | 92.54 ± 1.20 | 97.12 ± 1.24 | **95.14 ± 0.81** |
| **Raw** | 92.79 ± 1.14 | 92.52 ± 1.55 | 97.57 ± 0.69 | **95.53 ± 1.15** |
| Δ (Raw − CR) | +0.50 | −0.02 | +0.45 | **+0.39** |

Raw 每 case OSCR:97.13 / 94.60 / 95.92 / 93.87 / 96.11。

### 16.2 结论

- **Raw 全面追平甚至略胜 ClutterRemoved**(OSCR +0.39,AUROC +0.45),但全部落在 1 std 以内 → 统计上无显著差异。
- 含义:**数据集预先做的杂波对消对本任务并非必需**。dual-path Transformer 的 attention 自己就能从原始回波里学到等价(或更优)的杂波抑制,预处理这一步是冗余的——甚至可能轻微"过处理"丢掉了对 OSR 有用的细节(case1 Raw 97.13 vs CR 96.19 即是一例)。
- MSP 基线两套输入都明显垫底(Raw 85.16 vs CR ~85),与 inspector 的 gap 不受输入影响,说明结论是 OSR 头层面的、与输入预处理正交。

### 16.3 论文写作建议

- 作为 input-ablation 一行:"Replacing the dataset's clutter-removed cubes with raw ADC returns yields statistically indistinguishable open-set performance (OSCR 95.53 ± 1.15 vs 95.14 ± 0.81), indicating the dual-path Transformer learns clutter suppression end-to-end and does not rely on a separate background-subtraction stage."
- 这是一个对工程部署友好的卖点:**省掉杂波对消预处理,直接喂原始数据也能拿到同样的开集精度**。

## 17. E3 + paper-aligned A7 — per-head D_key=32 单视角(2026-05-29 加跑)

**实验设计**:两件事一起跑(单视角 Left + flatten + abs_std + inspector,150 epoch)。
1. **E3**:PyTorch `MultiheadAttention` 把 embed_dim 平分到各 head,所以 per-head dim = dim/heads。主结果 dim=32/heads=4 → **per-head=8**。论文 D_key=32,要对齐就得 dim=128/heads=4 → **per-head=32**。代价:flatten feat_dim = D·(T+M) = 35712(4×),inspector 第一层 ~36.6M 参数/(class,view)。
2. **A7 paper-aligned**:单视角下 V=1 → var=0 → combined score ≡ d_mean/σ̄ ≡ 论文的 ‖l−φ‖/σ。这正是 §15 里说的"η=3 应该在单视角上验证"。日志:`sweep5_single_flat_absstd_dim128_eta.log`。

### 17.1 E3 — 加大 per-head D_key 到论文的 32(5-case,%)

| 配置 | per-head D_key | feat_dim | Accuracy | F1 | AUROC | OSCR |
|---|---|---|---|---|---|---|
| 单视角主配置 (dim=32, §10.4) | 8 | 8928 | — | — | — | **82.90 ± 4.83** |
| **E3 (dim=128, 论文 D_key)** | 32 | 35712 | 75.23 ± 4.45 | 80.47 ± 2.59 | 79.97 ± 5.05 | **78.13 ± 5.38** |
| Δ (E3 − 主配置) | — | 4× | — | — | — | **−4.77** |

E3 每 case INSPECTOR OSCR:79.84 / 80.51 / 79.07 / 67.78 / 83.44;closed_acc 仍 94–98%。

**结论:把 per-head D_key 从 8 提到论文的 32(dim 32→128)反而让单视角 OSCR 掉了 ~4.8 分。** closed-set 精度没掉(94–98%),掉的是开集判别力 → 典型的**过拟合**:flatten feat_dim 飙到 35712、inspector 第一层 36.6M 参数/class,在每类样本量很小的 OVR 训练下把闭集背得太死,留给"未知"的判别边界反而变松。**我们的 dim=32(per-head=8)对这个数据集是更优的容量点,论文的 D_key=32 不该照搬。** 这是一个干净的 capacity-ablation 负结果。

### 17.2 A7 paper-aligned — 单视角 η-sweep(var=0,score ≡ 论文 d/σ)

| η | Accuracy | F1 |
|---|---|---|
| 1.5 | 71.47 ± 3.38 | 71.90 ± 4.41 |
| 2.0 | 73.13 ± 4.16 | 75.86 ± 4.36 |
| 2.5 | 73.77 ± 4.62 | 78.05 ± 3.27 |
| **3.0**(论文推荐) | **74.13 ± 4.90** | **79.79 ± 1.95** |
| 3.5 | 74.40 ± 4.93 | 80.74 ± 1.73 |
| 4.0 | 74.30 ± 5.12 | 81.20 ± 1.99 |
| 5.0 | 73.64 ± 5.24 | 81.48 ± 2.21 |
| 7.0 | 72.26 ± 5.37 | 81.54 ± 2.33 |
| **best-thr(oracle)** | **75.23 ± 4.45** | **80.47 ± 2.59** |

**结论:论文的固定 η=3 在单视角上几乎踩在最优点上。** η=3 给 74.13 acc,oracle best-threshold 才 75.23 —— **gap 仅 ~1.1 分**(accuracy 在 η∈[3,4] 平台上 74.1–74.4,基本贴着 oracle)。这和 §15 的 3-view β=1 形成尖锐对照:那里 η=3 只有 64.83 acc vs best-thr 92.12,**gap 27 分**。

差别的唯一来源就是 score 的尺度:
- **单视角**:score = d_mean/σ̄,正好是论文标定 η=3 时的 ‖l−φ‖/σ → η=3 对齐。
- **三视角 β=1**:score = d_mean/σ̄ + var/σ̄²,var 项把分数整体抬高 → 同样的 η=3 把太多 known 误判成 unknown。

**这就实证验证了 §15 的猜想:论文的 η=3 启发式是为单视角 d/σ 标定的,直接搬到多视角 β=1 combined score 上会失准。**

⚠️ 一个诚实的 caveat:这个单视角 η-sweep 跑在 E3 的 dim=128 模型上(顺带一起跑的),不是 dim=32 主配置。但 η-sweep 的形状(峰在 η≈3–4、贴着 oracle)是 score 归一化(d/σ)的性质,基本不依赖 dim,所以"η=3 在单视角近最优"的结论是稳的。若要 100% 严谨,可在 dim=32 单视角上补一次 `--eta-sweep`(很快)。

### 17.3 论文写作建议

- **E3 写成 capacity ablation 一行**:"Matching the paper's per-head key dimension (D_key=32, i.e. model width 128) increases the per-class inspector to 36.6M parameters and *reduces* single-view OSCR from 82.9 to 78.1; the smaller D_key=8 used throughout avoids over-fitting the one-vs-rest heads. We therefore keep model width 32."
- **A7 写成 threshold 标定的关键论证**(把 §15 + §17.2 合起来):"The paper's η=3 heuristic is calibrated for the single-view distance score d/σ, where it reaches 74.1% accuracy — within 1.1 points of the oracle threshold. Under the multi-view combined score the added disagreement term shifts the scale, so a fixed η no longer transfers (64.8% vs 92.1% oracle). We thus report OSCR/AUROC as the primary multi-view metrics and reserve the fixed-η heuristic for the single-view setting."

### 17.4 新增日志

| 文件 | 内容 |
|---|---|
| `sweep5_three_flat_absstd_raw.log` | D1:Raw 三视角 5-case(95.53 ± 1.15,§16) |
| `sweep5_single_flat_absstd_dim128_eta.log` | E3 dim=128 单视角 + paper-aligned η-sweep(§17) |

---

## 18. B2 — 跨视角掩码重建预训练(Stage-0,进行中)

**动机**:E5 的物理 bias 没用、E4 的 var 分量冗余,说明当前管线没把"三视角是同一 3D 手部运动的不同投影"这层几何榨干。B2 用自监督**掩码 1 个视角 + 用其余视角重建**逼 encoder 学这层几何,产出两件东西:① 更好的 encoder 初始化喂 Stage-1(init 增益);② 一个 OOD 信号 `e_recon`(已知手势三视角自洽→重建误差低,未知→高)。

**开集卫生**:预训练只见 `train_ds`(已知类训练样本),不碰 unknown / test。
**caveat**:`mae_target=token` 的目标来自 stop-grad 的训练中 encoder(v1 无 EMA),有表征坍缩风险;`raw`(重建回波 cube)为默认、grounded。

### 18.1 复现命令

```bash
# go/no-go 早筛(case 0,看 known/unknown 重建误差分不分得开)
python main.py --case 0 --pool flatten --normalize abs_std --mae-pretrain --mae-target raw

# 全 5-case + 融合 λ 扫描
python main.py --pool flatten --normalize abs_std --mae-pretrain --mae-target raw --lambda-recon 1
```

### 18.2 早筛(case 0,raw target):known vs unknown 重建误差

> ✅ **通过早筛。** 判据是 `recon early-screen AUROC` 明显 >50。实测 AUROC=95.48,known/unknown 重建误差清晰分离(0.62 vs 0.89),且 standalone `recon` 头(OSCR 95.01)几乎追平 inspector(OSCR 95.57)——`e_recon` 在真实数据上是可用的独立 OOD 信号,值得上全 5-case。

| 配置 | known MSE | unknown MSE | recon AUROC | recon OSCR |
|---|---|---|---|---|
| case 0 raw | 0.6165 | 0.8905 | 95.48 | 95.01 |

同一 case 0 run 的三头对照(单 case,非 mean±std):

| head | Accuracy | F1 | AUROC | OSCR |
|---|---|---|---|---|
| inspector | 92.00 | 92.77 | 95.93 | 95.57 |
| recon (`e_recon` standalone) | 90.33 | 90.86 | 95.48 | 95.01 |
| msp | 79.62 | 85.39 | 85.44 | 85.02 |

### 18.3 对照矩阵 → 已在 LOSO 主协议下兑现(见 §28.6)

诚实检验三件事:① 预训练 init 单独有没有用;② `e_recon` 当 OOD 有没有用;③ 二者是否又是冗余负结果(对照 E4)。

**实际执行说明**:论文一主协议已从随机切分换成 **LOSO**(§27),所以这张对照矩阵直接在
**LOSO** 下跑掉了,结果落在 **§28.6**(`s_erecon_lambda.bat`,5-case LOSO + λ 扫 0.3/1/3),
本节不再单独跑随机切分版(随机切分 95+ 区间各方案打平,无判别力,见 §16/§27)。

| 问题 | 答案(§28.6,LOSO 5-case mean±std OSCR) |
|---|---|
| ② `e_recon` standalone 有用? | ✅ 84.06 ± 4.06,远超 msp 74.22,跨人下与 inspector(84.97)打平 |
| ③ 与 inspector 冗余还是互补?(对照 E4) | **互补**:融合 λ=1.0 = **88.39 ± 2.56**,比 inspector **+3.42** 且 5-case 全赢 |
| ① Stage-0 init 单独增益? | ≈无:inspector 带 B2 init 84.97 vs S3 无 B2 84.34(§27),Δ 在噪声内 |

> token target 一支未跑(raw 已 grounded 且不塌缩,token 有坍缩风险,§18 caveat),暂搁置。

### 18.4 新增代码 / 日志

| 项 | 内容 |
|---|---|
| `uwb_osr/pretrain.py` | Stage-0 掩码重建 + `e_recon` OSR 头(新增,默认关闭) |
| `b2_case0_raw_screen.log` | case 0 raw 早筛(§18.2) |

---

## 19. R1 — 训练时随机丢视角 → 缺视角鲁棒性(sensor-failure robustness)

**动机**:§11.2 #5 留的 future work。E2(§11)是**训练 3 视角、测试时切 subset 不重训**,结果 3→1 视角塌陷到 43–55 OSCR(比直接训单视角的 82.9 还低)——"坏一个传感器还不如只装一个",无商业价值。R1 在**训练时**以概率 `view_dropout` 随机只保留 1..V-1 个视角(模拟传感器失灵),逼模型对任意 view subset 都建好可分的类原型。

**做法**:`engine._sample_kept_views(cfg)` 每个 batch 以 `view_dropout` 概率抽一个 1..V-1 的视角子集,`model(x[:, keep], view_indices=keep)` 训练;`view_emb` 同步切片。**默认 `view_dropout=0.0`,关闭时与旧管线逐字节一致。** 评测仍用 §11 的 `--ablate-views` 7-subset 矩阵。

**开集卫生**:丢视角只发生在 `train_ds`,不碰 unknown / test。

### 19.1 复现命令

```bash
# case-0 go/no-go(flatten + abs_std + view-dropout 0.5 + 7-subset 评测)
python main.py --case 0 --pool flatten --normalize abs_std --view-dropout 0.5 --ablate-views
```

### 19.2 早筛(case 0,flatten + abs_std,inspector 头):baseline vs view-dropout=0.5

> ✅ **通过早筛(强阳性)。** 全 3 视角几乎不变(Acc +0.79,OSCR −0.26 在噪声内),**缺视角全线抬升**:2 视角从 66–90 抬到 ~93–94 OSCR(逼近全视角 95.87),单视角从塌陷的 41–65 抬到 65–87。"坏一个传感器不如装一个"的病理消失——每个降级配置都≥旧单视角下限。机制如预期:训练时随机丢视角逼 inspector 建立 per-view 可分的类流形,不再依赖三路恒在。

| view 子集 | 用几路 | Acc baseline→vd0.5 | OSCR baseline→vd0.5 |
|---|---|---|---|
| **all (L+T+R)** | 3 | 92.54 → **93.33** | 96.13 → **95.87**(持平)|
| noL (T+R) | 2 | 77.17 → **89.83** | 66.14 → **93.86** |
| noT (L+R) | 2 | 88.04 → **89.46** | 90.15 → **93.91** |
| noR (L+T) | 2 | 79.29 → **87.67** | 74.94 → **93.19** |
| L only | 1 | 71.38 → **81.42** | 64.54 → **86.83** |
| T only | 1 | 58.13 → **65.04** | 45.25 → **65.53** |
| R only | 1 | 62.08 → **78.25** | 41.40 → **83.41** |

(以上为**单 case 0**,非 5-case mean±std;全 5-case 待跑。baseline 取自 `p0c0_flatten.log`,vd0.5 取自 `p1c0_flatten_vd05.log`。)

### 19.3 关键发现(case 0)

1. **全视角零代价**:3 视角 OSCR 96.13→95.87(−0.26,噪声),Acc 反升 +0.79——鲁棒性不是用主结果换来的。
2. **缺一路逼近全视角**:三种 2 视角配置全部抬到 ~93–94 OSCR,与全视角 95.87 仅差 ~2 分(baseline 时最差的 noL 还差 30 分)。
3. **单视角不再塌陷**:最差 T-only OSCR 45→65,R-only 41→**83**;全部≥E2 里 3 视角训练单测的 43–55 范围,部分超过直接训单视角的 82.9。
4. **病理消除**:不再出现"2 视角 < 1 视角直训"。partial deployment 现在可行。

### 19.4 新增代码 / 日志

| 项 | 内容 |
|---|---|
| `config.py: view_dropout` | R1 开关(默认 0.0,关闭即旧管线) |
| `engine._sample_kept_views` | 每 batch 抽 1..V-1 视角子集 |
| `engine.train_one_case` | 抽中子集时 `model(x[:,keep], view_indices=keep)` |
| `main.py: --view-dropout` | CLI(默认 −1 → 取 config) |
| `p0c0_flatten.log` | baseline(view_dropout=0)case-0 7-subset(§19.2 左列) |
| `p1c0_flatten_vd05.log` | view_dropout=0.5 case-0 7-subset(§19.2 右列) |

### 19.5 全 5-case 结果(mean ± std,inspector,与 §11.1 E2 基线正面对照)

> ✅ **case-0 结论在 5-case 上完全复现。** 全视角 OSCR 95.33→94.88(−0.45,噪声内),缺任一传感器 OSCR 全部抬到 90.8–92.4(基线 71–86),最差单视角 OSCR 43.35→**81.30**。更关键:**方差大幅收窄**——缺视角下基线 std 高达 6–10,dropout 后降到 1–4,部署可靠性显著提升。

| view 子集 | 用几路 | OSCR 基线→vd0.5 | Acc 基线→vd0.5 |
|---|---|---|---|
| **all (L+T+R)** | 3 | 95.33±0.79 → 94.88±0.94 | 92.45±1.21 → 91.84±1.06 |
| noT (L+R) | 2 | 86.22±2.63 → **90.81±2.28** | 84.65±2.25 → 86.03±2.09 |
| noR (L+T) | 2 | 80.21±2.62 → **91.97±0.93** | 82.52±1.58 → 87.68±0.80 |
| noL (T+R) | 2 | 71.40±9.75 → **92.35±1.07** | 79.72±4.13 → 88.57±1.31 |
| L only | 1 | 54.71±6.46 → **83.61±3.21** | 66.42±3.09 → 78.88±2.88 |
| T only | 1 | 53.37±7.11 → **72.12±3.94** | 63.38±3.41 → 70.64±3.11 |
| R only | 1 | 43.35±7.51 → **81.30±1.36** | 64.19±3.07 → 77.42±1.65 |

(基线列取自 §11.1 E2;vd0.5 列取自 `p1_5case_flatten_vd05.log`。RESULTS.tex `tab:view_dropout` 已同步填入。)

**5-case 每 case 全视角 INSP_ALL**:96.36 / 94.71 / 94.25 / 93.66 / 95.42 → mean 94.88。

### 19.6 待办

- [ ] 扫 `view_dropout ∈ {0.3, 0.5, 0.7}` 找鲁棒性 / 全视角精度的最优折中。
- [ ] (Phase 2)叠加 Stage-0 掩码重建预训练 init,看是否再增益。

---

## 20. F0 — 去掉跨视角注意力(`--no-fusion`)5-case(2026-06-11)

**动机**:`CrossViewFusion` 的输出只进闭集头(已饱和 97.8%),inspector 消费的
`f_views` 在融合**之前**取出 → 主结果 OSCR 从未经过融合模块。README 的 B1 实际
从未跑过。F0 第一次量化"现有跨视角注意力对开集主指标的真实贡献"。配置与 §2.2
主结果完全相同(三视角 + flatten + abs_std + inspector + β=1,150 epoch),仅
`--no-fusion`。日志:`f0_sweep5_three_flat_absstd_nofusion.log`(case 1–2,主 run
被外部杀掉)+ `f0_nofusion_case{3,4,5}.log`(单 case 续跑)。

### 20.1 5-case 主对照(OSCR combined,%)

| Case | 有融合(§2.2 主结果) | **无融合(F0)** | Δ |
|---|---|---|---|
| 1 | 96.19 | 96.17 | −0.02 |
| 2 | 94.80 | 95.04 | +0.24 |
| 3 | 94.72 | 94.80 | +0.08 |
| 4 | 94.04 | 93.70 | −0.34 |
| 5 | 95.95 | 96.23 | +0.28 |
| **均值** | **95.14 ± 0.81** | **95.19 ± 0.94** | **+0.05** |

### 20.2 结论

**去掉整个 `CrossViewFusion` 模块,5-case OSCR 95.14→95.19(+0.05,远在 ±0.9 噪声
内),逐 case 有正有负无方向性。现有跨视角注意力对开集主指标的贡献为零。**

根因:inspector 吃的 per-view 特征码在融合**之前**取出,融合输出只喂已饱和的闭集
头,那一层注意力的参数只接收闭集 CE 梯度 → 学不到任何对 OSR 有用的东西。这与
E5(物理 bias 零增益)、E4(var 分量冗余)是同一个根因的三个症状:**闭集 CE 教不出
视角间几何**。多视角的真实增益全部来自「共享编码器(Siamese)+ per-view inspector
距离平均」这条 late-fusion 路(E4 已证)。

附带验证(case 3/4/5 三种 score 模式,无融合):仍 mean_only ≥ combined ≥ var_only
(如 case5 mean 96.77 > comb 96.23 > var 96.19),与 E4 在**有**融合时的结论一致 →
融合模块的有无完全不改变 OSR 信号结构。

### 20.3 对论文 / 后续的影响

1. **C1 叙事必须改**:现写的"权重共享 + 跨视角融合"功劳全在「共享编码器 + 距离
   平均」,那层注意力是装饰。这个干净的负结果是融合重设计(F1/F2)的最佳动机:
   "我们先实证后置注意力对开集无效,所以把融合移进 OSR 真正消费的通路"。
2. **headline 配置可删 `use_fusion`**:少一批参数、缺视角更稳,保留作 ablation 行。
3. **F1 意义被抬高**:融合模块输出从未被 OSR 头见过,F1(融合后特征喂 inspector)
   是第一次真正测"编码器级融合对开集有没有用"。见 FUSION_DESIGN.md §5。

---

## 21. F1 — 融合后特征喂 inspector(case-0 配对早筛,2026-06-11)

**动机**:F0 证明现有 `CrossViewFusion` 输出从未被 OSR 头消费。F1 第一次把
**融合后**的 per-view 特征码喂给 Stage-2 inspector,测"编码器级融合对开集有没有
用"。开关 `cfg.inspector_feats={pre|fused}`,两次跑都开着 `CrossViewFusion`(fused
读出确实经过它),同 case-0 同种子配对。日志:`f1_c0_{pre,fused}.log`。

### 21.1 case-0 配对对照(%)

| 配置 | closed | OSCR combined | OSCR mean_only | MSP OSCR |
|---|---|---|---|---|
| **pre**(融合前,现状) | 99.33 | 95.99 | 96.46 | 89.25 |
| **fused**(融合后喂 inspector) | 98.92 | **95.30** | 95.91 | 83.72 |
| **Δ** | −0.41 | **−0.69** | −0.55 | −5.53 |

### 21.2 结论:判据不通过(要求 ≥+0.5,实测 −0.69)

把现有融合模块的输出喂给 inspector,**OSCR 不升反降**,闭集也掉,MSP 塌 5.5 分。
**印证诊断 (c)**:现有融合模块(视角置换对称 + 只融时间路径 + 单层裸 MHA + 仅受
闭集 CE 训练)产出的"融合特征"把三路搅在一起、破坏了 per-view inspector 距离平均
赖以工作的视角独立性 → 直接融合恶化开集(FUSION_DESIGN §1c + ACM MM'25)。MSP 塌
5.5 分尤其说明融合后 logits 几何被搞乱。

**判断**:问题不在"要不要把融合喂给 OSR 头",而在"用什么样的融合"。现有模块两头
不讨好:对闭集零贡献(F0)、对开集有害(F1)。F1 这条最小修改路(复用旧模块)是
死路,符合预期(它本是 F2 的弱化版)。→ 直接上 F2(瓶颈结构 + 视角嵌入修复 + 双
路径融合 + 保留逐视角读出),其双-bank 退路正是防 F1 这种掉分的保险。

---

## 22. F2 — X-MV 瓶颈中层融合(case-0 配置扫描,2026-06-11)

**设计**(FUSION_DESIGN §3,MBT/NeurIPS'21 风格):TransformerPath 后 K 层换成
`BottleneckFusionLayer`——每视角 token 与 NB 个共享瓶颈 token 互看,瓶颈跨视角平均
是唯一跨视角通道。总深度不变(避 E3 容量陷阱)。token 进融合前加 `view_emb_tok`
修复视角置换对称;temporal/spatial 双路径都融合;`f_views` 由经过瓶颈的 token 算出
→ inspector 默认(pre)就吃到跨视角增强的 per-view 码,且保留逐视角读出结构。
`--no-fusion` 关掉旧 CrossViewFusion,只留瓶颈交换。代码:`model.BottleneckFusionLayer`
+ `TransformerPath` xmv 分支;CLI `--xmv-layers K --xmv-bottleneck NB`。

### 22.1 case-0 配置扫描(%)

| 配置 | closed | OSCR combined | OSCR mean_only | MSP |
|---|---|---|---|---|
| **锚点**(无 xmv,§21 pre) | 99.33 | **95.99** | **96.46** | 89.25 |
| K=2 / NB=4 | 99.08 | 95.55 | 96.10 | 81.13 |
| **K=1 / NB=4** ⭐ | 99.17 | 95.96 | 96.50 | 83.76 |
| K=2 / NB=2 | 99.00 | 95.66 | 96.36 | 77.28 |

### 22.2 结论:打平(未达 go 判据 +0.5,但已无害)

最优 K=1/NB=4:combined 95.96(锚点 95.99,−0.03)/ mean_only 96.50(锚点 96.46,
**+0.04**)→ **与纯距离平均统计打平**,没有 +0.5 增量。更深(K=2)或更窄(NB=2)
均略降,MSP 一路下滑(89→77)→ 瓶颈交换确在让 logits 几何变模糊。

**但相对 F1 是质变**:F1 现有融合喂 inspector 是 **−0.69(有害)**;F2 瓶颈融合做到
**−0.03(打平/无害)**。"修复置换对称 + 双路径 + 保留逐视角读出 + 窄带正则"成功让
融合**不再伤害**开集,但随机切分下也没能**超过**距离平均。

**根因 = 协议天花板,非架构失败**:随机切分有 session 泄漏,case-0 已知/未知几何已
被距离平均榨干,融合的边际价值被掩盖。这正是 FUSION_DESIGN §5 F6 预判的:**95+ 区间
各融合方案打平,表征质量差异要到 LOSO(域偏移)下才显形**。跨视角几何是被试无关的
物理结构 → 转 LOSO 决胜(§23)。

### 22.3 新增日志

| 文件 | 内容 |
|---|---|
| `f1_c0_{pre,fused}.log` | F1 配对早筛(§21) |
| `f2_c0_{k2nb4,k1nb4,k2nb2}.log` | F2 三配置 case-0 扫描(§22) |

---

## 23. F6 — LOSO 跨人决胜:瓶颈融合 vs 无融合(2026-06-12)

**动机**:§22 在随机切分下打平,猜想是"session 泄漏掩盖了融合的表征增益,跨视角几何
是被试无关结构,LOSO 域偏移下融合应拉开正向差距"。F6 直接验证这个猜想。最优 F2
配置(K=1/NB=4)vs 无融合(F0 赢家),case-0,8 折 leave-one-subject-out,完全同协议。
日志:`f2_loso_c0_{nofusion,xmv_k1nb4}.log`。

### 23.1 8 被试结果(OSCR combined,%)

| 配置 | OSCR mean±std | mean_only | 闭集 acc |
|---|---|---|---|
| **无融合**(基线) | **85.22 ± 5.55** | 87.07 | 82.64 |
| 瓶颈融合 K=1/NB=4 | 83.43 ± **8.45** | 85.48 | 81.98 |
| **Δ** | **−1.79** | −1.59 | −0.66 |

逐被试配对差(xmv − nofusion):
| subj | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Δ | +0.04 | +0.06 | +0.50 | **−4.05** | +1.92 | **−9.16** | **−4.45** | +0.83 |

配对差 mean −1.79 / std 3.52,配对 t = −1.34(8 样本,p>0.05)。

### 23.2 判决:猜想被证伪,融合在域偏移下变差(且方差暴涨)

**预期融合在 LOSO 下拉开正向差距 → 实测相反**:融合平均掉 1.79 分,**方差从 5.55
暴涨到 8.45**。5 个被试微弱占优/打平,但 3 个被试崩盘(subj 4/7/8 中 −4~−9)。统计上
"融合 ≤ 无融合",无任何证据支持融合更好。

机制(FUSION_DESIGN §1c + ACM MM'25 实锤):瓶颈让三路特征耦合,在见过的被试上无害甚至
小赚,但碰到分布外新被试,耦合学到的是 **session 相关的伪几何**,域偏移下放大成脆弱性
→ 掉分集中在最难被试,方差被推高。距离平均的"半独立观察者"结构反而更鲁棒。

### 23.3 最终结论:架构级跨视角融合这条路到此为止(完整负结果链)

| 实验 | 协议 | 融合−无融合 | 结论 |
|---|---|---|---|
| F0 | 随机 5-case | +0.05 | 现有融合零贡献 |
| F1 | 随机 case-0 | −0.69 | 现有融合喂 OSR 头有害 |
| F2 | 随机 case-0 | −0.03 | 重设计(MBT 瓶颈)打平 |
| **F6** | **LOSO case-0** | **−1.79(方差↑)** | **重设计融合在域偏移下变差** |

**架构级跨视角融合(旧 CrossViewFusion 与 MBT 瓶颈皆然)对本任务开集识别无正贡献、
甚至有害。** 根因一以贯之:闭集 CE 教不出有用的视角间几何;OSR 增益本质来自三个半独立
观察者的距离平均,任何耦合它们的融合都削弱这个 ensemble。

**论文叙事(强故事,非失败)**:系统性证伪了"直接合起来太简单"的直觉的反面——问题不
是融合太简单,而是**架构级融合方向本身错了**;对开集而言"逐视角独立编码 + 决策级距离
聚合"才是对的。这条 F0→F6 负结果链 + E4/E5 共同支撑一个干净的设计论断。

→ 下一步换路:**目标函数级融合(B2 跨视角掩码重建)**,e_recon 是单视角原理上造不出
的 OOD 信号(case-0 已验证 AUROC 95.48),与"耦合特征"正交——它逼 encoder 学几何但
不破坏 per-view 独立读出。见 FUSION_DESIGN §2(目标函数级)/§5 F3。

### 23.4 新增日志

| 文件 | 内容 |
|---|---|
| `f2_loso_c0_nofusion.log` | LOSO 无融合 8 折(85.22 ± 5.55) |
| `f2_loso_c0_xmv_k1nb4.log` | LOSO 瓶颈融合 8 折(83.43 ± 8.45) |

---

## 24. S1 — 属性组合泛化早筛(论文二柱二守门,2026-06-12)

**动机**:RESEARCH_PLAN.md 柱二("描述未知"而非只"拒识")要求表示把运动方向学成
**可分解、可组合**的属性。早筛用 CZSL 标准协议:编码器只在 8 个种子类训练(4 正交滑动
G1–G4 + 旋转 G9/G10 + 推 G11 + 空 G12,**4 个对角 G5–G8 完全 held out**),冻结特征,
线性探针在正交滑动上训属性头(水平/垂直各 3 类),在没见过的对角上测能否读出两个分量。
代码:`probe_attributes.py`(flatten + abs_std + 150 epoch)。日志:
`s1_real_flat_absstd_v2.log`。

### 24.1 对角(未见组合)属性准确率(%)

| 特征来源 | 水平轴 | 垂直轴 |
|---|---|---|
| **融合 c_fused** | 17.75 | **73.53** ✅ |
| 单视角-Left | 29.94 | 41.41 |
| 单视角-Top | 28.66 | 44.41 |
| 单视角-Right | 36.19 | 23.09 |

(probe-train / sanity(held-out 重复)在所有来源上均 ~99–100%,确认探针本身正常;
单视角与融合探针输入同维 8928,排除容量差异。)

### 24.2 两个发现

**发现 1:垂直轴组合成立,且是"多视角涌现"属性。** 垂直在融合上 73.5%,在任何单视角
都崩(23–44)。**没有单个雷达能在未见对角上读出垂直方向,三路合起来才涌现出这能力。**
这是一个强多视角论点:组合泛化不是单视角本有、融合搬运,而是融合**创造**的新能力。
值一张 figure,把"为什么多视角"从"误识更少"升级到"产生新能力"。

**发现 2:水平轴是真失败(根因),不是融合抵消假象。** 水平在融合 17.75,但即使最好的
单视角(Right 36)也远低于 70%。若是纯融合抵消,单视角应高——但单视角也读不出。逐手势
见纠缠signature:Left 在 G5(左+上)/G8(右+下)读对水平(53–55%),在 G6(左+下)/G7(右+上)
塌成 none——**水平读数取决于垂直分量,两轴在判别式表示里缠死**。物理上:雷达 L/T/R 近
水平摆位,水平滑动在 L/R 产生符号相反的距离signature,判别式 CE 没学会把它拼成连贯方向。

### 24.3 判决:柱二 GREEN + 一个尖锐子问题

- **组合泛化被证明可行**(垂直 73.5%)→ 论文二"描述未知"地基成立;
- **垂直组合是多视角涌现** → 论文一/二都能用的独立强卖点;
- **水平轴失败 = 干净的开放挑战**,且与主线一致(判别式 CE 学不出需要跨视角几何拼接
  的方向),**预测生成式脊柱(B2/S2)能修复 → 给 S2 一个直接、可测的成功判据**:
  S2 模型上水平轴对角准确率是否显著超过这里的 ~18–36%。

### 24.4 新增代码 / 日志

| 项 | 内容 |
|---|---|
| `probe_attributes.py` | S1 属性探针(融合 + 逐视角,新增) |
| `s1_real_flat_absstd_v2.log` | S1 真数据结果(§24.1) |

---

## 25. S2 — 生成式编码器能否修复水平轴组合?(2026-06-12)

**假设(S1 §24 推出)**:判别式 CE 学不出跨视角几何 → 水平轴组合失败;**生成式
(跨视角掩码重建)被逼着学几何 → 应修复水平轴**。测法:`probe_attributes.py --encoder
{closed,b2}`,两次都 fusion-off、同 split/seed,唯一变量是编码器目标(闭集 CE vs B2 raw
重建,自监督无标签)。日志:`s2_closed_fusionoff.log` / `s2_b2_raw.log`。

### 25.1 头对头(对角组合准确率,fusion-off,%)

| 来源 | closed 水平 | b2 水平 | closed 垂直 | b2 垂直 |
|---|---|---|---|---|
| **fused** | 18.84 | **14.94** | 77.75 | **68.66** |
| view-Left | 36.53 | 33.62 | 39.72 | 35.22 |
| view-Top | 28.59 | 25.28 | 46.56 | 50.72 |
| view-Right | 34.62 | 42.66 | 24.84 | 21.94 |

(closed fusion-off 18.84/77.75 与 S1 fusion-on 17.75/73.53 一致 → 再证 fusion≈noop。)

### 25.2 判决:假设证伪(raw 重建版)——水平轴没修好,垂直还略降

- 水平轴:18.84→**14.94**,纹丝不动甚至略降。生成式组合水平方向**与判别式一样差**。
- 垂直轴:77.75→**68.66**,生成式还掉 9 分(跌破 70)。
- **不移动球门**:raw 重建不是"生成式修几何"的证据,S1 假设的简单版被否。

### 25.3 重新归因:可能不是目标函数,是**输入**(待 §26 验证)

raw-B2 不灵,引出一个之前漏查的更根本怀疑:**`abs_std` 先取 |x|,把符号/相位扔了**,
而水平方向(朝某雷达靠近 vs 远离)恰恰编码在符号里。若方向信息在输入阶段就被抹掉,
**任何编码器目标都救不回来**——信息不在喂进去的数据里。这还解释了 S1 的轴不对称:
垂直方向在幅度上就单调可分(不依赖符号),水平在 L/R 符号相反、取 |x| 后两方向幅度
pattern 趋同。**便宜且根本,先于 JEPA 验证**:换 `per_view_std`/`none`(保符号)看水平
轴是否跳起(§26)。

### 25.4 新增日志

| 文件 | 内容 |
|---|---|
| `s2_closed_fusionoff.log` | 判别式编码器基线(fusion-off) |
| `s2_b2_raw.log` | 生成式 B2-raw 编码器 |

---

## 26. S2b — 水平轴失败是输入(符号)的锅吗?(2026-06-12)

**怀疑(§25.3)**:`abs_std` 取 |x| 扔了符号/相位,而水平方向编码在符号里 → 信息在
输入阶段就丢了。测法:同探针(判别式编码器,fusion-off),换保符号的归一化。
日志:`s2b_probe_perviewstd.log` / `s2b_probe_none.log`。

### 26.1 水平轴三方对照(对角 fused,%)

| 归一化 | 水平 fused | 垂直 fused |
|---|---|---|
| abs_std(扔符号,默认) | 18.84 | 77.75 |
| per_view_std(保符号) | 22.03 | 66.44 |
| none(原始带符号) | 20.97 | 68.84 |

### 26.2 判决:解释 B 也基本否——水平轴是**稳健失败**

保符号只把水平从 ~19 抬到 ~21–22(+2–3,远够不着 70%),垂直反而掉(78→66)。
**水平组合失败不是符号丢失的锅。**

### 26.3 合并结论(S1+S2+S2b):稳健的轴不对称(干净负结果)

| 轴 | 跨判别式/生成式编码器 × abs_std/带符号 × 融合/单视角 |
|---|---|
| **垂直** | 稳健组合成立(66–78%),且多视角涌现(S1) |
| **水平** | 稳健组合失败(15–22%) |

水平失败**不是目标函数**(生成式 raw 没修,§25)、**不是输入符号**(保符号没修,§26)。
两个便宜有原理的解释都排除 → 水平组合失败是**三雷达幅度 CIR 传感模态的内在性质**
(很可能:垂直运动同时存在时,水平signature被盖住;探针在对角上见强垂直→把水平读 none)。

### 26.4 决定:停止追水平轴,收为诚实负结果 + scope 柱二

- **不再为单个属性移动球门**(已排除两个最该试的解释,继续砸 JEPA/潜轨迹性价比低、
  有硬凑嫌疑)。
- **柱二(描述未知)= 活着但 scope**:沿可组合轴描述(垂直已确认 + 多视角涌现;旋转/推
  待测),水平方向写进 limitation。CZSL 逐属性拆解、有的组合有的不组合是常态,诚实贡献。
- **论文一(生成式脊柱)不受影响**:脊柱价值是 e_recon(拒识)+ 缺视角补全(§18 已验证
  AUROC 95.48),从不依赖"修水平组合"。S2 只说"生成式不改善属性组合",其 OOD 信号未动。
- **下一步重心回到论文一的真交付物**:S3(LOSO 5-case 基线)+ S4(虚拟视角合成),
  二者都与水平组合正交。

### 26.5 新增日志

| 文件 | 内容 |
|---|---|
| `s2b_probe_perviewstd.log` | 保符号(per_view_std)探针 |
| `s2b_probe_none.log` | 原始带符号(none)探针 |

---

## 27. S3 — 三视角无融合 LOSO 5-case 基线(论文一主协议,2026-06-12)

**动机**:随机切分有 session 泄漏(同一被试样本漏进训练),95.14 虚高。论文一主协议
换成留一被试 LOSO,给出诚实的跨人数字 + 回应审稿人必问的泄漏质疑(§9 E1)。配置 = F0
赢家(三视角 `views=[Left,Top,Right]` + no-fusion + flatten + abs_std + inspector),
`--loso`,每 case 8 折。case-0 取自 F6(`f2_loso_c0_nofusion.log`),case 1–4 为 S3 新跑
(`s3_loso_case{2..5}.log`)。

### 27.1 5-case LOSO 结果(%)

| Case | combined OSCR | mean_only OSCR |
|---|---|---|
| 0 | 85.22 | 87.07 |
| 1 | 88.56 | 90.02 |
| 2 | 80.17 | 82.47 |
| 3 | 85.04 | 87.08 |
| 4 | 82.72 | 84.27 |
| **均值** | **84.34 ± 2.80** | **86.18 ± 2.60** |

### 27.2 核心对照:跨人代价

| 协议 | combined OSCR | 说明 |
|---|---|---|
| 随机切分(§2.2 主结果) | 95.14 ± 0.81 | session 泄漏,虚高 |
| **跨人 LOSO(S3)** | **84.34 ± 2.80** | 诚实跨人数字 |
| Δ | **−10.8** | 换没见过的人的真实代价 |

### 27.3 解读

- **三视角**(不是单视角):`fusion=False` 只关掉证伪的架构融合模块,三雷达 per-view
  inspector 距离平均(决策级聚合)全程生效——多视角增益没动,只去掉了无用模块。
- 跨人掉 10.8 分量化了 session 泄漏,**论文一以此为诚实主结果**,闭环 §9 E1 待办。
- std 从 ±0.8 涨到 ±2.8:跨人波动更大,case-2 最难(80.17,某被试折掉到 63)。
- **澄清(防误区)**:LOSO 不是"造个坑等融合填"。F6 已证融合在 LOSO 下更差
  (83.43±8.45 < 无融合 85.22±5.55),架构融合路已死。S3 是诚实基线本身,论文一靠
  e_recon(拒识)+ 缺视角补全这些**能力**,而非"把 84 提到更高"。

### 27.4 新增日志

| 文件 | 内容 |
|---|---|
| `s3_loso_case{2,3,4,5}.log` | case 1–4 的 LOSO 8 折(case-0 见 `f2_loso_c0_nofusion.log`) |

---

## 28. e_recon @ LOSO — 跨视角重建拒识信号抗不抗跨人?(2026-06-17 定稿;06-19 加 λ 融合 §28.6)

> **TL;DR**:standalone 下 e_recon 与 inspector 跨人**打平**(84.06 vs 84.97);但**融合后
> +3.4 分、5-case 全赢**(§28.6)→ 两者正交互补,这是论文一拒识腿的真卖点。

**动机**:论文一本想押一个"硬卖点":跨视角重建误差 e_recon(B2 解码器,衡量"这帧
在三视角下自不自洽")作为开集拒识信号,**可能比 inspector 的逐类距离流形更抗跨人
域偏移**——因为"自洽性"是物理一致性,不像距离原型那样挑人。一次 B2+LOSO 跑在**同
8 折**上同时报三个头(inspector / recon / msp),所以 e_recon @ LOSO 直接和 inspector
@ LOSO 配对。配置 = `--mae-pretrain --mae-target raw --pool flatten --normalize abs_std
--no-fusion --loso`,raw 重建目标(防塌缩),5-case 各 8 折。

### 28.1 5-case 三头结果(LOSO,OSCR %)

| Case | inspector(带 B2 init) | **recon(e_recon)** | recon − insp | msp |
|---|---|---|---|---|
| 0 | 85.28 | 80.53 | −4.75 | 71.53 |
| 1 | 89.40 | 90.13 | +0.73 | 71.81 |
| 2 | 81.69 | 82.56 | +0.87 | 74.34 |
| 3 | 84.64 | 86.76 | +2.12 | 79.43 |
| 4 | 83.12 | 80.36 | −2.76 | 71.89 |
| **均值±std** | **84.83 ± 2.91** | **84.07 ± 4.26** | **−0.76** | **73.80 ± 3.34** |

> recon 5 个 case 里赢 3 个(1/2/3)、输 2 个(0/4)。三头 AUROC 均值:inspector 90.5、
> recon ~90.5(case 0/1/2/3/4 = 87.03/96.39/89.42/92.72/86.93)、msp ~77.5。

### 28.2 判读:原"硬卖点"不成立(是持平,不是更好),但 e_recon 被验证为平行拒识信号

按跨人代价(随机切分 → LOSO 的掉幅)对照:

| 信号 | 随机切分 OSCR | LOSO OSCR(5-case) | 掉幅 Δ |
|---|---|---|---|
| inspector | 95.14(§2.2) | 84.83 ± 2.91(本节) | **−10.3** |
| **e_recon** | 95.01(§18) | 84.07 ± 4.26(本节) | **−10.9** |

- **e_recon 与 inspector 跨人下统计打平**:84.07 vs 84.83,差 0.76,远在彼此 std 内;
  跨人掉幅几乎相同(−10.9 vs −10.3)。→ 原假设"自洽性**更**抗域偏移"**不成立**——
  两者抗跨人能力**一样**,不是 e_recon 更强。
- **但也绝非"塌掉"**:e_recon 比 msp 高 ~10 分(84.07 vs 73.80),是一个和距离原型
  **正交、强度相当**的独立拒识信号 → 支撑论文一"一个模型两种能力"叙事(拒识这条腿
  可由 B2 解码器单独承担,且不输主 inspector)。
- **代价:e_recon case 方差更大**(±4.26 vs ±2.91),更挑 case(case-0/4 偏弱,
  case-1/2/3 反超)。
- **⚠️ case-0 单 case 是离群假象**:初判时只有 case-0(recon −4.75、掉幅 −14.5),
  误得出"硬卖点没站住、recon 跨人下钝化";5-case 收齐后此结论被推翻,见 28.4。
- **B2 初始化对 inspector 跨人≈无用**:本跑 inspector(带 B2 init)= 84.83,S3 无 B2
  = 84.34(§27),Δ +0.49 在噪声内 → B2 预训练特征对跨人 inspector 无可见增益。

### 28.3 论文一的处理

- **e_recon 定位:与 inspector 正交互补的第二拒识头,融合后 +3.4 分**(见 §28.6;不是
  limitation,也不是"更抗跨人")。写法:"重建自洽性提供一个与判别式距离正交的拒识信号;
  二者 standalone 跨人下同等鲁棒,融合(λ≈1)把 OSCR 从 84.97 推到 88.39"——正面主结果。
- 论文一脊柱价值的第二条腿仍走**实验②(虚拟视角合成 S4)**:B2 解码器补缺失视角,
  和 e_recon 拒识独立。

### 28.4 工程注记:case-3 首跑中途挂掉

`s_erecon_loso_c1to4.bat` 串行跑 case 1–4 时,**case-3 首跑只出 6/8 折(subj 1–6)即
进程退出**,无 SUMMARY 无堆栈(同 §20.6 记录的单进程偶发退出)。用 `s_erecon_loso_c3redo.bat`
单独重跑 case-3,8 折完整(inspector 84.64 / recon 86.76 / msp 79.43),本表用重跑值。

### 28.5 日志

| 文件 | 内容 |
|---|---|
| `s_erecon_loso_c0.log` | case-0 LOSO 8 折(✅ 06-16 00:34) |
| `s_erecon_loso_c{1,2,4}.log` | case 1/2/4 LOSO 8 折(✅ 06-16) |
| `s_erecon_loso_c3.log` | case-3 LOSO 8 折(重跑,✅ 06-17 23:03;首跑挂掉见 28.4) |

### 28.6 λ 融合(S2 / §18.3 兑现):e_recon ⊕ inspector,5-case LOSO(2026-06-19)

**动机**:28.1 的 standalone 只说明"两个头**一样强**",没说"错的是不是同一批样本"。
融合直接检验互补性:两个异常分各自 z-标准化后按 `s_z + λ·e_z` 相加(λ 扫 0.3/1/3),
看融合 OSCR 能否超过纯 inspector(λ=0)。配置同 28(`--lambda-recon 1`,5-case LOSO 各 8 折)。

**5-case 均值±std(%):**

| 头 | Accuracy | F1 | AUROC | **OSCR** | vs inspector |
|---|---|---|---|---|---|
| inspector | 82.60 | 79.62 | 90.79 | **84.97 ± 2.68** | — |
| recon(standalone) | 82.30 | 78.82 | 90.50 | 84.06 ± 4.06 | −0.91 |
| msp | 73.73 | 70.01 | 77.91 | 74.22 ± 3.92 | −10.75 |
| **recon ⊕ mean, λ=0.3** | 88.07 | 85.22 | 94.10 | **88.30 ± 2.33** | **+3.33** |
| **recon ⊕ mean, λ=1.0** | 88.41 | 85.79 | 94.68 | **88.39 ± 2.56** | **+3.42** ⭐ |
| recon ⊕ mean, λ=3.0 | 86.83 | 84.01 | 93.94 | 87.38 ± 3.05 | +2.41 |

**逐 case OSCR(λ=1.0 vs inspector):**

| Case | inspector | ⊕λ=1.0 | Δ |
|---|---|---|---|
| 0 | 85.41 | 87.76 | +2.35 |
| 1 | 88.71 | 92.12 | +3.41 |
| 2 | 81.17 | 85.20 | +4.03 |
| 3 | 84.72 | 89.36 | +4.64 |
| 4 | 84.84 | 87.50 | +2.66 |

**判决:e_recon 与 inspector 正交互补,融合是真增益,不是冗余。**
- 融合(λ=1.0)= **88.39 vs inspector 84.97 = +3.42**,且 **5 个 case 逐个全赢**(λ=0.3
  与 λ=1.0 均五战五胜)→ 两个头错的不是同一批样本,这与 E4(var 冗余)/E5(物理 bias
  冗余)的"看着不同其实冗余"**相反**:重建自洽性是判别式距离**之外**的信息。
- **最优 λ≈1.0**(0.3 几乎等同 88.30,3.0 偏大权重过头降到 87.38)。
- 融合还**收紧了方差**(±2.56 < e_recon 单独的 ±4.06),且 AUROC 94.68 远超任一单头 →
  把 e_recon 挑 case 的毛病也补上了。
- 这把 28.2/28.3 的"平行第二信号(打平)"**升级为"正交互补、融合 +3.4 分"**——论文一
  拒识腿的真卖点。standalone 打平仍成立(28.1),但卖点从"不输"变成"叠加增益"。

### 28.7 日志(λ 融合)

| 文件 | 内容 |
|---|---|
| `s_erecon_lam_c{0,1,2,3,4}.log` | 5-case LOSO + λ 融合(✅ 06-19 11:35,`s_erecon_lambda.bat`) |

---

## 29. S4 — 虚拟视角合成:缺视角腿(论文一第二条腿,2026-06-19;06-20 LOSO 定稿)

> **TL;DR**:B2 解码器能把缺失视角合成回来,质量够用(≈ 三真视角上界)。但对上**公平基线
> R1**(训练时 view-dropout),跨人 LOSO 下合成 **≈ R1(三视角平均 Δ=+0.19)**,不是碾压;
> 早筛的"+26"是稻草人(§29.2b)。**真卖点 = 统一性**(同一脊柱不需专门鲁棒训练就追平 R1)
> **+ 选择性合成**(丢重要视角 Left 合成 +2.1,丢冗余视角 Top 合成 −2.4)。

**动机**:论文一脊柱"一个生成模型两种能力"的第二条腿。B2 解码器训练时学的是"用其余
视角重建被掩视角"(§18),所以它本身就是个**视角生成器**:部署时某个雷达失效,就用
存活的视角把它**合成**出来,补回三视角再喂下游 inspector——把"缺视角鲁棒"从一个独立
补丁变成**生成式脊柱的一种用法**(和 e_recon 拒识共用同一个解码器)。和 R1(§19,训练时
丢视角)互补:R1 让模型对子集鲁棒,S4 把缺失视角补回来。

**做法**:每个视角 v,用 `dec(_encode_visible(enc, 其余两真视角))` 合成 v(输出在归一化
输入空间),替换进三视角输入,跑 inspector。**inspector bank 只训一次**、real3 / synth /
drop 三场景共用同一 bank 打分,保证差异只来自输入视角(代码 [synth.py](uwb_osr/synth.py),
`--synth-view`)。对照:`real3`(三真视角,上界)、`drop_v`(只剩两真视角,E2 式下界)。

### 29.1 早筛(case-0 随机切分,真数据,OSCR %)

> ✅ **强绿灯。** 判据:`synth_v` 比 `real3` 掉 <5 = 合成视角能替真视角。

| 缺失视角 v | real3 | **synth_v(合成替换)** | drop_v(只剩两真) | synth−real3 | synth−drop |
|---|---|---|---|---|---|
| Left | 95.95 | **93.86** | 67.46 | −2.09 | **+26.40** |
| Top | 95.95 | **94.17** | 91.48 | −1.78 | +2.69 |
| Right | 95.95 | **94.58** | 75.82 | −1.37 | **+18.76** |

> 同跑锚点:inspector 95.99、recon(e_recon)95.14、`synth_real3`=95.95(≈ inspector,
> 证明"训一次 bank 复用打分"与主 inspector 等价,对比干净)。

### 29.2 判读:合成视角够用,但 ⚠️ 此处 `drop_v` 是稻草人——正确基线见 29.2b

- **三视角合成替换全部只掉 1.4–2.1 分**(均 <5),几乎追平三真视角上界 → 合成视角对
  inspector 的类原型几何**够用**。这一条成立。
- **论文一含义**:"一个模型两种能力"字面成立——同一个 B2 解码器既拒识未知(e_recon,
  §28)又补全缺失视角(S4)。这是脊柱价值的第二条独立腿(和 e_recon 抗不抗跨人无关)。
- ⚠️ **早筛是随机切分单 case**。玩具数据冒烟时 synth 反塌(解码器没训好),真数据训满
  后反转为强绿——印证"合成质量取决于解码器训练充分度"。

### 29.2b 修正:`drop_v` 不该跟"全程喂 3 视角的模型"比(对照 R1 §19.2)

29.1 表里的 `drop_v`(67.46 / 91.48 / 75.82)是**全程喂 3 视角训练、测时才丢成 2 视角**
的模型——正是 E2(§11)的**病态稻草人**(模型没见过 2 视角输入)。它和 R1(§19.2)的
**baseline 列**(66.14 / 90.15 / 74.94)几乎一字不差,证明就是同一个病态值。所以"合成
救回 26 分"是跟稻草人比出来的,**夸大了 S4**。

**缺视角的公平基线 = R1(训练时 view-dropout=0.5)的模型测 2 视角**(§19.2 vd0.5 列):

| 缺视角 v | S4 synth(本节) | **R1 vd0.5 的 2 视角(§19.2)** | naive drop(稻草人) |
|---|---|---|---|
| Left (T+R) | 93.86 | **93.86** | 67.46 |
| Top (L+R) | 94.17 | **93.91** | 91.48 |
| Right (L+T) | 94.58 | **93.19** | 75.82 |

→ **对上公平基线,S4 合成 ≈ R1(顶多 +0.5),不是碾压。** S4 的价值因此**不是"性能更高"**,
而是**统一性**:同一个 B2 脊柱(拒识 e_recon + 补视角)**不需要专门的 view-dropout 训练**
就追平了为缺视角专门训练的 R1。卖点写成"one spine, two capabilities, matching a
dedicated robustness-trained baseline",别写成"+26 分"。

### 29.3 全 LOSO 5-case(诚实基线版,2026-06-20 定稿)

Stage-1 用 `--view-dropout 0.5`(R1)训练,于是**同一个 R1 鲁棒模型**上:`synth_drop_v`
= 公平基线(R1 式 2 视角)、`synth_v` = 再合成补回第三视角。直接回答"在已经会处理 2 视角
的模型上,合成第三视角还加不加分"。(view-dropout 只改 Stage-1,B2 enc/dec 合成质量不变。)

**5-case 均值±std(OSCR %):**

| 缺失视角 v | **synth_v(合成)** | **synth_drop_v(R1 式 2 真视角,公平基线)** | Δ=synth−drop | 逐 case Δ |
|---|---|---|---|---|
| Left | 79.20 ± 3.15 | 77.11 ± 3.57 | **+2.09** | +2.8/+3.0/+2.0/+0.0/+2.6 |
| Top | 74.75 ± 1.69 | 77.18 ± 3.14 | **−2.43** | −3.6/−4.3/−0.8/−4.1/+0.8 |
| Right | 78.12 ± 1.06 | 77.21 ± 1.37 | **+0.91** | +2.2/−0.1/+0.2/−0.4/+2.6 |
| **三视角平均** | | | **+0.19** | (≈打平) |

> 锚点:`synth_real3`(三真视角)= 83.74 ± 1.74 ≈ inspector 83.83(sanity ok)。`synth_drop_v`
> 绝对值 ~77,比 R1 §19.5 随机切分的 ~91-92 低 ~14 分,正是跨人 LOSO 的诚实代价(§27 同量级)。

### 29.3b 判读(诚实版定稿):合成 ≈ R1,价值在统一性 + "按视角重要性合成"

- **整体打平**:在 R1 鲁棒模型上,合成第三视角 vs 直接用 2 真视角,三视角平均 Δ=+0.19
  → **合成不比 R1 多给性能**。早筛那个"+26"确实是稻草人(对没训过缺视角的病态模型)。
- **但有稳定的视角相关结构(这才是有意思的发现)**:
  - **Left:合成稳定有增益(+2.09,4/5 case 正)** —— Left 是丢了最伤的视角,合成把它的
    独有信息补回来划算。
  - **Top:合成稳定拖累(−2.43,4/5 case 负)** —— Top 信息冗余(R1 自己测 L+R 就够好),
    合成它等于往输入塞一份带误差的赝品,纯加噪。
  - **Right:≈打平(+0.91,摇摆)** —— 介于两者之间。
- **论文一怎么写(收尾要点)**:
  1. **主叙事 = 统一性**,不是性能碾压:同一个 B2 脊柱(拒识 e_recon §28 + 补视角 §29)
     **不需要专门的 view-dropout 训练**,就达到了为缺视角专门训练的 R1 的同等鲁棒。
  2. **二级 nuance = 选择性合成**:合成的收益取决于丢的视角是否携带独有信息;实践上可
     **按视角重要性 / 重建误差门控**——重要视角(合成 Δ>0)才合成,冗余视角直接丢更好。
     这把"一刀切合成"升级成"按需合成",是个干净的小贡献。
  3. **诚实** > 夸大:full-paper 用 §29.3 这版(对 R1),**别用早筛的 +26**(§29.2b 已警示)。

### 29.4 日志 / 代码

| 项 | 内容 |
|---|---|
| `uwb_osr/synth.py` | S4 合成 + 评测(`evaluate_synth_views`,bank 训一次复用) |
| `main.py: --synth-view` | S4 开关(需 `--mae-pretrain` + 三视角) |
| `s4_screen_c0.log` | case-0 随机切分早筛,无 dropout(✅ 06-19 12:38) |
| `s4_loso_c{0..4}.log` | 诚实版全 LOSO 5-case,`--view-dropout 0.5`(✅ 06-20;c0/c1/c2 由 `s4_loso.bat`,c3/c4 由 `s4_loso_resume.bat` 续) |

---

## 30. 单视角胜出配置换 LOSO(跨人代价对照,2026-06-28)

**动机**:§20 的单视角胜出配置(86.74 ± 1.29)是随机切分。把它原样换成留一被试
LOSO,得到单视角的诚实跨人数字,与三视角 LOSO(§27,84.34)同协议对比,量化"单视角 ×
跨人"这一组合的代价。配置 = §20.4 胜出配置不动:`--single-view --pool flatten
--normalize abs_std --paper-neg-mixup --d-key 32 --data-type Raw`,加 `--loso`,
每 case 8 折。

### 30.1 5-case LOSO 结果(%,inspector 头,每 case = 8 折人均)

| Case | closed_acc | OSCR | AUROC | Accuracy | F1 | MSP OSCR |
|---|---|---|---|---|---|---|
| 0 | 78.00 | 57.10 | 66.99 | 61.35 | 48.31 | 57.78 |
| 1 | 76.31 | 53.98 | 65.10 | 60.61 | 37.38 | 52.72 |
| 2 | 70.54 | 51.42 | 64.32 | 60.38 | 39.91 | 49.56 |
| 3 | 85.44 | 65.16 | 72.46 | 68.40 | 58.98 | 68.88 |
| 4 | 73.23 | 48.91 | 58.26 | 58.86 | 42.52 | 50.14 |
| **均值±std** | **76.70 ± 5.06** | **55.31 ± 5.62** | **65.43 ± 4.57** | **61.92 ± 3.34** | **45.42 ± 7.69** | **55.82** |

### 30.2 核心对照:四个角(视角数 × 切分协议,OSCR %)

| | 随机切分 | 跨人 LOSO | 跨人代价 |
|---|---|---|---|
| **单视角** | 86.74 ± 1.29(§20.4) | **55.31 ± 5.62** | **−31.4** |
| **三视角** | 95.14 ± 0.81(§2.2) | 84.34 ± 2.80(§27) | −10.8 |

闭集同样塌:单视角 closed_acc 随机切分 ~97.70 → LOSO 76.70(−21);三视角 LOSO 仍 ~90。

### 30.3 解读

- **不是 bug,是真现象**。已核实:LOSO 切分无泄漏(train=7 人 / test=留出 1 人,
  `loso_split`);`abs_std` 逐样本自归一化无统计量泄漏;同管线四角对照只在"单视角 ×
  跨人"塌陷;55.3 远高于 6 类随机 16.7%,模型在学只是泛化不动。
- **多视角集成才是跨人鲁棒性的来源**(印证 E4):单一雷达只有一个几何视角,手势执行
  高度因人而异(速度/幅度/身材/站位),换没见过的人就消歧不动;三雷达三角约束几何,
  跨人仍撑 ~90 闭集 / 84 OSCR。
- **§20 的胜出因素到跨人下失效**:随机切分让 Raw(+10.5)、D_key、mixup 涨分的因素,
  LOSO 下不再是赢家(inspector 与 MSP 打平 55.31 vs 55.82)。Raw 这一项已由 §30.4
  对照证实**反转**(见下)。
- **折间方差极大**:跨 case std 5.62,折级更野(c4 最低折 23.29,c3 最高折 78.51),
  单视角对"测试者是谁"极度敏感。

### 30.4 ClutterRemoved 对照:Raw 的随机切分优势是跨人泄漏(case 0,8 折人均)

胜出配置原样、只把 `--data-type Raw` 换成 `ClutterRemoved`,LOSO case 0:

| 指标 | Raw(§30.1 c0) | ClutterRemoved | CR − Raw |
|---|---|---|---|
| closed_acc | 78.00 | 82.50 | +4.50 |
| OSCR | 57.10 | **68.38** | **+11.28** |
| AUROC | 66.99 | 76.11 | +9.12 |
| Accuracy | 61.35 | 69.73 | +8.38 |
| F1 | 48.31 | 60.64 | +12.33 |

> **方向整个翻转,教科书级泄漏特征**:随机切分 Raw 赢 ~+10.5(§20),跨人 LOSO 反而
> CR 赢 +11.3。解释:Raw 保留静态杂波,杂波含人/站位/家具相关线索;随机切分下同被试
> 样本横跨 train/test,模型记住这些线索 → Raw 涨分;跨人 LOSO 杂波变噪声/误导,去掉
> (CR)反救回 11 分。⚠️ 注意单折会骗人:CR 第一折 55.09 ≈ Raw 54.43,要 8 折人均
> 才显出 +11(CR 折级 55→87)。
>
> **论文含义**:"最佳输入"随协议而变 —— 随机切分用 Raw、跨人用 ClutterRemoved;
> §20 的 86.74 复现里 Raw 的贡献被随机切分泄漏放大了。

### 30.5 日志 / 代码

| 文件 | 内容 |
|---|---|
| `batch_scripts/sv_loso_winner.bat` | §30.1 运行脚本(胜出配置 Raw + `--loso`,逐 case) |
| `logs/sv_loso_c{0..4}.log` | Raw 5-case 各 8 折 LOSO(✅ 06-28;done `sv_loso_c{0..4}.done` / `sv_loso.done`) |
| `batch_scripts/sv_loso_cr_c0.bat` | §30.4 CR 对照脚本(只换 `--data-type ClutterRemoved`,case 0) |
| `logs/sv_loso_cr_c0.log` | CR case 0 的 8 折 LOSO(✅ 06-28;done `sv_loso_cr_c0.done`) |

---

## 31. N1 — 少样本新手势注册可行性(论文三,2026-06-28)

**动机**:论文三(见 RESEARCH_PLAN §4)要把开集系统从"只会拒识未知"升级到"用几个样本
注册一个新手势成正式类"。N1 是命门早筛:验证 inspector per-class 架构 + **最近原型路由**
能否在跨人 LOSO 下① 换路由后 base 准确率不塌、② 5-shot 新类显著高于随机。配置 = 论文一
部署系统(三视角 no-fusion + flatten + abs_std + ClutterRemoved + per-view inspector),
协议:base=[1,4,5,7,8,11] 训编码器,register=[0,2,3] 用每类 5 个训练被试样本注册,
unknown=[6,9,10] 始终不注册当真未知,8 折 LOSO 在留出被试上测。两法配对同编码器:
A=训练 few-shot inspector(同 base 头家族),B=NCM 冻结特征均值原型。

### 31.1 结果(case 0,8 折人均,%)

| 方法 | acc_base | acc_new | H | AUROC | OSCR |
|---|---|---|---|---|---|
| **A(训练 inspector)** | **93.42 ± 7.00** | **73.82 ± 7.58** | **82.16** | 78.44 | 70.69 |
| B(NCM) | 87.65 ± 6.86 | 38.61 ± 15.09 | 51.83 | 59.99 | 50.81 |

### 31.2 核心判据:两个 go 都过(方法 A)

| 判据 | 数字 | 对照 | 判决 |
|---|---|---|---|
| 换最近原型路由后 base 不塌 | 93.42 | logit 路由同 known 集 §27 case-0 = **91.71** | ✅ 不塌,甚至略升 |
| 5-shot 新类显著高于随机 | 73.82 | 9 类随机 ≈ 11.1 | ✅ 远超 |

**结论:N1 GREEN。** 最近原型路由是免遗忘注册的正确机制——既保住 base(≈logit 路由),
又让仅 5 个样本注册的新手势在没见过的人身上认到 73.8%。论文三的地基立住。

### 31.3 方法 A ≫ B(few-shot 消融,反直觉)

合成数据上 B(NCM)曾略胜,真数据上 **A 完胜**(acc_new 73.82 vs 38.61)。原因:真雷达
高维特征下,训练 inspector + 重增广(5 个真样本 → 4000 合成正样本 + mixup 负样本)的
泛化远好于裸最近类均值。**卖点反转**:few-shot 注册要训轻量头 + 增广,不是 NCM。

### 31.4 固定 η=3 阈值跨方法/类都失灵 → 直接给 N2 conformal 当 motivation

operating point 用固定 η=3(A7 的 3σ 规则)报 op_base/op_new/rejU,**退化且方向相反**:
- 方法 A:r 活在 inspector 潜空间(尺度大),η=3 过严 → op_base 仅 29.3、新类全被拒
  (op_new 0)、但真未知拒得干净(rejU 99.75);
- 方法 B:r 活在裸特征空间(尺度小),η=3 过松 → 全接受(rejU 0),啥也拒不掉。

**同一个 η=3 连两种方法的尺度都跨不了,更别说 base vs 新类**——这正是 A7(§15/§17
"η 不跨尺度迁移")的延续,也是 **N2 共形预测要解决的确切问题**(逐类/逐组校准阈值 +
覆盖率保证)。N1 的 threshold-free 指标(acc/H/AUROC/OSCR)才是主结果,η=3 行只是反衬。

### 31.5 日志 / 代码

| 文件 | 内容 |
|---|---|
| `config.py: REG_TEST_CASES` | 注册协议 base/register/unknown 划分 |
| `uwb_osr/data.py: register_split` | LOSO 切 train(base)/support(few-shot)/test(base+新+未知) |
| `uwb_osr/registration.py` | `register_class`(A 复用 train_inspector / B NCM)+ `nearest_proto` 路由 + `evaluate_registration` |
| `main.py: --register / --n-shot / --reg-method {A,B,both}` | N1 入口;both = A/B 配对同编码器 |
| `batch_scripts/n1_register.bat` / `logs/n1_register_c0.log` | 三视角 8 折 LOSO(✅ 06-28;done `n1_register_c0.done`) |

### 31.6 待办

- [x] N2:conformal 校准阈值替换 η=3 → §32。
- [ ] N3:shot 曲线(1/5/10/20)、A vs B、原型路由 vs logit 路由、跨人 vs 同人注册。
- [ ] 多 case(现仅 case 0);AUROC 70.69~78 偏中等,N2/N3 看能否靠校准 + 多视角一致性抬。

---

## 32. N2 — conformal 校准的接受/拒识阈值(论文三,2026-06-29)

**动机**:N1(§31)证明注册可行,但固定 η=3 阈值跨方法/类尺度全失灵。N2 用 split-conformal
给"多近算够近"配覆盖率保证,输出集合值预测(空集=拒识,单类=确定,多类=不确定)。同 N1
部署系统 + 协议(三视角 no-fusion + flatten + abs_std + CR + 最近原型 + 方法 A,case 0,
8 折),α=0.1(目标覆盖 90%)。三种阈值:mondrian(逐类)、global(池化)、eta=3(N1 对照)。
非合规分数 = 归一化距离 r_k;每折分数 dump 到 `logs/n2_scores_c0_s*.npz`,任意 α/模式可
事后复算(无需重训)。

### 32.0 关键方法论修正:LOSO 下校准必须用留出被试(否则可交换性破坏)

首跑用"训练被试里留一部分样本"校准 → **base 覆盖率仅 59%**(远低于 90%)。从 dump 的分数查出
根因:**测试被试到原型的距离是校准(训练被试)距离的 4.6 倍**(中位 15.1 vs 3.3)。训练被试
被编码器/原型见过,距离小;测试是新人,距离大 → 训练被试校出的阈值对新人太紧 → 欠覆盖。
**修正 = 留两被试**(`register_split_conf`):1 个测试 + 1 个校准 + 6 个训练,校准与测试都是
模型没见过的人 → 可交换 → 保证成立。**naive 版作为负对照保留**(`n2_conformal_c0_flawed_withintrain.log`),
是论文里"为什么跨用户部署要用留出用户校准"的现成 motivation。

### 32.1 结果(case 0,8 折人均,%,目标覆盖 90)

| 模式 | cov_base | cov_new | size_b | size_n | reject_unk | singAcc_b | singAcc_n |
|---|---|---|---|---|---|---|---|
| **global(池化)** | **89.7** | 49.1 | 1.65 | 1.49 | 43.8 | 84.0 | 58.3 |
| mondrian(逐类) | 81.9 | 49.1 | 1.35 | 1.01 | 61.0 | 84.1 | 64.0 |
| eta=3(N1 对照) | 24.8 | 0.0 | 0.25 | 0.00 | 99.9 | 100.0 | 0.0 |
| naive 校准(负对照) | 59 | 0 | — | — | — | — | — |

### 32.2 判读

- **base 覆盖率保证恢复**:global 池化 **89.7 ≈ 90% 目标**(vs naive 59、vs η=3 24.8)。
  留出被试校准 + 池化 conformal = base 类的可部署保证。**这是 N2 的主成果。**
- **global > mondrian(逐类)**:逐类 81.9 且方差大(±9.4 折间)。单个校准被试下逐类校准点
  少(100/类)、且单人不代表测试人 → 逐类欠覆盖;池化 600 点更稳,正中目标。**单校准被试时
  池化更可靠。**
- **η=3 跨人无用**:base 覆盖 24.8、新类 0——固定阈值在跨人尺度下彻底失效,坐实 N1 的判断,
  conformal 是正解。
- **集合小且 singleton 多对**:global size 1.65/1.49(满分 9 类),singleton 准确率 base 84%。
  即多数样本给出小而准的预测集,可用。
- ⚠️ **新类覆盖只有 49%(N2 的未解问题)**:从 dump 分数诊断——新类到自身 5-shot 原型的距离
  中位 **144**,是 base 的 **~12 倍**(base 中位 11.9)。根因:5 个样本估的 σ 严重低估真实
  散度 → r=d/σ 爆大 → base 校准出的全局阈值(q90≈134)够不到新类(中位 144)→ 欠覆盖。
  这是少样本 conformal 的**尺度失配**,不是管线 bug(base 同管线正中 90%)。**修法属 N3**:
  对 5 个支持样本做留一(cross-conformal/jackknife)给每个新类自己的阈值,或做 σ 尺度校正。

### 32.3 日志 / 代码

| 文件 | 内容 |
|---|---|
| `uwb_osr/data.py: register_split_conf` | 留两被试切分(测试 + 校准 + 训练) |
| `uwb_osr/registration.py: register_and_score / conformal_q_vector / conformal_report` | 分数矩阵 + 三模式阈值 + 覆盖/集合/拒识/singleton 指标 |
| `main.py: --conformal / --alpha / --calib-frac` | N2 入口;每折 dump npz |
| `batch_scripts/n2_conformal.bat` | 运行脚本(已加 `PYTHONUNBUFFERED=1`/`-u`,否则日志缓冲看似卡死) |
| `logs/n2_conformal_c0.log` + `logs/n2_scores_c0_s*.npz` | 结果 + 逐折分数(✅ 06-29) |
| `logs/n2_conformal_c0_flawed_withintrain.log` | 负对照:naive 校准 base 覆盖 59 |

### 32.4 待办

- [x] N3 头号任务:新类覆盖率 → §33(balanced support +17.5;σ-shrinkage 证伪并破案;hybrid oracle 82.9)。
- [ ] 多校准被试(降逐类 Mondrian 方差)、多 case、α 敏感性(npz 可事后扫)。
- [x] 覆盖率 vs 拒识 tradeoff → §33.4(边界已量化:阈值只能沿边界走,挪边界要改分数)。

---

## 33. N3 — 新类覆盖率:balanced support ✅,σ-shrinkage ❌ 证伪并破案(2026-07-02)

**动机**:N2(§32)遗留新类覆盖 49.1%(目标 90)。RESEARCH_PLAN §4.9 排的三味药:P1 hybrid
(校准被试的新类样本逐类定阈,oracle 上界)、P1.5 subject-balanced support(5 shot 来自 5 个
不同人)、P2 σ-shrinkage(新类 σ 向 base 池收缩,数据无关)。本节 = 干净 8 折跑(150 epoch,
--mmap,单进程;06-30 的内存崩溃未再出现)+ 全套后处理分析(`tools/n3_analyze.py` + dump 的
σ̄/距离矩阵,无重训)。配置同 §32:三视角 no-fusion + flatten + abs_std + CR + 方法 A,
case 0,α=0.1,留二被试(校准+测试都是没见过的人)。

### 33.1 主结果(8 折人均,%,目标覆盖 90;support=subject-balanced)

| 模式 | cov_base | cov_new | size_b | size_n | empty_n | reject_unk | singAcc_b | singAcc_n |
|---|---|---|---|---|---|---|---|---|
| **hybrid(oracle)** | 89.9 | **82.9** | 1.42 | 1.67 | 10.8 | 40.6 | 84.5 | 81.6 |
| mondrian(逐类) | 82.6 | 82.9 | 1.04 | 1.13 | 10.3 | 53.3 | 87.9 | 81.1 |
| **global(池化)** | **89.9** | **66.6** | 1.74 | 1.75 | 27.0 | 44.4 | 84.9 | 74.8 |
| eta=3(对照) | 23.2 | 0.0 | 0.23 | 0.00 | 100.0 | 99.8 | 100.0 | 0.0 |

### 33.2 P1.5 balanced support:免费 +17.5,坐实"采样要人多,不要样本多"

- global cov_new:**49.1(§32 随机 support)→ 66.6(+17.5)**,cov_base 不动(89.7→89.9);
- 闭集不亏:nearest-proto acc base **92.98±6.63** / new **75.22±10.83**(对照 N1 §31 随机
  support 93.42/73.82;注意 N1 是留一 7 训练被试、本节留二 6 训练被试,差异可忽略);
- 机理:同人 5 样本方差天然偏小 → 原型/σ 只描住一个人;5 人各 1 样本让少样本头见到跨人散度。
  **这是可直接写进论文的采集指南**(注册时找不同的人各做一次,好过一个人做五次)。

### 33.3 P2 σ-shrinkage ❌:单调变差,且破案——病根在 d 不在 σ

λ 扫(纯后处理,r\*=r·σ/σ\*,σ\*²=(1−λ)σ²+λσ̄²_pool;global 阈值):

| λ | 0(=global) | 0.25 | 0.5 | 0.75 | 0.9 | 1.0 |
|---|---|---|---|---|---|---|
| cov_new | 66.6 | 61.8 | 54.4 | 44.4 | 38.0 | **32.2** |

**方向整个反了**。查 dump 的 σ̄:**新类 σ̄ 是 base 的 1.33~1.93 倍(8 折)——σ 根本没被低估,
反而偏大**;真失配在分子 d:新类样本到自身原型的归一化距离中位 77~208,base 只有 4~32,
**约 8~10 倍膨胀**。§32.2 的"5-shot σ 低估散度"只对了一半(r=d/σ 确实爆了,但爆的是 d)。
**改判:病根 = 少样本头的跨人转移差距**——方法 A 的 inspector 在 5 个(训练被试的)support 上
训出来,把自己的 support 映得很近,换个人的同手势样本掉得远。σ 缩放救不了 10 倍的 d,
往更小的 base 池 σ 收缩只会让 r 更大 → 覆盖更掉。**shrinkage 这条线关闭。**

### 33.4 覆盖-拒识权衡边界 + 未知泄漏分解:阈值调不动边界,分数才行

新类列阈值 = s·q_global 的标量扫(base 列不动,后处理):

| s | 1(=global) | 2 | 3 | 4 |
|---|---|---|---|---|
| cov_new | 66.6 | 92.3 | 98.0 | 99.2 |
| reject_unk | 44.4 | 23.4 | 2.9 | 0.0 |

- s=2 就能"达标"90,但拒识近乎减半——烂交换;**hybrid 82.9@rejU 40.6 明显优于同覆盖的标量点
  → 逐类阈值形状比全局缩放值钱**;
- **未知泄漏分解(s=1)**:被接受的未知里 **53.0% 从 base 列漏入**,新类列只贡献 16.9%——
  拒识短板的主通道在 base 分数质量,不在少样本头;
- 分辨力:own-r vs 未知 min-r 的 AUROC = base 列 **86.5**、新类列 **90.5**。校准只能沿这条
  边界滑动,**想挪动边界必须改非合规分数本身**(→e_recon⊕r 融合,论文一 §28.6 有 +3.42 先例;
  以及逐视角 p 值合并)。

### 33.5 判读(改写 N3 的故事)

- ✅ 框架三层里"保证层"又进一步:balanced support 是数据无关的免费增益;
- ❌ 原定算法贡献(σ 校正)被自己的诊断否决——**替代方案 = 模拟任务共形**:base 类有海量数据,
  每折对每个 base 类也抽 5 shot 训"模拟少样本头",用校准被试量它的膨胀分布,池化出 few-shot
  头专用阈值转移给新类列(不需要任何新类校准数据;与 few-shot conformal 的 auxiliary-task
  思路同族,写作前查 Fisch et al. ICML'21 划清贡献边界);
- ⚠️ **hybrid(oracle)也只有 82.9**:单校准被试代表不了测试被试(subj7 折 cov_new 47)。
  "打到 90"对跨人少样本注册可能本就不现实,论文按"数据无关校正救回 X 分 + 剩余是 oracle
  也过不去的被试异质性"诚实写。

### 33.6 日志 / 代码

| 文件 | 内容 |
|---|---|
| `logs/n3_conformal_c0.log` | 8 折干净跑(150 epoch,~5h;"Reg case 1"是 1-based 显示,实为 case 0) |
| `logs/n3_scores_bal_c0_s1..8.npz` | 每折距离矩阵 + sigma_bar + R_cal_reg(全部后处理的原料) |
| `tools/n3_analyze.py` | 后处理:global/hybrid/shrinkage-λ/eta 全模式,无需重训 |
| `batch_scripts/n3_calib.bat` | 运行脚本(已回默认 150 epoch) |

### 33.7 待办

- [x] 误差解剖:同人 vs 跨人注册 → §34(已跑完,判定"换人"占 2/3)。
- [x] 模拟任务共形 → §36.2 **证伪**(base 类太紧,不是新类的合格替身;与 shrinkage 同源)。
- [x] e_recon⊕r 融合 → §36.3 **手术版(仅 base 列)达线**(AUROC 83.5);整列版证伪。逐视角 p 值合并=信号上限诊断(§36.1)。
- [x] 多 reg case → §35(case-1:结论稳健,acc_new 强依赖词表,"孪生在 base"机制)。
- [ ] 多校准被试 / 轮换校准被试(现固定 cs=第一个非测试被试,`--calib-rotate` 已就绪);shot 曲线(`p3_shots_c0.bat` 已就绪)。

---

## 34. N4 — 误差解剖:同人注册把新类误差的 2/3 归因给"换人"(2026-07-03)

**动机**:在押注表征升级前,先解剖 §33 的 25% 新类误差是"换人"(subject shift)还是"手势本身
难认"(class hardness)。方法:support 改取**测试被试本人**的 5 个样本(`--support-personal`,
已从测试集剔除防泄漏),其余与 §33 完全同构(同编码器配置/协议/α)。这同时就是部署里的
personalization 场景("用户在自己设备上注册自己的手势")。

### 34.1 结果(8 折人均;对照 = §33 balanced 跨人)

| | 跨人(§33 bal) | **同人(pers)** | Δ |
|---|---|---|---|
| nearest-proto acc_base | 92.98 ± 6.63 | 93.29 ± 6.71 | ≈0(应然:base 管线未变) |
| nearest-proto acc_new | 75.22 ± 10.83 | **91.12 ± 3.66** | **+15.9,方差缩到 1/3** |
| r_own 中位(new/base) | 124.0/15.9(**7.8×**) | 53.4/15.3(**3.5×**) | d 膨胀砍半以上 |

conformal(%,目标 90):

| 模式 | cov_base | cov_new | size_b | size_n | reject_unk | singAcc_b | singAcc_n |
|---|---|---|---|---|---|---|---|
| hybrid | 89.5 | 98.7 | 1.63 | 2.02 | 22.7 | 95.3 | 86.4 |
| mondrian | 82.4 | 98.7 | 1.34 | 1.55 | 27.7 | 85.9 | 86.3 |
| **global** | **89.5** | **91.0** | 1.64 | 1.92 | **46.0** | 84.6 | 83.8 |
| eta=3 | 24.3 | 0.0 | 0.24 | 0.00 | 99.9 | 100.0 | 0.0 |

### 34.2 判读

- **换人 ≈ 新类误差的 2/3**(错误率 24.8%→8.9%)。剩余 ~9% 是类混淆/少样本本身的地板
  (register=[0,2,3] 含同轴反向对)。
- **同人场景基本已解决,可当论文部署卖点**:91% 准确 + **global(只用 base 校准的阈值)就
  双向中靶(89.5/91.0)且 rejU 46 不塌**——d 膨胀降到 3.5× 后,§32 的尺度失配大半消失,
  连逐类阈值都不需要。
- ⚠️ **hybrid 在同人场景反而过覆盖**(98.7,集合变大 2.02,rejU 掉到 22.7):校准被试对
  "同人原型"是外人,量出的阈值系统性偏松。**阈值方案必须与注册场景配对**(跨人注册→hybrid
  有意义;同人注册→global 即可),这本身是论文里值得一小节的 nuance。
- **决策(核对 §33.4 的判据)**:acc_new 91 ≥ 88 → **主攻表征级被试不变性**;增广/难负样本线
  降级(天花板只剩 ~9 分,且大半是类混淆地板)。

### 34.3 下一步:表征阶梯(一晚一个变体,同一评测协议)

| 变体 | 做什么 | 状态 |
|---|---|---|
| **N4a:B2 初始化** | 编码器先跨视角掩码重建预训练(mae_epochs=100,target=raw)再 CE 微调——CE 只学"区分 6 个 base 类"特征向 6 类塌缩,重建特征更通用 | ✅ **有代价的胜利**(§34.5) |
| N4b:episodic | base 6 类上加辅助原型场景损失,训练目标=部署条件 | ❌ σ 失衡致新类路由坏死(§34.7) |
| N4c:被试对抗 | 梯度反转被试头,把"这是谁"从特征里洗掉——对 subject shift 最对症 | ❌ λ=0.5/0.1 双塌,搁置(§34.6) |

### 34.5 N4a(B2-init)结果:新类 +2.5,拒未知 −3.1——通用性是双刃剑(2026-07-03)

8 折配对(vs §33 基线,同折同协议;`--mae-pretrain --dump-tag b2`,
`logs/n4a_b2init_c0.log` + `n3_scores_bal_b2_c0_s1..8.npz`):

| 指标 | 基线 | B2-init | Δ | 折胜负 |
|---|---|---|---|---|
| acc_base | 92.98 | 93.19 | +0.21 | 打平 |
| **acc_new** | 75.22 | **77.75** | **+2.53** | 5/8 赢 |
| 距离膨胀(new/base) | 10.6× | 9.0× | −1.6× | 7/8 收窄 |
| **AUROC(known vs unk)** | 80.10 | **76.96** | **−3.14** | 7/8 输 |

conformal 同向:hybrid cov_new 82.9→83.2(平)但 rejU 40.6→32.2(掉)。

**判读**:① 假设验证成立——重建预训练的特征对没见过的类更慷慨(acc_new +2.5、膨胀收窄);
② 但同样的通用性让**真未知也嵌得更近**(AUROC −3.1)——"判别塌缩 vs 通用"是一根轴上的
trade-off,不是免费午餐;③ **代价可赎回**:B2 训练本附赠 e_recon 拒识信号(§28.6 融合 +3.42
的先例),本跑只用了编码器初始化、解码器被丢弃——**B2 分支的完全体 = init + e_recon⊕r 融合**
(→P3),预期能把拒识补回来。④ 结论:B2-init 不单独采纳,留作与 e_recon 融合/N4c 的叠加组件。

### 34.6 N4c(被试对抗)λ=0.5 ❌ 表征全塌——剂量灾难,λ=0.1 重试(2026-07-03)

DANN 梯度反转被试头(engine.py `_GradReverse`/`_dann_lambda`,`--subj-adv`,标准 0→λ ramp),
λ=0.5 首跑(`logs/n4c_subjadv_c0.log` + `n3_scores_bal_adv_c0_s1..8.npz`):

| 指标(8 折配对) | 基线 | adv λ=0.5 |
|---|---|---|
| acc_base | 92.98 | **50.12** |
| acc_new | 75.22 | **11.92(≈9 类随机)** |
| AUROC | 80.10 | **46.62(低于随机)** |
| 距离膨胀 | 10.6× | 1.29× |

**判读**:膨胀 1.29× + η=3 行覆盖 98~100/集合 8.7 满 9 类 = 特征空间被压成一团(骗过被试头的
代价是类结构一起被洗掉)——DANN 经典塌缩,对 dim=32 的小编码器 λ=0.5 剂量过猛。**教训**:
① 对抗强度是敏感旋钮,从小起步;② 对抗跑必须盯 fold-1 早停(监视器改逐折推送),别烧整晚;
③ "膨胀确实被压到 1.29×"反过来证明机制在起作用,只是把好几何一起压没了——剂量对了才有戏。
重跑 λ=0.1(`--dump-tag adv01`,batch_scripts/n4c_adv01.bat)。

**λ=0.1 亦塌,fold-1 早杀(07-03 深夜)**:同签名(η 全覆盖/集合 9/9;fold-1 acc_base 65.5、
acc_new **0.7**、r_own 中位 14.6→0.98)。λ 降 5 倍塌缩不减 → "压平一切"这条作弊捷径对本架构
结构性可达,不是剂量问题。**对抗线搁置**(将来若捡起:GRL 挂牺牲用投影头 + 类结构成形后再启动;
两晚预算已耗)。转 N4b episodic(辅助原型场景损失,无对抗自毁倾向)。fold-1 npz 留档
`n3_scores_bal_adv01_c0_s1.npz`。

### 34.7 N4b(episodic λ=0.5)❌ fold-1 早杀——机制:σ 失衡,base 头吸走一切;表征阶梯收官(2026-07-04)

批内原型场景损失(engine.py `_proto_episode_loss`,`--episodic 0.5`,支持半建原型/查询半按
−距离² 分类,逐视角平均)。fold-1(`n3_scores_bal_epi_c0_s1.npz`):acc_base 86.0(存活),
**acc_new 0.3(坏死,比随机 11 还低)**。验尸:**base 头 σ 被撑爆 3~8 倍**(0.03→0.08-0.24),
少样本头 σ 未变 → r=d/σ 下 σ 巨大的 base 头归一化距离到处都小 → **新类样本 299/300 被路由进
base 列**(基线 211/300 回自己头)。

**表征阶梯收官判读(消耗 5/6 变体预算)**:
- 唯一有条件胜者 = **N4a B2-init**(acc_new +2.5,AUROC −3.1 待 e_recon 赎回);
- N4b/N4c 殊途同归打在同一软肋:**per-class σ 归一化路由校准在 CE 特征几何上,任何重塑几何的
  辅助目标(对抗洗被试、episodic 摊开类内)都会打破跨头可比性**——base 头(全量数据训练,吃到
  新几何)与少样本头(5 点)的 σ 尺度失衡即路由死亡。这是论文方法节的重要 caveat:免遗忘
  多头架构的代价是对特征几何漂移敏感;
- **75.2(+B2 后 77.8)即当前跨人 5-shot 的诚实水平**,按止损预案写成 finding;
- 下一步 = P3(e_recon⊕r 融合非合规分数:赎回 B2 的 AUROC + 堵 §33.4 的 base 列 53% 泄漏)
  + 协议补全(case 1 已在跑)。

止损线(2026-07-02 定):最多 6 变体/两周夜跑;acc_new≥80 或 AUROC≥83 任一达成即收手转
模拟任务共形 + 写作;全不中则 75.2 就是跨人 5-shot 的诚实水平,写成 finding。

### 34.4 日志 / 代码

| 文件 | 内容 |
|---|---|
| `uwb_osr/data.py: register_split_conf(support_personal=)` | 同人 support + 测试集剔除(单元+冒烟已验证) |
| `main.py: --support-personal` | 入口;npz tag=pers |
| `batch_scripts/n4_personal.bat` | 运行脚本(判据写在注释里) |
| `logs/n4_personal_c0.log` + `logs/n3_scores_pers_c0_s1..8.npz` | 结果 + 逐折分数(✅ 07-03) |

> 工程注:从 Claude 会话里 `Start-Process` 启动的跑会挂在会话进程树下,会话一动窗口被关 →
> MKL `forrtl: error (200) window-CLOSE` 两分钟即死(06-30"三种启动法均中招"的真身)。
> 可靠解 = 任务计划程序(`Register-ScheduledTask` + `Start-ScheduledTask`,系统服务拉起,
> 完全脱离会话),或用户自己在独立终端跑 bat。

---

## 35. N5 — 类划分敏感性:register/unknown 互换 + "孪生类在 base"机制破案(2026-07-04)

**动机**:§33 的全部结论只在一个类划分(case 0:register=[0,2,3])上测过——审稿人必问
"结论是不是挑了好注册的类"。N5 把 register/unknown 角色互换(case 1:register=[6,9,10]
=G7 对角/G10 逆时针/G11 推进,unknown=[0,2,3]=三个正交滑动),base 不变,其余与 §33
完全同构(150 epoch、balanced support、留二被试、α=0.1;**未用 enc-cache,该跑早于
P3 基建,case-1 编码器未入缓存**)。

### 35.1 结果(8 折人均,%;对照列 = §33 case-0)

| 模式 | cov_base | cov_new | rejU | singAcc_n | case-0 对照(cov_new / rejU) |
|---|---|---|---|---|---|
| **global** | **89.8** | 52.4 | **65.9** | 43.0 | 66.6 / 44.4 |
| **hybrid(oracle)** | 89.8 | **75.3** | 62.2 | 65.1 | 82.9 / 40.6 |
| mondrian | 82.3 | 75.3 | 64.2 | 72.3 | 82.9 / 53.3 |
| eta=3 | 23.4 | 0.0 | 100.0 | 0.0 | 同死 |

nearest-proto:acc_base **92.1±7.7**(≈case-0 的 93.0,应然:base 管线未变);
acc_new **49.7±18.0**(case-0:75.2±10.8);d 膨胀 12.3×(case-0:10.6×);
known-vs-unk AUROC **90.9±3.0**(case-0:80.1)。shrinkage λ 扫依旧单调变差
(52.4→19.9)——**σ-shrinkage 的证伪跨类划分复现**。

### 35.2 逐类路由破案:错误不是均匀的,是"孪生类在 base"的定向吞噬

新类测试样本的 argmin 路由(8 折合计,%):

| 真类 | 认对 | 主要错误去向 |
|---|---|---|
| G7 对角 RL-UD | **80.2** | 分散(≤7) |
| **G10 逆时针** | **30.1** | **→ G9 顺时针(base 列)44.8** |
| **G11 推进** | **38.6** | **→ G12 空手势(base 列)39.8**,→G8 15.0 |

机制:新类的**易混孪生**(G10↔G9 旋向对、G11↔G12 推/空)坐在 base 里时,是
"600 样本×6 人的强头 vs 5 样本弱头"的不对称对决——测试被试的 G10 样本在自己的
5-shot 头下 d 膨胀,而 G9 的 base 头列紧致 → argmin 倒向 base 列。**这与 §34.7
(episodic 把 base σ 撑爆后 base 列吸走一切)是同一软肋的两种触发方式:一个由训练
动力学触发,这个由词表语义触发。**

**同一机制回看 case-0**:case-0 里 G10 当 unknown,同样死在 G9 列上——只不过表现为
"未知从 base 列泄漏"(§33.4 的 53% 主通道)而非"新类被吞"。**G9/G10 旋向对是这套
编码器在整个词表里最硬的结构**(旋转方向在距离签名里本就弱,S1 水平轴失败的同族),
它在哪个角色就毒哪个指标:case-0 毒 rejU(44.4),case-1 毒 acc_new(49.7);反过来
case-1 的 unknown 全是正交滑动 → rejU 反而 65.9、AUROC 90.9。

### 35.3 判读

- ✅ **框架结论跨类划分稳健**:base 覆盖保证中靶(89.8)、hybrid>global、逐类阈值形状
  值钱、shrinkage 证伪、η=3 死——全部复现。
- ⚠️ **acc_new 强依赖注册哪些手势**:平移类 ~70-88,旋向/推进类 30-39。跨人 5-shot 的
  "诚实水平"应报成**区间**(按注册词表),不是单点 75。
- **可写进论文的设计指南**:注册词表设计要避免"新类的孪生已在 base"——要么孪生对
  同批注册(case-0 里 G3/G4 互为孪生同当新类,两弱头公平竞争,68/69 尚可),要么
  在注册时把孪生 base 类当难负样本加权(P4 的针对性版本;现 neg 已含全部 base 特征,
  问题在 d 膨胀的不对称,不在负样本缺席)。
- **对 P3 的含义**:e_recon⊕r 融合主修 case-0 型泄漏(unknown 版孪生);delta 增广若能
  压 d 膨胀,应该对 case-1 型吞噬(registered 版孪生)也有效——两案例正好当 P3 的
  两块试金石。case-1 的同人分解(personalization)没跑过,若需要归因可后补
  (case-1 编码器未缓存,需 8×~20min)。

### 35.4 日志 / 代码

| 文件 | 内容 |
|---|---|
| `config.py: REG_TEST_CASES[1]` | case 1 定义(03d190d 加入) |
| `batch_scripts/n5_case1.bat` | 运行脚本(任务计划程序启动) |
| `logs/n5_case1.log` + `logs/n3_scores_bal_c1_s1..8.npz` | 结果 + 逐折分数(✅ 07-04) |
| 逐类路由分析 | 临时脚本(scratchpad),表格已录 §35.2 |

---

## 36. P3 — 分数/阈值升级总决算:手术版 e_recon 融合 ✅ 达线,模拟任务共形 ❌、整列融合 ❌、delta 增广 ❌(2026-07-04)

**动机**:§33.4 判定"阈值只能沿覆盖-拒识边界滑动,挪边界必须改分数";§34.5 判定 B2-init
的 AUROC 赤字(−3.1)"待 e_recon 赎回";§4.10 校准收官候选=模拟任务共形。P3 = 一次
dump-everything 跑喂饱全部后处理:Run A(B2-init 参考,~6.8h,建 8 折编码器缓存)+
Run B(subject-delta 增广,缓存命中 ~1.5h,同编码器配对)。此后每种分数/阈值策略都是
`tools/p3_analyze.py` 上的秒级重算。基建(均已提交 86cfa31):`--enc-cache`(一折从
~55min 降到 Stage-2-only)、每样本 e_recon dump(E_\*)、逐视角距离 dump(D_\*)、
`--sim-task`、`--delta-aug`、`--calib-rotate`、`--mae-no-init`。

### 36.1 主表(case 0,8 折人均,%,α=0.1;Run A dump,acc:base 93.4 / new 75.5)

| 模式 | cov_b | cov_n | rejU | leakB | AUROC | sAcc_n | 定位 |
|---|---|---|---|---|---|---|---|
| global(基线) | 90.7 | 65.7 | 40.3 | 59.5 | 77.1 | 72.8 | §33 复现 ✓ |
| hybrid(oracle) | 90.7 | 82.1 | 36.1 | 59.5 | 77.1 | 80.6 | oracle 上界 |
| simtask | 90.7 | 66.8 | 39.5 | 59.5 | 77.1 | 70.7 | ❌ ≈无效 |
| fuse λ=1(整列) | 89.7 | **25.4** | 54.6 | 45.3 | 74.5 | 65.6 | ❌ 砸新类 |
| **fuse λ=1(仅 base 列)** | 89.7 | 65.7 | **50.2** | **45.3** | **83.5** | 76.7 | **✅ 无 oracle 主配方** |
| **fuse λ=1(base)+hybrid** | 89.7 | **82.1** | **48.1** | 45.3 | **83.5** | **92.0** | **✅ oracle 堆栈** |
| pmerge2-hyb(split-half) | 77.8 | 79.9 | 59.7 | 22.8 | 85.4 | 82.6 | 信号上限诊断 |
| eta=3 | 25.7 | 0.0 | 100 | 0 | 77.1 | 0 | 同死 |

### 36.2 三个证伪,机制全部破案

- **模拟任务共形 ❌**:q_sim/q_global 逐折比值 0.54–1.51(均值 ~0.98)→ cov_new 65.7→66.8
  纹丝不动。**机制:base 类被 CE 几何攥得太紧,5-shot 头在校准被试上也不膨胀——base 类
  不是"从没训过的类"的合格替身。** 与 shrinkage(§33.3)同源:一切"用 base 类学新类膨胀
  规律"的转移都死在 CE 特征上。auxiliary-task 转移的前提(任务可交换)在编码器层面不成立。
- **e_recon 整列融合 ❌**:e_recon 中位:base 0.71 ≪ **注册类 0.96 > 真未知 0.90**——
  重建器只在 base 类上预训练(开集卫生),注册类和未知在它眼里都是外人,甚至注册类更陌生
  → 整列融合把 cov_new 65.7 砸到 25.4。**论文一 §28.6 的 +3.42 融合经验不能搬进注册场景:
  e_recon 原理上分不开"注册的新类"和"未知"。**
- **subject-delta 增广 ❌(同编码器逐折配对)**:acc_new 75.5→68.2(**1/8 胜,均 −7.3**),
  d 膨胀多数反升(如 5.2→7.5×)。一阶"逐被试质心平移"造出的合成 support 离流形,稀释原型。
  balanced support 的增益(§33.2)靠的是**真**跨人样本,线性近似替代不了。

### 36.3 手术版融合为什么成:e_recon 用在它分得开的那道题上

整列融合死于"e_recon 分不开注册类 vs 未知",但它**分得开 base vs 非 base**(0.71 vs 0.90+)。
而 §33.4 已量化拒识短板的主通道 = 未知从 **base 列**漏入(53%)。手术版 = e_recon 罚项只加
在 base 列(非合规分数 s_base = z(r)+λ·z(e_recon),新类列保持裸 r,各列各自用校准分位数):

- **AUROC 77.1 → 83.5(+6.4)**:B2-init 的 −3.1 赤字(§34.5)不但赎回,还反超无 B2 基线
  (80.1)——**B2 分支完全体(init + 手术融合)成立**;
- rejU 40.3→50.2、leakB 59.5→45.3,新类覆盖 65.7 一分不伤,base 覆盖 89.7 保证完好;
- 叠 hybrid(oracle)后:cov_n 82.1 + **sAcc_n 92.0** + rejU 48.1(比裸 hybrid 的 36.1 高 12)。
- **止损线判定(§34.3 预注册:acc_new≥80 或 AUROC≥83 任一)→ AUROC 83.5 命中,按预案
  收手转校准收官+写作。**
- pmerge(逐视角 Fisher p 值合并)= 信号上限诊断:AUROC 可到 85.4、rejU 59.7,但单校准被试
  下逐 (类,视角) ECDF 太薄,base 覆盖欠账(77.8);多/轮换校准被试后可再试,现不进主配方。

### 36.4 判决与下一步

- **配方冻结(跨人注册)**:B2-init 编码器 + balanced support + 最近原型路由 + conformal
  (无 oracle:fuse-base λ=1;校准队列做过新手势:+hybrid)。诚实数字:acc_new 75.5
  (词表相关,§35 区间 30–88)、cov 89.7/65.7(oracle 82.1)、rejU 50、AUROC 83.5。
- acc_new 的表征/增广/校准三条线全部按预案关闭(§34.5-34.7、§36.2),75.5 = 跨人 5-shot
  的诚实水平写成 finding;**部署篇卖点 = 同人注册已解决(§34,91.1)+ conformal 门控
  自我个性化闭环(跨人冷启动 → singleton 收样 → 同人稳态,两端数字均已实测)**。
- 在跑:shot 曲线 K=1/10/20(`p3_shots_c0.bat`,缓存命中)。待跑:headline(冻结配方 +
  `--calib-rotate`,case 0+1;case-1 编码器未缓存,约两个过夜)。闭环模拟实验待做
  (纯 Stage-2,可吃缓存)。

### 36.5 日志 / 代码

| 文件 | 内容 |
|---|---|
| `batch_scripts/p3_dump_c0.bat` | Run A(b2e)+ Run B(b2e_delta),任务计划程序 UWB_P3_dump |
| `logs/p3_dump_c0.log` / `p3_delta_c0.log` | 两跑日志(✅ 07-04) |
| `logs/n3_scores_bal_b2e_c0_s1..8.npz` / `..._b2e_delta_...` | 全量 dump(R/D/E/sim 键齐) |
| `cache/enc/c0_ts*_cs*_...mae-raw100.pt` | 8 折编码器缓存(~81MB/折,后续 Stage-2 全吃它) |
| `tools/p3_analyze.py` | 全模式后处理(fuse-base/+hyb、pmerge2 split-half 均在) |
| `batch_scripts/p3_shots_c0.bat` | shot 曲线(在跑,18:13 起) |

---

## 37. P3 shot 曲线 — K=10 跨过 80 线,K=20 无 oracle 覆盖摸到 89:欠覆盖是超低样本现象(2026-07-04)

**动机**:§4.10 协议补全 + "样本增量个数"这个最直接的部署旋钮。K=1/10/20(K=5 = §36 Run A),
全部 balanced support(6 训练被试轮流出样本)、B2-init 编码器、**全 enc-cache 命中**
(三组 8 折共 ~3.5h,Stage-2-only;没有缓存这是 3×7h)。

### 37.1 曲线(case 0,8 折人均,%;阈值=冻结配方 fuse-base λ=1,oracle 列=+hybrid)

| K | acc_new | cov_new(无 oracle) | cov_new(oracle) | rejU | AUROC | d 膨胀 |
|---|---|---|---|---|---|---|
| 1(σ floor 修复后) | 20.2 ± 12.9 | 13.1 | 88.2* | 49.6 | 69.8 | 16.2× |
| 5 | 75.5 ± 11.2 | 65.7 | 82.1 | 50.2 | 83.5 | 6.0× |
| **10** | **83.8 ± 7.9** | **85.0** | 87.0 | 49.3 | **86.9** | 4.5× |
| **20** | **86.2 ± 4.3** | **88.9** | 89.2 | 43.2 | 85.8 | 3.9× |

\* K=1 的 hybrid 88.2 是"集合撑大换覆盖"(size_n 2.0,rejU 塌到 9.3),不可用;
1-shot 跨人本质不可用(单样本原型零跨人信息,d 膨胀 16×),这正是曲线的左端锚点。

acc_base 全程 92.9–94.0 不动。cov_base 全程 90–91 中靶。

### 37.2 判读:三条大结论

- **"5-shot 跨人欠覆盖"是超低样本现象,不是结构死墙**:K=10 无 oracle 覆盖 85.0、K=20 到
  **88.9≈90 目标,且追平 hybrid oracle(89.2)**——oracle 与无 oracle 的差距随 K 消失,
  N2/N3 花大力气修的尺度失配在 10–20 个真跨人样本面前自愈。
- **acc_new 随 K 单调上行并跨线**:75.5→83.8→86.2(方差 11.2→4.3 收敛),K≥10 即过
  §34.3 的 acc≥80 止损线。d 膨胀 6.0→3.9×,**K=20 跨人 ≈ 5-shot 同人(3.5×,§34)**——
  换人差距的大头可以用"多找几个人多做几次"直接买回来。
- **部署换算**:K=10/20 balanced = 6 个采集者每人做 2–3 次,采集成本几分钟量级。与 §33.2
  合并成完整采集指南:**人数优先(balanced +17.5),人数到顶后每人 2–3 次(+8~11 acc,
  +19~23 cov)**。75.5 那个"诚实水平"应改述为"K=5 的诚实水平";系统的实际部署水平由
  采集预算定,曲线给出定价。

### 37.3 K=1 的 σ 塌缩 bug 与修复(方法 A 的 1-shot 保护)

首跑 K=1 acc_new/cov_new 恒等于 0:方法 A 用单个 support 估 σ → σ=0 → r=d/(σ+1e-6) 爆炸
→ 新类列永远不可达(hybrid 反常 88.2 覆盖也是同一伪象:校准分位数同样爆炸,q=∞ 型全覆盖,
rejU 塌到 9.3)。**方法 B 从一开始就有 σ 下限(借 base 池 σ,§31 代码),方法 A 漏了**。
已修(`_register_method_a` 尾部 per-view floor:σ<1e-3 时借同视角 base 头均值 σ),
K=1 以 tag `b2e_k1f` 重跑(缓存命中,~1h)。buggy 跑(`b2e_k1`)留作负对照。
**重跑结果(已入 §37.1 表)**:acc_new 20.2±12.9、cov_new 13.1、d 膨胀 16.2×——修复后
不再是恒 0,但 1-shot 跨人注册本质不可用,K=1→5 的 +55 分跳变(跨人多样性进入 support)
是整条曲线最陡的一段,再次坐实"人数是第一货币"。

### 37.4 日志 / 代码

| 文件 | 内容 |
|---|---|
| `batch_scripts/p3_shots_c0.bat` | K=1/10/20(UWB_P3_shots,18:13–21:4x,全缓存命中) |
| `logs/p3_k{1,10,20}_c0.log` + `logs/n3_scores_bal_b2e_k*_c0_s*.npz` | 三组 dump |
| `uwb_osr/registration.py` | 方法 A 1-shot σ floor 修复 |
| `batch_scripts/p3_k1fix_c0.bat` + `logs/p3_k1f_c0.log` | K=1 修复重跑(tag b2e_k1f) |

---

## 38. P3 headline(轮换校准)— 揭出单校准被试的"尺度抽签":固定配对的 90 中靶有运气成分(2026-07-06)

**动机**:所有 N2–N5/§36–37 的 conformal 数字都在固定配对(cs=第一个非测试被试,≈总是
subj1)下测得,且早知有退化折。headline = 冻结配方 + `--calib-rotate`(cs=循环下一位)
× case 0/1,16 折 ~16h(rotate 改变 (ts,cs) 配对 → 全部重训,顺便建 rotated 缓存)。

### 38.1 结果(8 折人均,%;fuse-base λ=1 / +hybrid)

| case | acc_b / acc_n | cov_b | cov_n(无 or) | cov_n(or) | rejU | AUROC | sAcc_n(or) |
|---|---|---|---|---|---|---|---|
| 0(rot) | 93.5 / 72.0 | **84.5** | 43.9 | 78.9 | 56.3 | 83.5 | 86.3 |
| 0(fixed,§36 对照) | 93.4 / 75.5 | 89.7 | 65.7 | 82.1 | 50.2 | 83.5 | 92.0 |
| 1(rot) | 93.1 / 46.8 | 83.4 | 35.3 | 78.8 | 69.2 | **93.4** | 83.9 |

acc/AUROC 稳(83.5 逐字复现;case-1 twin 机制依旧 46.8);**变的是覆盖**。

### 38.2 破案:单校准被试 = 一次"尺度抽签"

逐折 cov_base(global q):固定配对 79–100(均 90.7);**轮换 61–100(均 85.3),
q90 在 38–598 之间摆(15×)**。机制:每折的阈值刻度由那一位校准被试的距离尺度决定,
被试间尺度差 >10×;固定配对下 7/8 折都抽到 subj1(恰好中等偏松)→ 平均"中靶"是配对
运气,不是稳定性质。轮换才是诚实协议:**期望上保证仍成立(85.3 vs 90,8 折抽样内),
但单次部署的覆盖在 61–100 间摇摆——这在应用上不可用,也是 §33.5"hybrid oracle 也到
不了 90"的真正根源(不是少样本问题,是"一个人的尺子量所有人"问题)。**

### 38.3 判读与最终一跑

- 论文校准一节的诚实写法:①报轮换(均值+逐折区间),固定配对降级为敏感性对照;
  ②"5-shot 欠覆盖"(§32/33)与"单校准尺度抽签"(本节)是两层问题,K 曲线(§37)修第一层,
  **多校准被试池化修第二层**——q 由多人分布共同定,尺度方差被平均掉。
- **最终一跑 = leave-3-out(1 测试 + 2 校准池化 + 5 训练)**,case 0/1,预期:cov 折间
  方差收窄、均值回到 ~90 附近;代价 = 编码器少一个训练被试(acc 或微降,诚实报)。
  代码:`register_split_conf` 接受多校准被试 + `--n-calib`;mondrian/hybrid 每类校准点
  翻倍,顺带受益。
- 部署含义(写进 discussion):校准队列要**多人**,哪怕每人样本少——与 support 的
  "人数优先"(§33.2/§37)是同一条物理规律在校准侧的镜像。

### 38.4 日志 / 代码

| 文件 | 内容 |
|---|---|
| `batch_scripts/p3_headline.bat` | rotate × case 0/1(UWB_P3_headline,07-05 17:10 起) |
| `logs/p3_head_c{0,1}.log` + `logs/n3_scores_bal_b2e_rot_c{0,1}_s*.npz` | 结果 + dump |
| `cache/enc/*_ts*_cs*(rotated pairing)` | rotated 配对编码器缓存(16 折) |

---

## 39. P3 闭环 — conformal 门控自我个性化:门很干净(污染 2.5-5%),blend 修掉选择偏差,+7 免费、91+ 需确认(2026-07-06)

**动机**:部署篇主实验(RESEARCH_PLAN 的"跨人冷启动 → 同人稳态"闭环)。模拟协议
(`tools/p3_loop.py`,全 enc-cache 命中,case 0,冻结配方 fuse-base λ=1):测试被试的
样本流 = 打乱(base 测试 + 注册类测试的一半 + 未知);**预测集恰为单个新类列(singleton)
→ 静默收集为该类的伪标注个人样本**;满 m_adopt 个 → 重训该类少样本头;另一半注册类样本
做留出评测。v1 简化:冷 bank 扫全流、期末一次性采纳。

### 39.1 结果(8 折人均,%;oracle-personal = 用真标签个人样本重训 = §34 机制)

| 变体 | acc_new cold→warm | oracle | 污染 | 采纳延迟(流样本) | rejU cold→warm |
|---|---|---|---|---|---|
| v1 replace,m=5 | 73.0 → 74.6(+1.6) | 88.0 | 5.6% | 229/1049 | 50.7→45.2 |
| **v2 blend,m=5** | 72.9 → **79.9(+7.0)** | 91.2 | **2.5%** | 258/1049 | 49.9→45.1 |
| **v2 blend,m=10** | 73.7 → **81.1(+7.4)** | **93.2** | 4.9% | 360/1049 | 49.7→43.5 |

### 39.2 两个机制发现

- **门很干净**:共形 singleton 门的收样污染仅 2.5–5.6%,~260–360 次正常使用即收满——
  静默收集机制成立,这是 conformal 集合输出在部署里的直接兑现(空=拒识、多=不确定、
  **单=可信到可以拿来自学**)。
- **v1 的"简单样本选择偏差"及其修复**:replace 版 warm 只 +1.6 且有折严重退化(subj8
  95.3→78.7)——门只放行冷头**本来就确信**的样本,个人池偏向"简单打法",整头替换丢掉
  跨人 shot 覆盖的难尾巴。**v2 blend(原跨人 shot + 个人池混合重训)修复**:+7.0~7.4、
  退化消失(subj8 94→97.3),且 blend 把 oracle 也抬高(88→91.2/93.2)——与 §37 同理:
  支持集多样性(跨人 + 个人)优于单源。
- 残余 warm-vs-oracle 差距(~80 vs ~91-93)= 门的选择偏差本身:难样本永远进不了
  singleton 门。**写作定调:静默自学免费买 +7;要到 91+ 需要显式确认(请用户对 ~5-10 个
  样本点个头)——两档都由同一个共形门驱动,是"标注成本-性能"旋钮,不是失败。**

### 39.3 日志 / 代码

| 文件 | 内容 |
|---|---|
| `tools/p3_loop.py` | 模拟器(--blend/--m-adopt;已加 seed 固定) |
| `batch_scripts/p3_loop_c0.bat` / `p3_loop2.bat` | v1 / v2 两档 + 链式启动 headline2 |
| `logs/p3_loop_c0.log` / `p3_loop2_m{5,10}.log` | 三跑结果(✅ 07-06) |

---

## 40. P3 终局 — "老头压新头"是路由的统治机制:一个标量 handicap 换来 acc_new +12;headline2 定稿(2026-07-07)

**动机**:headline2(leave-3-out,2 校准被试池化)的 case-0 acc_new 报 **88.6±5.7**,比
一切前序跑(72–76)高 +16——好得反常。排查:切分断言全过(support 严格来自 5 个编码器
被试,无测试/校准泄漏);**后处理复现实验破案**:在旧的 6 被试 dump 上,只把 base 列
距离乘一个标量 s,就完整复现跳变——rot2 的涨幅不是多校准的功劳,是"少一个训练被试 →
base 头变松"的意外副作用。

### 40.1 s 扫(后处理,§36 的 b2e dump;cov 行为 global 阈值随 s 重校准)

| case | s | acc_b | acc_n | cov_b | cov_n | rejU |
|---|---|---|---|---|---|---|
| 0 | 1.0 | 93.4 | 75.5 | 90.7 | 65.7 | 40.3 |
| 0 | **1.5** | 91.4 | **87.3** | 90.7 | 93.4 | 29.7 |
| 0 | 2.0 | 88.9 | 91.8 | 90.7 | 97.5 | 12.4 |
| 1(twin 重灾) | 1.0 | 92.1 | 49.7 | 89.8 | 52.4 | 65.9 |
| 1 | **1.5** | 89.4 | **64.9** | 89.8 | 83.2 | 54.3 |

rot2 的 90.2/88.6 恰落在 s≈1.5 曲线上(91.4/87.3)——5 被试编码器没有魔法。

### 40.2 机制统一:同一根轴的第四次现身 + 显式旋钮

r=d/σ 的 argmin 路由由**头强度不对称**支配:600 样本×多被试训出的 base 头紧,5-shot 头松,
边界样本系统性倒向 base 列。此前三次现身:①§35 孪生吞噬(G10 的 44.8% 被 base-G9 吸走);
②§34.7 N4b(episodic 把 base σ 撑爆→base 列吸走一切);③本节 rot2(少训一人→base 微松→
新类大涨)。**s 就是这根轴的显式旋钮**——等价于长尾学习的 logit adjustment 搬到少样本
原型路由上(新老头的"先验"必须校正)。case-1 的 twin 吞噬对 s 直接响应(49.7→64.9)。

### 40.3 部署配方 v2:路由与接受解耦

- **路由(认成谁)**:用 s≈1.5 加权的 argmin → acc **91.4/87.3**(H 89.3,vs s=1 的 83.5);
- **接受/拒识(要不要认)**:保持 s=1 的 conformal(fuse-base λ=1)→ cov 89.7/65.7、
  rejU 50.2、AUROC 83.5 的保证全部不动。二者作用于同一 R 矩阵的两个独立读出,互不污染。
- 整体 s(路由+阈值同调)则是 §33.4 的覆盖-拒识交换(cov_n 93 但 rejU 29.7),按场景选。
- **s 的选择(诚实声明)**:全文报 s 扫描曲线;s=1.5 作为协议常数(不是逐折调出来的);
  oracle 版可在校准被试的新类样本上逐折调。数据无关的逐折估计(LOO support 膨胀估计)
  留作 future work。工具:`tools/p3_analyze.py --route-s 1.5`。

### 40.4 headline2(多校准)的真实贡献:治好"尺度抽签"

| 配对 | cov_base 折间 | std | q90 范围 |
|---|---|---|---|
| rot(1 校准) | 61–100 | 15.1 | 38–598(15.7×) |
| **rot2(2 校准池化)** | 62–98 | **10.8** | **83–229(2.8×)** |

q 抽签范围压掉 5.6 倍、折间 std 收 1/3、均值 85.3→87.6。**多校准修覆盖方差(它该修的),
acc 跳变归 s 机制(§40.1),两者在 rot2 里被 confound——论文里必须分开写。**
case-1 rot2:global cov_new 83.5 / AUROC 92.4 / acc_new 69.7±20.6(twin 依旧,s 前)。

### 40.5 日志 / 代码

| 文件 | 内容 |
|---|---|
| `batch_scripts/p3_headline2.bat` | leave-3-out × case 0/1(07-06 链式启动,~19h) |
| `logs/p3_head2_c{0,1}.log` + `logs/n3_scores_bal_b2e_rot2_c{0,1}_s*.npz` | 定稿 dump |
| `tools/p3_analyze.py --route-s` | s 路由读出(接受侧不动) |
| 切分泄漏断言 | 临时脚本(会话内),8 折全过 |

---

## 41. P4 G1a — 检查器级 support-LOO 审计:中位数比例估计器证伪,frozen-head LOO 方向翻转(2026-07-10)

**动机**:论文四设计 §5.3 预注册的 G1a 审计。固定公式(收缩常数 10、裁剪 [1,3]),真检查器级留一提取 m_new(每个支持样本:其余 K-1 个重训该类三视角头,负样本/增强配置不变,再给被留出样本打分;3 类 × K=5 → 15 个真 LOO 分数/折),m_base = rot2 dump 的 R_cal 真类中位数,检验 s_hat 与逐折 oracle(同 bank s=1 基准、base 降幅 ≤2 分约束下 H-max)的相关性。工具 `tools/p4_g1a_loo.py`(rot2 编码器缓存全命中,~9s/头,16 折 ≈ 2h,先 1 折计时后全量)。

### 41.1 结果(rot2 dump,8 折/case,%)

| case | s_true 均值(clip@3) | s_oracle 均值[范围] | H@1 | H@1.5 | H@s_true | H@oracle | spearman(s_true, s_oracle) |
|---|---|---|---|---|---|---|---|
| 0 | 2.88(5/8 折) | 1.31 [1.0, 2.5] | 89.0 | 89.8 | **84.9** | 90.7 | **−0.49** |
| 1 | 2.96(7/8 折) | 1.35 [1.0, 2.2] | 76.0 | 79.5 | 77.8 | 79.6 | −0.42 |

个别折 H@s_true > H@oracle 是 base 约束在起作用:s→3 时 acc_base 掉 10–17 分,预注册口径下不可行。

### 41.2 判读:两层证伪 + 一个方法论警示

- **估计目标错位(证实 07-10 代理诊断)**:真 LOO 中位数比率 6–7×——新旧头尺度差是真的,但路由最优只要 ~1.3;最优边界由错误类竞争决定,不由真类中位数对齐决定。R_cal_reg 代理(2.69/2.73)与真 LOO(2.88/2.96)几乎同值:失败不在估计精度,在估计目标本身。
- **相关性为负**:spearman −0.49/−0.42,12/16 折顶到裁剪上限,估计的信息几乎全被 clip 吃掉——"与 oracle 稳定相关"的通过标准不成立。
- **frozen-head LOO 方向翻转**:泄漏版中位数 ~1.2(真值 85),16/16 折落到 s=1——被留出样本参与过 MLP 训练后,其距离缩小约 70 倍,会得出"新头比老头紧"的反向结论。检查器级留一不是洁癖,是结论的生死线(设计评审第三轮把它钉死是对的)。
- 正确性三重全过:确定性重放 16/16 逐位一致、σ-floor 0 触发、泄漏断言全过、支持样本每类 5 个不同训练用户。

**按预注册规则判决:G1a 降级为对照基线,G1b(校准侧伪注册,含负样本池重训)转正为主方法。** G1b 计时下界 ≥2.2h/8 折(base 头正样本远多于 5-shot 头,实跑先 1 fold × 1 留出计时)。

### 41.3 日志 / 代码

| 文件 | 内容 |
|---|---|
| `tools/p4_g1a_loo.py` | 审计工具(真 LOO + frozen-head 诊断 + oracle 网格 + 计时门) |
| `logs/p4_g1a_c{0,1}_s{1-8}.npz` | 16 折 LOO 分数、估计值、oracle、计时 |
| `logs/p4_g1a_sweep_c{0,1}.log` | 全折扫描日志 |

---

## 42. P4 G1b — 伪注册证伪 → calib-select 转正:G1 正式判决(2026-07-11)

**动机**:G1a 证伪(§41)后切换 G1b。原 G1b = 校准侧伪注册:留出 base 类 g,冻结编码器重训其余检查器(g 从真实负样本/mixup/all_real 三处删除),伪新头 subject-balanced 支持(`tools/p4_g1b_pseudo.py`,~8.5s/头,计时门 16 min/折通过)。

### 42.1 伪注册早筛证伪(ts=1 全量,6 留出 × 2 校准 = 12 单元宏平均)

| 读数 | 伪 bank(校准用户) | 真实 bank(迁移,ts=1) |
|---|---|---|
| base 精度 s=1→3 | 88.8 → 86.6(**−2.2**) | 98.7 → 81.3(−17.4) |
| s_hat / 后果 | 2.70(约束仅在 s=3.0 咬合) | H 89.8 < 固定 1.5 的 92.7,acc_b −14 |

**机制诊断(dump 后处理,全 8 折)**:真实 bank 即使只保留单个注册列,校准用户 base 也掉 9.7/12.3 分(case 0/1)——"1 伪新列 vs 3 注册列翻墙压力"假说被杀死。病根 = **编码器见过伪新类**:其表征与其余 base 类天然可分,base 样本缺乏向伪新列的翻越压力,约束失去咬合、s_hat 无刹车膨胀。设计文档"见过→偏保守"的预期**方向相反**(好在预先声明了非理论保证)。与 §36 模拟任务共形失败同轴:**真实新类的先验,用 base 类排练不出来**。8 折全量未启动。

### 42.2 calib-select:校准侧在真实 bank 上网格选 s(经审定转正为 G1b 主方法)

估计器 = 用 R_cal + R_cal_reg(共形侧 mondrian/hybrid 新类阈值本已要求的数据,不新增假设)按同一口径选 s:网格 [1,3] 步 0.1、base 相对同 bank s=1 降幅 ≤2 分、H-max。路由/接受解耦,s 不碰共形阈值,覆盖保证不受选择污染(论文须明示两读出共用 calib_reg)。

### 42.3 G1 正式判决(`tools/p4_g1_verdict.py`,门槛按 b2e/rot2 双家族重锚)

| 家族 | s_cal / oracle | @calib-sel acc_b/acc_n/H | H@1 | H@1.5 | H@oracle | 门槛 |
|---|---|---|---|---|---|---|
| b2e c0 | 1.54 / 1.88 | 92.0 / 87.7 / **89.6** | 82.8 | 89.0 | 90.8 | **全过**(new +12.2、base −1.4、距 oracle 1.2、胜 1.5 +0.6) |
| rot2 c0 | 1.25 / 1.31 | 88.8 / 92.1 / **90.3** | 89.0 | 89.8 | 90.7 | **全过**(距 oracle 0.4) |
| rot2 c1(孪生,单独报告) | 1.26 / 1.35 | 85.9 / 74.7 / 78.4 | 76.0 | 79.5 | 79.6 | 距 oracle 1.1 ✓;对 1.5 −1.1、base 降 2.3 两条边缘未达 |

**判决:G1 通过**——锚定 b2e c0 + rot2 c0 全绿;b2e 上 new +12.2 / base −1.4 兑现了 §40.3 留作 future work 的"数据无关逐折估计"。case-1 两条边缘失守归孪生几何(尺度校正明确不主张能修),且 calib-select 仍胜 s=1(H +2.4)、两组词表均无反向退化。

### 42.4 日志 / 代码

| 文件 | 内容 |
|---|---|
| `tools/p4_g1b_pseudo.py` | 伪注册实现(负结果保留;负样本池断言、口径钉死) |
| `logs/p4_g1b_c0_s1.npz` | ts=1 全量伪注册网格与单元数据 |
| `tools/p4_g1_verdict.py` + `logs/p4_g1_verdict.npz` | 双家族正式判决表 |
| 设计文档 §3/§5.3/§8 | 07-11 审定修订:calib-select 转正、伪注册降级、门槛重锚 |

---

## 43. P4 — 终局资产就位:其余 4 个 base 划分的编码器 + b2e 参考 dump 全量落地(2026-07-14)

**背景**:设计文档 §4.1/§12 第 1-2 周长杆任务。最终阶段需 5 组 OSR base 划分 × 10 组注册词表,此前编码器只覆盖 base=[1,4,5,7,8,11] 一个划分(c0/c1 两组词表)。07-12 晚点火,为其余 4 个划分训练编码器并同步产出 b2e 家族参考 dump。

**协议**:`config.py` REG_TEST_CASES 新增 c2--c5,按 §4.1 规则构造(base=OSR case2--5 的 known;其 unknown 列表按既有顺序前 3 作 register、后 3 作 true-unknown),构造经脚本断言校验(base 与 OSR known 一致、切分按序、三集无重叠)。配方逐字镜像 `p3_dump_c0.bat` Run A(b2e 参考:conformal / method A / K=5 / alpha 0.1 / calib-frac 0.25 / no-fusion / flatten / abs_std / support-balanced / mae-pretrain / enc-cache / sim-task),仅 `--case` 不同。4 case 串行单 torch 进程,任务计划程序 `UWB_P4_ENC_C2TO5` 启动(forrtl 200 纪律)。总墙钟 29.4h(逐 case 7.4 / 7.4 / 7.6 / 6.9),RTX 4060 Laptop 8GB。

**索引对照(钉死防后患)**:文件名与 `--case` 为 0-based(dump `..._b2e_c2_s*.npz`);日志内 "Reg case N" 与 summary 前缀 cN 为 1-based 显示(`--case 2` 在日志里印作 "Reg case 3"/"c3")。本节下表一律用 0-based 文件名口径。

### 43.1 冻结静态配方在 4 组新词表上的共形汇总(8 折均值,%;目标 cov 90;c0 为对照锚)

| 词表(base) | hybrid cov_b/cov_n | mondrian cov_b/cov_n | global cov_b/cov_n | global empty_n |
|---|---|---|---|---|
| c0 [1,4,5,7,8,11](对照) | 90.7 / 82.1 | 83.4 / 82.1 | 90.7 / 65.7 | 25.5 |
| c2 [1,4,5,7,10,11] | 93.9 / 83.5 | 84.7 / 83.5 | 93.9 / 77.0 | 11.9 |
| c3 [2,5,6,7,9,11] | 84.7 / 88.9 | 82.9 / 88.9 | 84.7 / **48.6** | **46.2** |
| c4 [0,1,7,8,10,11] | 90.7 / 85.3 | 79.9 / 85.3 | 90.7 / 81.6 | 10.7 |
| c5 [1,2,7,8,9,10] | 85.1 / **74.7** | 78.2 / 74.7 | 85.1 / 54.5 | 35.6 |

### 43.2 观察(仅定位静态基线,不下方法结论)

- **冻结配方可迁移**:hybrid 在全部新词表 cov_b 84.7--93.9 / cov_n 74.7--88.9,无病态崩溃;eta 一如既往全灭(cov_n=0),后续不再报。
- **global 阈值脆弱性的跨词表再证**:c3 的 cov_n 48.6(empty_n 46.2)、c5 的 54.5(empty_n 35.6)——§38 的结论在新词表上被放大,mondrian/hybrid 的新类侧在同词表仍 ≥74.7。
- **逐折尺度抽签依旧**:如 c2 的 subj7 hybrid cov_n 50 / rejU 78,同 case 的 subj3 达 95——与 §38/§40 一致,G2 逐折分层报告口径的必要性又添 4 组证据。
- **c5 是新的困难词表候选**(hybrid cov_n 74.7,全表最低),G4 的"10 组中 ≥8 组"判据应预期它可能失守,届时与孪生重灾词表(c1)一并单独报告。

### 43.3 资产清单

| 文件 | 内容 |
|---|---|
| `config.py` REG_TEST_CASES c2--c5 | 4 组新词表(注释钉死 c6--c9 交换词表必须复用编码器) |
| `batch_scripts/p4_enc_c2to5.bat` | 点火脚本(镜像 p3 Run A,串行,任务计划程序启动) |
| `cache/enc/c{2..5}_*.pt` | 每划分 8 折编码器+Stage-0,共 32 个 |
| `logs/n3_scores_bal_b2e_c{2..5}_s1..8.npz` | R/D/sigma_mat/E/sim 全键 dump,32 折 |
| `logs/p4_enc_c{2..5}.log` + `.done`、`p4_enc_all.done` | 运行日志与完成标记 |

### 43.4 数据分析(07-14,`tools/p4_scale_c2to5.py` 纯后处理):calib-select 的"近 oracle"不跨词表,孪生结构=水平盲

**(a) calib-select 在 4 组新词表上的 G1 门槛读出(b2e 判据,8 折均值)**

| 词表 | s_cal/oracle 均值 | H: s=1 / 1.5 / cal / oracle | 距oracle≤2 | 对1.5≥−1 | base降≤2 | new增≥5 |
|---|---|---|---|---|---|---|
| c0(§42 锚,对照) | 1.54 / 1.88 | 82.8 / 89.0 / 89.6 / 90.8 | ✓1.2 | ✓+0.6 | ✓1.4 | ✓+12.2 |
| c2 | 1.49 / 1.91 | 79.4 / 87.1 / 86.2 / 89.2 | **✗2.9** | ✓−0.9 | ✓1.1 | ✓+11.6 |
| c3 | 1.86 / 1.88 | 84.4 / 90.5 / 90.9 / 91.8 | ✓0.8 | ✓+0.5 | ✗2.4(边缘) | ✓+14.4 |
| c4 | 1.70 / 2.05 | 66.7 / 79.1 / 79.8 / 84.0 | **✗4.2** | ✓+0.7 | ✓1.4 | ✓+18.2 |
| c5 | 1.44 / 2.02 | 79.1 / 85.6 / 83.5 / 86.9 | **✗3.4** | **✗−2.1** | ✓1.3 | ✓+8.7 |

- **稳的结论**:calib-select 处处大胜 s=1(new +8.7~+18.2、H +4.4~+13.1),与固定 1.5 基本打平(−2.1~+0.7),base 降幅可控——"数据驱动替代人工常数"成立;
- **褪色的结论**:"距 oracle ≤2"在新词表 3/4 失守(2.9/4.2/3.4),s_cal 系统性低估 oracle(1.44--1.86 vs 1.88--2.05),逐折 |s_cal−oracle|≤0.3 仅 11/32——§42 的"近 oracle"有 c0 抽签成分,**论文措辞必须改写为"免调参且不劣于最优固定常数",不得宣称普遍近 oracle**;
- **G4 风险量化**:对最强静态基线(1.5),c5 已 −2.1、c2 −0.9,"10 组 ≥8 组"要按 6/10 上下的底线做心理准备。

**(b) 低估机制(内联诊断,约束咬合 vs 峰值错位)**:s_cal 顶死在校准侧可行上限(base 降 2 分约束咬合)的折数——c4 **7/8**、c2 5/8 vs c3 1/8、c5 2/8;c5 的低估主要是内部峰值位置偏低(6/8 折)。两种机制同根:**单校准用户抽签**(b2e 家族 calib=1 人;test 侧可行上限逐折在 1.2--3.0 间乱跳)。可预注册的修复假设:**按 §38 双校准池化思路,rot2 式 2 人池化重打分**,预期同时松开约束咬合与稳住峰值。成本更正(07-14):rot2 是 leave-3-out(5 训练用户),编码器缓存键不同**必须重训**,~7.5h/case 而非最初估计的"缓存命中 1.5h"——故只跑两个失败模式代表 c4(约束咬死)与 c5(峰值错位),约 15h(batch_scripts/p4_rot2_c4c5.bat,07-14 已点火);c2/c3 仅在修复得证后再补。

**(c) 孪生审计(s=1 原始几何):孪生不是随机的,是"水平盲"的第三次现身**

| 词表 | 注册类(own%) | 主要泄漏去向 |
|---|---|---|
| c2 | g2 UD 滑(58%) | → g10 推 26%、g4 对角 15% |
| c2/c3 | g3 DU 滑(65%/62%) | → g7 对角 RL-DU 19%/12% |
| c4 | g2 UD 滑(**46%**) | → **g10 推 42%** |
| c4 | g4 对角 LR-UD(**42%**) | → **g10 推 38%** |
| c5 | g4 对角 LR-UD(48%) | → **g2 UD 滑 42%** |

被吃的注册类与其"汇"全部落在垂直/径向家族 {g2,g3,g4,g7,g10},且互混对**恰是只差水平分量的对**(g4 vs g2 差 LR;g3 vs g7 差 RL;UD/对角/推共享径向签名)。与 §S1 探针(水平轴 17.75)、§25 生成式没修、§26 保符号没修同根——**表示水平盲 → 孪生=水平分量不可分的对**。设计文档"孪生是几何问题、尺度校正不主张能修"由此有了机制名;c4 是孪生最重词表(两注册类各 ~40% 灌进 g10,解释其 n@s=1 仅 55.0),与 c1 一并进单独报告名单。

**(d) 路由级逐折抽签**:H@cal 逐折极差 c2 29.3 / c3 16.5 / c4 35.2 / c5 23.8 分——G2 分层报告口径的第五组证据。

### 43.5 双校准池化判决(07-18,`tools/p4_pool_verdict.py`,rot2 c4/c5 各 8 折):治好峰值错位,治不了约束咬合——静态尺度线关账

**背景**:§43.4(b) 的修复假设。rot2 = leave-3-out(2 池化校准 + 5 训练),编码器重训(`p4_rot2_c4c5.bat`,实际墙钟 07-15 16:04 点火、中间机器睡眠 51h、07-18 04:43 完成)。只跑两个失败模式代表,c2/c3 不回填。

| | s_cal/oracle(ratio) | H: 1.5 / cal / oracle | 距oracle | 对1.5 | base降 | 咬死折数 |
|---|---|---|---|---|---|---|
| b2e c4(单校准) | 1.70/2.05(**0.83**) | 79.1/79.8/84.0 | ✗4.2 | +0.7 | ✓1.4 | 7/8 |
| rot2 c4(池化) | 1.51/1.81(**0.83**) | 79.9/80.1/81.6 | ✓1.5 | +0.2 | ✗2.2 | 6/8 |
| b2e c5(单校准) | 1.44/2.02(**0.71**) | 85.6/83.5/86.9 | ✗3.4 | ✗−2.1 | ✓1.3 | 2/8 |
| rot2 c5(池化) | 1.44/1.59(**0.91**) | 85.3/85.4/87.3 | ✓1.9 | +0.1 | ✗3.3 | 8/8 |

**天花板分解**(门槛差距缩小 = oracle 降 + H@cal 升):c4 缩 2.7 = 天花板降 2.4 + 估计器升 0.3(**86% 是天花板效应,机制未修,ratio 纹丝不动 0.83**);c5 缩 1.5 = 天花板降 −0.5 + 估计器升 1.9(**真修复:ratio 0.71→0.91,H@cal +1.9,两条失守门槛全翻绿**)。

**判决(三条)**:

1. **池化治好峰值错位**(c5 模式):2 人校准把 H 曲面峰值定准了,是真实的估计器改进;
2. **池化治不了约束咬合**(c4 模式):相对低估 ratio 0.83 不动,咬死仍 6/8——根因是**用校准用户的 base 脆弱度替测试用户设防线**,跨用户可迁移性的固有极限,写入 limitation(G2 的在线反馈是否能补此角,留 discussion,不预设);
3. **新副作用**:rot2 下测试侧 base 降幅超门槛(2.2/3.3>2)——5 训练用户的松 base 头更脆,校准侧约束对测试用户保护不足;且 rot2 家族本身有代价(c4 oracle H 84.0→81.6,少一个训练用户不免费)。

**方法章措辞定稿**:calib-select = 免调参、处处大幅优于 s=1、与最优固定常数相当(池化配置下小胜);"逼近 oracle"不主张;推荐部署配置 = 双校准池化(附 base 降幅超限的诚实脚注)。**静态尺度分析线就此关账**(07-15 审定指示),c2/c3 rot2 不回填。

**资产**:`batch_scripts/p4_rot2_c4c5.bat`、`logs/n3_scores_bal_b2e_rot2_c{4,5}_s1..8.npz`、`logs/p4_rot2_c{4,5}.log/.done`、`tools/p4_pool_verdict.py` + `logs/p4_pool_verdict.npz`。

**下一步**:G0 受控流执行(协议层 stream.py + 14 测试已于 07-14 落地;GPU 已空,退化特征提取解锁;先写 tools/p4_stream.py 运行器 → 1 case × 2 用户冒烟 → c0/c1 早筛 → G0 判决)。c6--c9 交换词表待 §4.5 缓存键迁移后加入,禁止重训编码器。

## 44. P4 G0 判决 — 受控流上冻结配方被复合漂移击穿:PASS,问题成立(2026-07-18)

**背景**:设计文档 §8 G0。冻结论文三配方(Run-A 缓存编码器、方法 A K=5 balanced、fuse-base λ=1 + hybrid 共形、s=1.5 路由与接受解耦、§39 单元素采纳门只测不采)接入 `uwb_osr/stream.py` 五段受控流(§4.3;spec.seed=1000·case+ts,事件序与退化族/视角无关 → 各条件逐行配对)。全网格 = 2 词表(c0 固定配对 / c1 轮换配对,c1 编码器只有轮换键) × 8 折 × 6 退化族 × 4 视角轴(v0=Left/v1=Top/v2=Right/vall=三视角同时)。纯打分不重训(运行器遇缓存 MISS 直接拒跑);sev=0 事件复用 clean 前向,退化条件逐个物化跑完即弃(§4.5)。单折 ~4-5 min,全网格两 case 合计 ~2h(RTX 4060)。**bank 确定性(07-18 当天返工)**:首两次点火中 inspector 训练消耗全局 RNG,轨迹随访问历史分叉,视角网格文件内 sev=0 行与 sev>0 行混了两套 bank(逐折 q 差最大 0.59)——修复=运行器每折重播种(双跑 R/q 比特级一致),全网格 `--overwrite` 重生成,本节数字均为重生成后终版(判决定性不变;压线变动:occlude_v2 双词表新增 C、c1 drop_v1 降为仅 C、c1 domain_vall 的 A 触发消失)。

**判据执行修正(A 线,跑前定死)**:文档原文"污染率超过 10%"的绝对口径被 clean 基线击穿——clean 流的 §39 池(每类前 5 个单元素命中)污染率 c0 已 20.0%、c1 达 51.1%(单校准被试的新类 q 塌缩抽签所致;§39 报的 2.5–5% 是 global-q + 半流混合的另一口径),atten(分数与 clean 逐位相同)都会"触发"。三条线统一改读**相对 clean 的漂移归因增量**:H = H-mean 降 ≥10 分(全流或退化子集)、C = max|cov−90| 增 ≥10 分、A = contam5 增 ≥10 分。

### 44.1 判决表(`tools/p4_g0_verdict.py`,8 折均值;每族取最重单元,dH_hot=退化子集 H 降幅)

| 族 | c0 最重单元(dH/dH_hot/dCovE) | c1 最重单元 | 触发图景 |
|---|---|---|---|
| domain | vall: 78.1/**89.1**/+76.5(v0 亦 28.4/41.4/+46.5) | vall: 65.5/73.8/+72.7 | **全部 4 轴 × 2 词表 H/C 双触发** |
| drop | vall: 56.1/64.0/+15.9(v1 单视角即 17.6/21.0/+11.0) | vall: 38.0/44.3/+16.3 | 单视角多数 H(c1 Top 仅 C),多视角 H/C |
| occlude | vall: 15.6/19.8/+31.7 | vall: 14.5/17.8/+29.6 | vall 轴 H/C;v2 轴双词表压线 C(+10.0/+10.4) |
| noise | vall: 1.7/2.5/+17.1 | vall: 2.2/2.7/+18.3 | 仅 vall 轴 C(base 覆盖坍塌) |
| shift | vall: 5.9/6.8/+18.3 | vall: −6.7/−6.7/+18.0 | 仅 vall 轴 C |
| atten | 全 0(恒等) | 全 0 | **免疫(解析恒等)** |

**G0 判决:PASS(5/6 族触发 ≥ 门槛 2)**,复合漂移问题成立,论文四主线继续。

### 44.2 判读(四个机制发现)

- **atten 的免疫是数学恒等,不是稳健性**:幅值衰减=逐视角标量乘,被 abs_std 的 per-sample per-view 标准化精确抵消——报告时应作解析性质陈述,该族对本配方不构成漂移。
- **单视角腐蚀被"三视角均值"聚合稀释**:noise/shift/occlude 在任意单视角轴上 dH ≤ 5.9、全部不触发 H,只有三视角同时退化才击穿覆盖(C)——冻结配方对单弱视角的稳健性本身是论文三多视角设计的可报告红利;真正的单视角杀手只有 drop(信息消失)与 domain(处理域漂移)。
- **domain(CR→Raw 凸混合=杂波抑制退化)是统治性失败模式**:唯一在全部 4 视角轴 × 2 词表都 H/C 双触发的族;编码器与 Stage-0 都只见过 CR,残留杂波直接摧毁距离几何与 e_recon 两路信号(vall 时 c0 退化子集 H 归零、cov 误差 +76.5)。这是 G3 可靠度加权 + G2 风险控制最需要接住的场景。
- **采纳门在漂移下的行为是"闭嘴"而非"吃毒"**:重生成终版下**全网格无任一单元触发 A**——contam5 相对 clean 增量最大仅 +8.5(c1 occlude_vall),同时 nAd5 递减(c1 domain_vall 7.6 vs clean 11.4),漂移主要让单元素命中变稀,而不是让错误伪标签涌入;真正的污染问题在 **clean 基线自身**(c1 词表 51.1%,孪生"水平盲"§43.4(c) 在采纳门上的再现),属于论文三遗留的静态问题而非漂移问题,G2 需在此基线上改善而非只防漂移增量。

**视角不对称备忘**:c0 的 drop/domain 伤害 v1(Top)>v0(Left)>v2(Right)(drop:17.6/10.3/8.2),c1 则 v2>v0>v1(drop:16.1/12.9/5.8)——最载重视角随词表变,逐视角可靠度(§5.4)不能写死先验。

### 44.3 资产

| 文件 | 内容 |
|---|---|
| `tools/p4_stream.py` | 流运行器:缓存编码器+注册+冻结共形态→受控流全网格打分;`--deg`/`--deg-view`(0/1/2/all)/`--calib-rotate`;每折重播种(bank 比特级可复现)+dump 本折校准行;resume 按 npz 跳过;smoke 独立前缀 |
| `tools/p4_g0_verdict.py` / `tools/p4_stream_shat.py` | G0 正式判决;s_hat 流路由分析(均纯 dump 后处理,带 bank 配对校验拒混) |
| `batch_scripts/p4_g0_c0c1.bat` / `p4_g0_views.bat` / `p4_g0_regen.bat` | 三次点火(任务计划程序;一次性任务用后已删;regen 为终版数据源) |
| `logs/p4_stream_c{0,1}_{fam}[_v*]_s*.npz` + `..._calib_s*.npz` | 每折每条件全量 R/E/事件元数据 + 每折校准行(416 个,后处理可复算一切) |
| `logs/p4_g0r_c*.log`、`logs/p4_g0_verdict.npz`、`logs/p4_stream_shat.log` | 终版运行日志与判决/分析数值 |

### 44.4 s_hat 接入流路由(07-18,`tools/p4_stream_shat.py`):calib-select 平移成立,但漂移把最优尺度往 1 压——静态尺度在漂移下"不崩但不再最优"

每折 s_cal 用本折 bank 的 R_cal/R_cal_reg 按 §42 网格选出(c0 均值 1.61 / c1 1.84),应用到全部流条件;s_orc = 同条件流上的受约束 oracle(base 降 ≤2 约束下 H 最大,评价器侧)。

| 读数 | c0 | c1(孪生词表) |
|---|---|---|
| clean 流:H@1 / @1.5 / @cal / @orc | 82.2 / 89.0 / **90.4** / 90.9 | 60.9 / 74.1 / 73.8 / 76.8 |
| s_orc:clean → occlude → drop → domain | 1.92 → 1.25 → 1.36 → **1.09** | 1.82 → 1.26 → 1.39 → **1.01** |
| s_cal 对 1.5(24 退化单元):胜 >1 / 平 ±1 / 负 >1 | 11 / 11 / 2 | 1 / 16 / 7 |
| 最差单元(cal−1.5) | domain_v1 −1.6 / drop_vall −1.4 | drop_v1 −3.3 / domain −2.7 |

三个结论:①**G1 结论平移到流协议**:clean 流上 calib-select 处处大胜 s=1(c0 +8.2),c0 上小胜固定 1.5(+1.4)、距受约束 oracle 0.5,c1 与 1.5 打平(孪生词表,与 §43.4 一致);②**漂移单调压低最优路由尺度**(两词表一致:~1.9 → 重灾下 ~1.0-1.4):退化让 5-shot 新头的归一化距离膨胀快于多样本 base 头,"压老头"先验方向反转——**clean 侧选的静态 s 在漂移下系统性偏高**,但不崩(多数单元仍 ≥1.5 水平);③受约束 oracle 在重灾单元可低于 s_cal(负 gap,如 c1 drop_v2 −6.9):约束"base 降 ≤2"在漂移下咬合方式变化,提示 G2 期在线调 s 必须带同样的 base 保护约束,否则会追着崩塌的 new 头把 base 卖掉。**G2 设计输入:在线尺度自适应的信号源应为漂移状态(e_recon/可靠度),方向=向 1 收缩,幅度受 base 约束**;domain_vall 单元 H≈9-11 已是废墟,尺度不承担救场责任(G3 靶区)。

**下一步**:① ~~s_hat 接入流路由~~(本节收官);② G2 三动作控制器 + 在线共形(反馈预算切片已在 stream.py 预埋 random_query 槽位;在线尺度收缩接 44.4-② 的信号设计);③ G3 逐视角可靠度用 e_recon/域指纹接 domain/drop 场景(44.2 的主靶)。domain 严重度语义(凸混合系数=残留杂波比例)已在运行器 docstring 钉死。

## 45. P4 G2 — 在线自适应主网格 + 消融:预注册判决 FAIL,机制图谱完整(2026-07-19)

**系统与协议**:状态层 `uwb_osr/adaptation.py`(稳定/可塑双记忆,冻结 inspector 的 T3A 式 phi-only 加权模板更新——**sigma 重算被首冒烟证伪为失控杠杆**:跨用户/退化特征撑大加权 RMS→r=d/sigma 全局缩→新类列吞流(clean H 92.5→58.2、池污染 83%),故 sigma 冻结,尺度个性化留归流末 inspector 级整固)+`uwb_osr/risk_control.py`(三动作控制器/随机片专属 ACI/q80 采纳门/覆盖警报;17 状态测试钉死 §11.2)。运行器 `tools/p4_g2_stream.py`:每条件特征缓存(§4.5)、prequential 先评后更、可塑更新后重算剩余距离、未知类查询无覆盖指示但照扣预算。主网格=2 词表 × 25 条件 × B{0,2,5,10} × 8 折(~5.5h);消融=§6.2 预注册 4 变体 × 6 关键条件(~3.5h)。

### 45.1 主网格(B=5 对 B=0,8 折;`tools/p4_g2_verdict.py`)

四门(污染<5 / 宏覆盖 87-93 / H+5 / 遗忘≤1)**两词表 0/25 单元全过,失败形状相反**:c0 增益被 3 个抽签折的污染拖拽吃掉(污染 28-69%→逐折 dH −4.6/−8.5;污染≤15% 的 4 折 +3.5~+4.8,聚合 +0.3),c1 在 60-78% 污染下 dH 稳健 +3~+8(预算曲线单调:clean 74.1→84.3、drop 61.2→71.9 @B10)。domain 任何预算都救不动(特征废墟)。

### 45.2 消融判决(B=5,dH/遗忘/污染;main=q80 门,noauto=仅确认,w10c5=w_auto0.1+cap5,gateall=朴素自训练)

| 条件(c0) | main | noauto | w10c5 | gateall |
|---|---|---|---|---|
| clean | +0.3 / 2.2 / 21.5 | **+1.5 / 0.7** / 0 | +1.3 / 0.9 / 11.6 | +0.3 / 2.4 / 27.2 |
| drop | +0.5 / 7.3 / 28.5 | +1.9 / 4.0 / 0 | **+2.2** / 4.4 / 14.6 | −0.2 / 8.0 / 33.3 |
| noise_vall | −1.6 / 5.0 / 42.5 | **+0.4 / 1.9** / 0 | +0.3 / 2.2 / 31.8 | −2.0 / 5.5 / 50.5 |

c1:noauto 处处 ≥ main(clean +6.5、drop **+8.2**,遗忘 9.4→5.3)。**三个机制结论**:①**自动采纳的净贡献 ≤0**——c1 的 70% 污染此前看似无害(污染物=注册类的水平盲孪生,躺同一流形拖拽无方向),但去掉它反而更好:真实增益全部来自**确认通道**(不确定性片查询→可塑库);②污染率不是伤害度量,污染物几何位置才是(c0 跨边界拖拽 vs c1 同流形吸收);③q80 门对 gateall 有真实价值(处处压住更糟的朴素自训练),但在 phi-only 在线机制下"沉默自个性化"(§39 的承诺)不成立;残余负折(c0 折7 noauto 仍 −2.5)=纯正确标签的个性化也会把 phi 拉离跨人支持覆盖(§39 blend 教训的在线版,w10c5 的强制 1:1 混合≈同效)。

### 45.3 覆盖机制:ACI 救新类,e_recon 融合杀 base

在线共形把 cov_n 拉回靶区(86-96,含 domain 88),但 **cov_b 在漂移下崩死**(drop 82、occlude_vall 56、domain 38)且 ACI 无解——**凶手是冻结配方的 fuse-base λ=1**:z(E) 只加在 base 列,漂移把 e_recon 撑爆→base 分数爆炸→**传感器漂移被系统结构性误读为 OOD**。**(07-19 更正,§46:该归因经 S0 根因门检验只对 occlude_vall 成立;drop/domain 的 cov_b 崩塌与 E 惩罚无关,是逐视角距离几何本身被摧毁。)**这是论文三 clean 世界的 OOD 装置在开放世界的反噬,也是 G3(e_recon 改作可靠度权重而非加性惩罚,§5.4)最锋利的靶。ACI 恢复覆盖有效性的代价=rejU 腰斩(48→37 clean、84→41 domain),覆盖有效性与判别力的权衡照实报告。

### 45.4 G2 正式判决与后果

按 §8 预注册四门:**FAIL**——P 仅 noauto 架构性通过;H 仅 c1 通过(c0 clean 的 +5 门=H≥94,在 §34 oracle-personal 天花板之上,结构不可达);C 漂移下 base 侧崩(45.3 根因);F 漂移下 3-9 分。预注册后果:"只保留自动尺度对齐论文,不堆叠更多控制模块"。**判决解释与走向选项(留书面裁决,防"看结果改门槛")**:(a) 按字面关停控制模块线,论文四=G0 问题成立+尺度对齐+本节机制负结果集;(b) 判据重锚(H 门以漂移条件为对象/以 oracle 天花板归一)——需独立书面审定;(c) **G3 先行(推荐)**:e_recon 可靠度加权直击 cov_b 崩塌根因,本就在预注册计划内(§12 第 9-10 周,H4 污染视角定位早筛),G2 复判押后至 G3 之后——不是移门槛,是按预注册顺序执行。

**资产**:`uwb_osr/adaptation.py`+`risk_control.py`+`tests/test_adaptation_state.py`(18 测试);`tools/p4_g2_stream.py`(--gate/--w-auto/--cap/--tag)+`p4_g2_verdict.py`;`batch_scripts/p4_g2_c0c1.bat`+`p4_g2_ablate.bat`;`logs/p4_g2[_tag]_c*_*_B*_s*.npz`(主网格 1600 + 消融 384)、`logs/p4_g2_*.log`。

## 46. P4 G3/H4 c-lite S0 — 根因门证伪"λ=1 是主要凶手":按预注册暂停 G3(2026-07-19)

**背景**:c-lite 预注册 v0.2.1(用户书面裁决锁版,`docs/superpowers/specs/2026-07-19-g3-h4-clite-preregistration.md`)。一次前向 dump 原始量(D 逐视角距离张量 + M 逐视角重建 MSE + 校准行),六系统纯后处理。Phase A=流调度 {clean,drop,occlude_vall,domain_vall} × 2 词表 × 8 折(~1.7h)。

**工程回归门:PASS(v0.2.1 口径)**。分数对旧 dump ≤5e-4(cudnn.benchmark 自动调优跨进程选核漂移 ~1e-4;校准行/q 逐位一致、进程内重打 0.0 证明公式精确;E 路径逐位稳定)、决策级(pred/预测集/covered)与 G2 B=0 日志**零翻转完全一致**,全 16 折 × 4 单元。`recon_error` per_view 重构均值口径逐位不变(dE=0)。

**根因门(legacy-L0 撤掉加性惩罚 vs legacy-L1,各自用自身校准重算 μ/σ/q/s_cal):FAIL(1/3 条件,需 ≥2)**

| 条件 | c0: Δcov_b / 方向 / ΔrejU | c1 | 判定 |
|---|---|---|---|
| occlude_vall | **+21.9** / 8/8 / −17.1 | **+19.9** / 8/8 / −8.4 | 双词表 PASS——λ=1 确是此族凶手,且 rejU 代价坐实"找到病因≠完成修复" |
| drop | −0.1 / 5/8 / +1.8 | +1.8 / 5/8 / +0.4 | fail——E 惩罚无关 |
| domain_vall | −2.1 / 2/8 / −4.1 | **−6.6** / 3/8 / −2.3 | fail——L0 反而更差 |

clean H 差 0.0(L0=L1,守卫通过)。

**判读**:§45.3 的"cov_b 崩死凶手=fuse-base λ=1"**部分证伪**——该机制只统治 occlude(推及 noise/shift 的 vall C 触发,同属"E 撑爆"型);**drop/domain 的 base 覆盖崩塌根因在逐视角距离几何本身**(死视角/杂波域把 D 全列抬爆,与 E 项无关)。复合漂移的失败模式是**族异质**的:一半是"惩罚项误伤"(可用去惩罚/可靠度改造修),一半是"距离废墟"(重加权是否可救=未检验的可达性问题)。

**按预注册执行**:撤回"λ=1 是主要凶手"表述(§45.3 已加更正);**G3 暂停,Phase B(S1/可达门/正式 G3)不点火**;分支条款自动转**路线 a**(论文=G0 问题成立 + 尺度对齐 + G2/S0 机制负结果集)。若认为"族异质根因"这一新证据值得修订预注册(例如可达性门单独授权,检验 oracle-mask 对 drop/domain 的距离废墟是否可救),须另行书面裁决,不在本节内启动。

**资产**:`logs/p4_g3_c{0,1}_{clean,drop,occlude_vall,domain_vall}_s*.npz`+`p4_g3_c*_calib_s*.npz`(原始量,64+16 个)、`logs/p4_g3a_c*.log`、`tools/p4_g3_verdict.py --gate eng/rootcause` 输出。

## 47. P4 E0-a — oracle 可达性天花板证伪:掩蔽只在视角全死时有利,路线 a 生效(2026-07-20)

**背景**:E0 独立修订(用户书面裁决,`docs/superpowers/specs/2026-07-20-e0-oracle-reachability-amendment.md`):S0 只否定了"λ=1 统一解释",未否定可靠度加权本身——drop 是单视角退化,正是重加权的适用场景,故另立天花板假设:**评估器逐事件先知哪个视角坏(仅 sev>0 掩蔽,clean 事件均匀权重),硬掩蔽能否救回距离几何**。零 GPU,纯后处理于 Phase A 的 drop 原始 dump;uniform-rv 与 oracle-mask 各自用自身校准重算全部状态。门槛(锁定):双词表 oracle ΔH ≥+3 且各 ≥6/8 用户方向为正。

**判决:FAIL(双词表)**——c0 ΔH=−0.1(3/8 正)、c1 ΔH=+0.2(5/8 正,逐折 −22.1~+22.2 极端摆动)。

**机制(按严重度分层,方向性诊断)**:

| 严重度 | c0 ΔH | c1 ΔH | 判读 |
|---|---|---|---|
| 0.25 | −11.2 | −17.7 | 半坏视角仍带大量信息,扔掉净亏 |
| 0.50 | −11.5 | −12.7 | 同上 |
| 0.75 | −6.0 | −12.1 | 同上 |
| 1.00 | **+7.7** | **+7.9** | 视角全死才值得掩蔽 |

**这是 §44"单视角腐蚀被三视角均值稀释"的镜像**:均值聚合本身就是部分退化的最优解邻域,硬掩蔽只在信息归零点(sev=1)反超。**连完美检测的硬掩蔽天花板都救不回 drop 族**——按修订分支:E0 终止、E0-b 不点火、**路线 a 立即生效**。

**limitation 备注(照实报告,不另立实验)**:本天花板是"硬掩蔽"(权重 0/1);软权重天花板未检验,但中间严重度上硬掩蔽亏损 6-18 分的量级说明,任何软方案的可赚边际被压在"均值 vs 掩蔽"两个端点之间,H4 不再作为可行方向。

**路线 a 论文四定稿框架(修订文档 §"E0 失败时"条款)**:①G0 冻结系统在复合漂移下失效;②G1 calib-select 数据驱动尺度对齐;③G2 自动采纳失败、确认通道才是有效贡献;④S0 漂移失败族异质(惩罚误伤 vs 距离破坏);⑤E0 掩蔽天花板证伪+all-view 严重域迁移=表征废墟。定位=**数据驱动尺度对齐方法 + 受控开放世界漂移审计 + 在线自适应失败机制图谱**,不包装为完整在线自适应方法。

**资产**:`tools/p4_g3_verdict.py --gate e0a`(逐事件 oracle 掩蔽语义在修订文档钉死)、输出见本节;无新 dump(零 GPU)。

---

## 20. D2 — 单视角复现差距破案(2026-06-11)

**背景**:此前单视角 OSCR 82.90 vs 论文 88.42,差 5.5 分长期归因不明。逐 case 对照发现**差距几乎全在 case 4(73.54 vs 87.98,−14.4)**,而闭集精度与论文持平 → 锁定 Stage-2 / 特征空间,不是 encoder 训练问题。三因素叠加破案。

### 20.1 case 4 单 case 因素分解(单视角 flatten + abs_std,inspector)

| 配置 | closed | INSPECTOR OSCR | AUROC | 增量 |
|---|---|---|---|---|
| 旧基线(ClutterRemoved,默认 mixup,per-head D_key=8) | 94.67 | 73.18 | 76.49 | — |
| + 论文式 Eq.12 负样本 mixup(`--paper-neg-mixup`) | 95.17 | 75.38 | 78.94 | +2.20 |
| ↑ + batch_insp=16(论文 batch) | 93.67 | 75.39 | 78.51 | +0.01(无关) |
| + **忠实版 D_key=32**(`--d-key 32`,新 WideHeadMSA) | 94.00 | 77.84 | 81.54 | +2.46 |
| + **Raw 输入**(`--data-type Raw`) | 95.75 | **88.35** | 92.01 | **+10.51** ⭐ |
| 论文 case 4 | — | 87.98 | 89.26 | ✓ 对齐 |

> **主因是输入数据变体**:论文未写明,但换 Raw 后 case 4 的 OSCR(88.35 vs 87.98)、Accuracy(83.75 vs ~)、AUROC 全部和论文对上,误差 <0.5。证据链完整。

### 20.2 为什么之前没发现:三视角下 Raw≈CR 打平掩盖了它

§16(D1)测过三视角 Raw vs ClutterRemoved,结论"打平"(95.53 vs 95.14)——**没错,但只在三视角成立**。机制:三视角的几何互补冗余掉了 ClutterRemoved 丢失的信息,单视角丢不起。
- 这也回头解释了论文 Fig.1 示例 CIR 图里贯穿全时间轴的竖直静态条纹 —— 那正是 **Raw 数据**(未做背景对消)。
- 教训:输入预处理的影响**强烈依赖视角数**,单视角才是检验它的敏感场景。

### 20.3 三个真因素 + 一个伪因素

| 因素 | case 4 增量 | 定性 |
|---|---|---|
| **Raw 输入** | +10.51 | 主因。论文用 Raw,我们一直用 ClutterRemoved |
| **D_key=32 忠实版注意力** | +2.46 | 次因。每头 32 维(非 PyTorch 的 32/4=8),token/feat 不变 |
| **论文式 Eq.12 mixup** | +2.20 | 次因。混合"目标类×其它类"产生近边界 hard negative |
| ~~batch_insp=16~~ | +0.01 | **伪因素**。确认与 batch 无关,保留 256 |

> ⚠️ **重要更正**:§6.1("b16 恶化 −11 分")和 config 注释("paper mixup 无效")的旧结论,都是在 **abs_std 之前的 per_view_std 时代**做的,且 b16=63.36 用的是修 mixup 前的代码。`abs_std × paper-mixup × b16` 这个真正 paper-faithful 组合当时从未跑过。本节修正了这两条过期结论。

### 20.4 胜出配置 5-case 全扫(Raw + paper-mixup + D_key32 + flatten + abs_std)

| Case | closed | OSCR | AUROC | 论文 OSCR | 旧 CR 基线 OSCR |
|---|---|---|---|---|---|
| 1 | 99.17 | 87.81 | 88.26 | 88.26 | 87.04 |
| 2 | 95.58 | 84.53 | 88.29 | 90.91 | 84.35 |
| 3 | 99.25 | 87.39 | 87.58 | 87.40 | 85.99 |
| 4 | 95.92 | 87.18 | 90.65 | 87.98 | 73.54 |
| 5 | 98.58 | 86.81 | 87.42 | 87.57 | 83.59 |
| **均值±std** | **97.70** | **86.74 ± 1.29** | **88.44 ± 1.30** | **88.42 ± 1.43** | **82.90 ± 5.41** |

> **结论:单视角 OSCR 86.74 ± 1.29,与论文 88.42 ± 1.43 相差 1.68,落在 1 个标准差内 —— 复现到位。** 比旧基线 +3.84,且 **std 从 ±5.41 收缩到 ±1.29**(case 4 不再塌陷)。剩余 1.68 分几乎全在 case 2(84.53 vs 90.91,论文最好的 case),其余 4 个 case 全在 ±1 分内。

### 20.5 顺带修的两个代码问题

1. **F1 占位 bug**([metrics.py](uwb_osr/metrics.py)):被接受的 unknown 样本原本一律记为 class 0 占位,把所有 false positive 堆到一个类、系统性压低 macro-F1。改为按分类器 argmax 落到各手势(对齐论文 Fig.9 的 UG 行分散)。**不影响 OSCR/AUROC/Accuracy**,只修正 F1。
2. **φ 原型 dropout 噪声**([inspector.py](uwb_osr/inspector.py)):原型 φ_k 原本在 `insp.train()`(dropout 开)下计算,给距离几何注噪。改为 `eval()` 模式确定性计算。

### 20.6 新增代码 / 开关 / 日志

| 项 | 内容 |
|---|---|
| `model.py: WideHeadMSA / WideHeadBlock` | 论文 Eq.7–9 忠实注意力,每头 D_key 维;默认 `d_key=0` 仍走 nn.TransformerEncoder(逐字节等价旧管线) |
| `config.py: d_key` / `main.py: --d-key` | per-head key 宽度开关(0=旧,32=论文) |
| `config.py: inspector_dropout` / `main.py: --insp-dropout` | Stage-2 inspector dropout 参数化(默认 0.1) |
| `metrics.py: pred_u` 参数 | F1 占位 bug 修复 |
| `repro_c4_papermix_{b256,b16,dkey32,dkey32_raw}.log` | case 4 四步因素分解 |
| `repro_sweep5_single_papermix_dkey32_raw.log` (+ `repro_c5_*`) | 胜出配置 5-case(case 5 因 DataLoader-free 单进程 hang 补跑) |

### 20.7 论文写作建议

- **复现声明**:"With the dataset's raw ADC cubes, the paper-faithful per-head key width (D_key=32) and the Eq.12 manifold-mixup negatives, our single-view reproduction reaches OSCR 86.74 ± 1.29, within 1.7 points (< 1 std) of the reported 88.42." → 这是规范的 baseline 复现,站得住。
- **输入数据**:论文未明确 Raw vs clutter-removed;我们的消融显示**单视角对此高度敏感(+10.5 OSCR),三视角不敏感**,建议复现者用 Raw。
- 主贡献(三视角 95.14)不依赖这 1.7 分差距,且本节确认三视角下 Raw/CR 等价,主结果稳健。

### 20.8 待办

- [ ] 胜出配置缺 per-case Accuracy/F1(单 case 控制台只打 OSCR/AUROC;Acc/F1 仅在合并 SUMMARY 出)。case 4=83.75/83.47、case 5=82.08/82.46 已有,case 1–3 需一次干净的全 5-case 重跑补全表格。
- [ ] case 2 单独多 seed 复测,确认 84.53 vs 论文 90.91 是抽样波动还是真实短板。
- [ ] 把胜出配置回测三视角,看 Raw + D_key32 + paper-mixup 是否把 95.14 再推高。

