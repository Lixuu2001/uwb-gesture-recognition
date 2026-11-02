# 多视角融合架构设计研究(MV-OSR Fusion Design)

> 回答的问题:三个雷达视角(Left/Top/Right)用什么方式融合最好?
> "直接合起来放进 Transformer" 是不是太简单、缺少视角间信息?
>
> 结论先行:**你的担心方向是对的,但病灶不在"融合模块太简单",而在三件更具体的事**
> (§1)。对应的药方不是"换一个更花哨的注意力",而是:①把融合放到 OSR 真正消费的
> 特征上;②用跨视角重建这类目标函数去"制造"视角间信息;③在架构上保留逐视角
> 独立读出,因为开集识别的增益恰恰依赖视角间的多样性。推荐架构见 §3,
> 实验阶梯见 §5。

---

## 1. 现状诊断:视角间信息现在到底在哪里流动

当前管线(`model.MultiViewOSRNet`)的数据流:

```
X (B,3,90,189)
 ├─ SharedDualPathEncoder(Siamese, 权重共享)
 │    ├─ f_views (B,3,8928)  ←—— inspector(主结果 OSR 头)吃的是这个
 │    └─ t_tok (B,3,90,32), s_tok (B,3,189,32)
 ├─ CrossViewFusion(t_tok)  ←—— 只作用于时间路径 token
 │    └─ c_fused → ClosedSetHead → logits   ←—— 融合只喂闭集头
 └─ f_views → per-view EvidentialHead → DS 融合(副线,已证弱于 inspector)
```

视角间信息目前只通过 5 条通道流动,逐条审计:

| # | 通道 | 强度 | 审计结论 |
|---|---|---|---|
| 1 | 编码器权重共享(Siamese) | 隐式 | 只共享"如何看",不交换"看到了什么" |
| 2 | `view_emb`(加在 pooled f 上) | 弱 | 只给特征打视角标签,不产生交互 |
| 3 | `CrossViewFusion` → 闭集头 | **错位** | 见下方 (a) |
| 4 | per-view inspector 距离平均 | **强,是主增益** | E4 已证:增益 = 几何互补 + 距离去噪 |
| 5 | 评测时三视角共享同一个 pred k | 中 | 视角身份分歧会推高少数视角的 d_v,部分捕捉"视角冲突" |

三个关键事实(也是回答你问题的核心):

**(a) 主结果根本没经过融合模块。** inspector 消费的 `f_views` 在
`CrossViewFusion` *之前* 取出;融合输出只进闭集头,而闭集已饱和(三视角 97.8%)。
`CrossViewFusion` 的参数只接收闭集 CE 的梯度。也就是说,**95.90 OSCR 的全部多视角
增益来自通道 1+4(共享编码器 + late fusion 距离平均),你现在的"跨视角注意力"
对主指标的贡献从未被测量过**——logs/ 里没有任何 `--no-fusion` 的 5-case 运行记录
(README 里的 B1 实验实际从未跑过)。

**(b) 闭集 CE 教不出视角间几何。** 三视角是同一个 3D 手部轨迹
p(t) 的三个径向投影 r_v(t)=‖p(t)−q_v‖,真正的"视角间信息"=三边定位式的联合结构。
但闭集任务 97.8% 已饱和,encoder 没有任何梯度压力去学这层几何。
E5(物理 bias 零增益)和 E4(var 分量冗余)都是这个根因的症状:
**不是先验没用,是目标函数不需要它**。要制造视角间信息,得换目标
(跨视角重建/对比/蒸馏),仅加注意力层没用。B2 的 case-0 早筛
(recon AUROC 95.48,standalone OSCR 95.01)正面验证了这一点。

**(c) 融合深度与 OSR 增益存在结构性张力。** 你的 OSR 增益来源是三个
"半独立观察者"的距离平均(E4 结论)。融合做得越深,三路特征越相关,
ensemble 去噪越弱;ACM MM'25 的多视角开集工作(arXiv:2508.01227)正式论证了
"直接融合多视角会放大单视角的 bias、**恶化**未知类识别"。
你自己的 E2 也提供了旁证:朴素三视角训练的特征缺一路就塌(43–55 OSCR),
说明 CE+共享编码器已经学出了某种脆弱的跨视角耦合。
**所以"更深的融合"不保证更高 OSCR——任何融合设计必须保留逐视角读出。**

顺带三个现有 `CrossViewFusion` 的具体技术缺陷(如果保留它,值得修):

1. **视角置换对称**:`view_emb` 加在 pooled `f` 上,`t_tok` 进融合时**没有视角身份**,
   MHA 在 V 轴上是置换等变的——模块学不出"Left 该如何看 Right"这种非对称关系
   (physical bias 想注入的正是这个,但加性 bias + 闭集 CE 学不动)。
2. **只融合时间路径**,`s_tok`(空间/距离路径)完全不交换信息;而距离轴恰恰是
   三视角几何关系所在的轴。
3. **单层 MHA、无 FFN、一次性后置**:没有"交换→消化→再交换"的迭代;
   只能同时刻互看(虽然 encoder 先做过时间混合,有间接跨时通路,但只有一轮)。

---

## 2. 设计空间:四个融合层级

| 层级 | 代表方法 | 对本项目的适配度 |
|---|---|---|
| **输入级(early)** | 通道堆叠 / range 轴拼接;本数据集上已有 CNN 多雷达闭集工作(98%+) | 闭集够用,OSR 无证据;缺视角最脆弱;不推荐为主线 |
| **编码器级(mid)** | ① token 全拼接联合注意力("直接合起来");② 时间对齐分解注意力(现状);③ **瓶颈 token(MBT,NeurIPS'21)**;④ 成对 co-attention(ViLBERT 式) | ③ 是本方案推荐;① 应跑成 baseline(§4.2);④ 与 ③ 等效但无信息瓶颈正则、参数更多 |
| **目标函数级** | 掩码视角重建(=你的 B2)、跨视角对比(supcon 开关已有)、跨视角蒸馏(KD 开关已有) | **对 OSR 期望值最高**:已知手势三视角自洽、未知不自洽,重建误差是单视角原理上造不出的 OOD 信号,且 case-0 已验证 |
| **决策级(late)** | per-view inspector 距离平均(现状最优)、DS/TMC(已试,弱)、不确定度加权 | 保留距离平均为骨架;退化视角加权留作 C3' 主线 |

**MBT 的两个结论直接适用**(Nagrani et al., NeurIPS 2021, arXiv:2107.00135):
(i) mid-fusion 优于 early 和 late——前几层让各模态先学好单模态特征,后几层再交换;
(ii) 把跨模态信息流限制在少量 bottleneck token 里,效果优于全 pairwise attention
且计算更省——**窄带交换本身是正则**,这对你每 case 仅 3600 训练样本、
且 E3 已证"容量↑→OSCR↓"的小数据场景尤其重要。

**多视角 OSR 的特殊性**(MM'25, arXiv:2508.01227 + 你的 E4):
闭集任务里融合越深越好;开集任务里,未知样本的判据一半来自"每个视角内离已知流形
的距离"(要求视角特征独立、干净),一半来自"视角间联合结构被违反"
(要求有联合表征)。**架构解 = 融合后仍保留逐视角读出;决策解 = ensemble 里
同时放 per-view 成员和 joint 成员。**这是 §3 设计的出发点。

---

## 3. 推荐架构:瓶颈 token 中层融合(X-MV Encoder)

### 3.1 结构

```
X (B,V,T,M)
  per-view 线性投影 + 位置编码 + 【视角嵌入加在 token 上】(修复置换对称)
  ┌──────────── 前 L−K 层:各视角独立 dual-path encoder(完全同现状)─────────────┐
  └──────────── 后 K=2 层:BottleneckFusionLayer ───────────────────────────────┘
       每层、每路径(temporal 和 spatial 都做):
         view v 的 token 序列 ← self-attn( [tok_v ; btl] )      ← 视角内 + 读写瓶颈
         btl ← mean_v( btl_v' )                                  ← 唯一的跨视角通道
       NB=4 个共享可学习瓶颈 token,初始 (1,NB,D)
  读出:
    f̃_v   = flatten(view v 的 fused tokens)(维度不变,仍 8928)→ per-view inspector
    z_btl = flatten(btl)(NB·D=128)        → 闭集头辅助输入 + 可选 joint inspector
    c_fused = mean_v(f̃_v) ⊕ z_btl          → ClosedSetHead
```

### 3.2 为什么是它(对照你的三个病灶)

- **病灶 (a)**:f̃_v 是融合后的逐视角特征,inspector 直接升级为消费跨视角信息;
  同时保留逐视角结构,距离平均 ensemble 不被破坏。
- **病灶 (b)**:瓶颈 token 是 B2 掩码重建的天然载体——Stage-0 重建被遮蔽视角时
  decoder 查询 [可见视角 tokens; btl],重建梯度逼 btl 编码"潜在 3D 运动状态"。
  **bottleneck ≈ 软三边定位的潜变量**,这是论文里很漂亮的物理隐喻:
  三个雷达不直接互看,而是共同读写一个"运动假设"。
- **病灶 (c)**:跨视角带宽被限制在 NB·D=128 维,视角特征的独立性大部分保留;
  对比之下 token 全拼接是 837×837 的无约束互看。
- **缺视角兼容**:任意视角子集 forward 时,循环里少一路即可,btl 照常工作;
  与 view dropout(R1,已验证强阳性)和 KD 完全兼容,无需新机制。
- **算力**:序列长从 90/189 变成 94/193,增量可忽略;远便宜于 token 拼接的
  (3·(90+189))² 注意力。dim=32 下新增参数 ~几万,远离 E3 的过拟合区。

### 3.3 参考实现(伪代码,贴近现有代码风格)

```python
class BottleneckFusionLayer(nn.Module):
    """MBT-style: per view, tokens attend to [own tokens ; shared bottleneck];
    updated bottleneck copies are averaged across views (the only x-view path)."""
    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)

    def forward(self, toks, btl):          # toks: (B,V,L,D)  btl: (B,NB,D)
        B, V, L, D = toks.shape
        outs, btls = [], []
        for v in range(V):                 # V<=3, 循环开销可忽略;缺视角天然兼容
            seq = self.layer(torch.cat([toks[:, v], btl], 1))
            outs.append(seq[:, :L])
            btls.append(seq[:, L:])
        return torch.stack(outs, 1), torch.stack(btls, 1).mean(1)
```

`TransformerPath` 改造:`depth` 拆成 `depth - K` 层共享独立编码 + `K` 层
`BottleneckFusionLayer`;temporal/spatial 两条路径各自持有(或共享)一组 btl。
token 进入第一层前加 `view_emb_tok[v]`(新参数,(V,1,D) 广播)。
`f_views` 的 flatten 维度不变(8928),inspector/缓存管线零改动。

### 3.4 超参与早筛

- NB ∈ {2, 4, 8},K ∈ {1, 2};先 case-0 早筛(你的标准流程),
  判据:closed ≥ 99 且 inspector OSCR ≥ 当前 case-0 锚点(96.2)。
- 风险预案:若 fused f̃_v 喂 inspector 反而掉分(= 病灶 (c) 应验),
  退路是 **双 bank**:pre-fusion f_views 的 inspector + btl 的 joint inspector
  并入同一个距离平均(把 joint 当"第 4 个视角"),保住独立性又补上联合信号。

---

## 4. 备选与对照方案

### 4.1 最小修改版(不动编码器,1–2 天)

如果暂时不想上瓶颈结构,现有 `CrossViewFusion` 改四点即可显著"去玩具化":

1. 输出 per-view fused tokens,readout 成 f̃_v 喂 inspector(并联旧 f_views 对照);
2. token 进融合前加视角嵌入(修置换对称);
3. `s_tok` 路径也融合(距离轴才是几何轴);
4. 升到 2 层、补 FFN(直接用 `TransformerEncoderLayer` 代替裸 MHA)。

这本质上是"时间对齐的 co-attention",表达力弱于 §3 但改动极小。

### 4.2 "直接合起来"baseline(必跑,1 天)

把 3×(90+189) 个 token 加视角嵌入后拼成一个序列,过同一个 dual-path(或单流)
Transformer。**预期它不赢**:小数据 + 837 token 全注意力 = 容量过剩
(E3 的教训),且缺视角时序列结构变化最大、退化最狠。
但论文需要这一行 ablation——审稿人必问"为什么不直接 joint attention",
MBT 给了文献答案,你要给数据集上的实证答案。

### 4.3 物理赌注:三边定位轨迹流(先验筛,1 天,不训练)

对每帧做 soft-argmax 测距 r̂_v(t) → 已知雷达坐标闭式三边定位 → 轨迹 p̂(t) 与
**定位残差** e_tri(t)。先用纯 numpy 在 case-0 上测:已知 vs 未知的 e_tri 分布
AUROC > 70 才值得做成可微模块(第 4 输入流 / OSR 评分分量)。
预期风险:手+臂是扩展目标,峰值测距噪声大;ClutterRemoved 下也未必干净。
BEV/LSS 式的特征级几何 lifting(逐 voxel 聚合三视角距离 bin 特征)列为远期,
当前数据规模与饱和闭集下性价比不高。

---

## 5. 实验阶梯(带 go/no-go,按性价比排序)

| # | 实验 | 改动量 | 成本 | go/no-go 判据 |
|---|---|---|---|---|
| **F0** | `--no-fusion` 5-case,量化现有融合贡献(补上从未跑过的 B1) | 0 行 | 半天 GPU | 无论结果都有用:≈0 → 重设计的动机 + 论文改叙事;>0 → 保留现模块 |
| **F1** | fused per-view 特征喂 inspector(§4.1 第 1 条,并联对照) | ~30 行 | 半天 | insp(f̃) − insp(f) ≥ +0.5 OSCR 才继续加深融合 |
| **F2** | 瓶颈融合 X-MV(§3),case-0 早筛 → 5-case + E2 矩阵 | ~80 行 | 1 周含调参 | 5-case ≥ 95.9+0.5 **且** E2 缺视角矩阵不回退 >2 分 |
| **F3** | B2 完成 5-case + λ_recon ∈ {0.3,1,3} 评分融合(§18 已排队) | 已有代码 | 数天 | mean_only + λ·e_recon > 95.90;与 F2 正交,可并行 |
| **F4** | 三边定位残差 numpy 早筛(§4.3) | ~60 行纯 numpy | 1 天 | AUROC > 70 才做可微版 |
| **F5** | token 拼接 joint baseline(§4.2) | ~50 行 | 1 天 | 论文 ablation 行,预期为负/平结果 |
| **F6** | 用 LOSO 重评 F0–F3 的最优配置 | 已有 `--loso` | 数天 | **融合的真正差异化战场**,见下 |

**关于 F6 的提醒**:随机切分协议有 session 泄漏,95+ 区间内各种融合方案可能
全部"打平"——表征质量的差异在域偏移下才显形。跨视角几何是被试无关的物理结构,
**好的融合应该在 LOSO 下拉开差距**;这也是论文叙事上"为什么要学视角间几何"
最有力的实验位置。

**开闭原则**(沿用你的习惯):每一步默认关闭、开关可逐字节回退;
开集卫生不变(融合/预训练只见 known-train)。

---

## 6. 对论文叙事的影响

1. **C1 升级**:从"加了两个雷达 + 一层注意力"(审稿人会嫌 incremental)升级为
   "面向开集的多视角融合架构设计:瓶颈交换 + 逐视角读出 + 跨视角自监督",
   并与 MM'25 的"深融合恶化开集"论点正面对话——你的 E4/E2 是独立证据。
2. **负结果链条更完整**:E5(物理先验)→ E4(分歧分量)→ F0(后置注意力)→
   F5(naive 拼接),系统排除 naive 路线后落到 F2/F3,故事是"设计是被证据逼出来的"。
3. **B2 的位置提前**:它不只是"未来工作的自监督预训练",它就是
   "目标函数级融合"、是制造视角间信息的主力,且 e_recon 是单视角原理上
   造不出的 OOD 信号——建议进正文方法章,不留在 future work。
4. **瓶颈 token 的物理隐喻**:btl ≈ 潜在 3D 运动状态,三视角通过它通信
   = 可学习的软三边定位。这句话值一张示意图。

---

## 7. 执行记录(2026-06-11 起)

**新增开关**(全部默认关闭,关闭时与旧管线逐字节一致):

| 开关 | 配置项 | 作用 |
|---|---|---|
| `--inspector-feats {pre,fused}` | `cfg.inspector_feats` | F1:inspector 消费融合前(pre,旧管线)或融合后(fused)的逐视角特征码,维度不变 |
| `--xmv-layers K` | `cfg.xmv_layers` | F2:每条路径最后 K 个 block 换成瓶颈交换层(总深度不变,容量持平) |
| `--xmv-bottleneck NB` | `cfg.xmv_bottleneck` | F2:共享瓶颈 token 数(默认 4) |
| `--device {cuda,cpu}` | `cfg.device` | 强制设备(GPU 忙时 CPU 跑冒烟) |

实现位置:`model.py: BottleneckFusionLayer / TransformerPath(xmv_*) /
MultiViewOSRNet.forward(f_views_fused)`;`inspector.py: extract(fkey)`。
附带修复:`pretrain.py _target_for` 现在接收被遮蔽视角的槽位(token-target
模式在 xmv 下取对 `view_emb_tok` 行;raw 默认目标不受影响)。

冒烟验证(合成数据,CPU):基线回归 / F1 / F2 / F2×R1×E2 七子集 / F2×B2
Stage-0 加载,5 项全过(`logs/smoke_f*.log`)。

**已知预期(读数时注意)**:F1/F2 的特征都依赖在场视角集合,E2 不重训直接切
子集会有 train/test 失配,缺视角矩阵可能比 pre-fusion 特征差——这正是
§3.2 说的"融合 × 独立性"张力的体现;补救是叠加 `--view-dropout 0.5`(R1),
让融合层在训练期就见过子集。5-case 主对照不受影响(全视角)。

**运行队列**:

| 运行 | 命令要点 | 状态 |
|---|---|---|
| F0 no-fusion 5-case | `--no-fusion --pool flatten --normalize abs_std --ablate-score` | 🏃 `logs/f0_sweep5_three_flat_absstd_nofusion.log` |
| F1 case-0 早筛 | F0 完成后:`--case 0 --pool flatten --normalize abs_std --inspector-feats fused --ablate-score` | 排队 |
| F2 case-0 早筛 | `--case 0 --pool flatten --normalize abs_std --xmv-layers 2 --no-fusion --ablate-score` | 排队 |

对照锚点:三视角 flatten+abs_std combined **95.14±0.81** / mean_only
**95.90±0.73**(5-case);case-0 combined ≈ **96.1–96.2**
(`p0c0_flatten.log` 96.13,§3 表 96.19)。

---

## 8. 参考文献(写作时可直接引)

- Nagrani et al., *Attention Bottlenecks for Multimodal Fusion*, NeurIPS 2021.
  arXiv:2107.00135 — 瓶颈 token、mid-fusion > early/late 的出处。
- *Enhancing Multi-view Open-set Learning via Ambiguity Uncertainty Calibration
  and View-wise Debiasing*, ACM MM 2025. arXiv:2508.01227 — "直接融合多视角
  恶化未知类识别"的正式论证,与本项目 E4/E2 互证。
- Han et al., *Trusted Multi-View Classification*, ICLR 2021 — 你已实现的
  evidential + DS 路线的出处(作为对照保留)。
- Eom et al., IEEE IoTJ 2025 — 单视角基线;其脚注 1 明确预言多天线空间信息
  会大幅提升,是 C1 的直接动机。
- Ahmed et al., *UWB-Gestures…*, Scientific Data 2021 — 数据集;
  另有 Khan et al., IEEE Access 2023(时间-距离参数 + 多雷达)等多雷达
  **闭集**工作(98%+),可用于论证"闭集已饱和,开集才是多视角的增量战场"。
- Lopez-Paz et al., ICLR 2016;Gupta et al., CVPR 2016 — 跨视角蒸馏(KD 开关)。
- (远期)Philion & Fidler, *Lift-Splat-Shoot*, ECCV 2020;Li et al.,
  *BEVFormer*, ECCV 2022 — 特征级几何 lifting 的参照系。
