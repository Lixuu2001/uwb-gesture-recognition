# Current continuation gate — compact and speed-robust UWB prior art (2026-10-09)

Published UWB-Gestures work already covers a lightweight UWB model, same-corpus early fusion/CNN-LSTM/attention and synthetic-noise evaluation, and speed-induced multi-scale features. Do not launch generic compact-CNN, fusion, noise augmentation, or speed/multi-scale grids as the requested new method. The referenced papers' protocols are not established as comparable to the project's eight held-user folds; don't rank their headline scores against local results. No candidate clears novelty and execution gates. Full bounded source notes: `results/cloud_20261009_compact_speed_prior_art/RUN_REPORT.md`.

---

# Current continuation gate — multi-view synchronization evidence (2026-10-09)

The descriptor confirms independent monostatic X4 views and 90 slow-time rows per gesture, but does not state a common trigger/timestamp or synchronization procedure. The omission does not prove asynchronous recording. MV-GLFNet assumes aligned TR dimensions and acknowledges view alignment and possible sync/calibration deployment costs. Thus index-aligned tensors are insufficient evidence for coherent cross-radar phase/delay fusion or calibrated 3-D trajectory claims. Same-corpus view-subset fusion is preempted; generic radar-camera DTW alignment is adjacent prior art. Do not run a generic alignment/fusion grid.

No labels/arrays were read, no training started, no active checkpoint writer found; restored data receipt is verified. Any candidate needs a distinct mechanism, defensible timing or motion truth, prior-art review, and frozen participant-disjoint protocol. No current candidate clears gates. Report: `uwb_direction1/results/cloud_20261009_multiview_sync_review/RUN_REPORT.md`.

---

# Current continuation gate — Pantomime rights/semantics follow-up (2026-10-09)

The Aalto catalog provides positive CC-BY-4.0 metadata for the dataset and links to Zenodo 4459969, but file-level license association and archive manifest are unverified; do not infer rights for source_files.zip. Public class names establish only a possible swipe/circular semantic overlap, not kinematic equivalence. Broad heterogeneous radar gesture transfer is prior art, and exact-endpoint search was bounded.

No candidate clears the novelty and execution gates. Zenodo is not allowlisted for this CPU-only executor. No archives/data were downloaded, no labels/arrays were read, no training started; the restored UWB corpus and 789-file snapshot are present. Do not ask for a manual upload or rerun old negative grids. Pantomime stays blocked until a distinct sensor-physics question, independent movement-level mapping, archive manifest/rights evidence, prior-art review, and frozen participant-disjoint protocol are all clear. See `uwb_direction1/results/cloud_20261009_pantomime_rights_semantics/RUN_REPORT.md`.

---

# Current continuation gate — 2026-10-09 (open-set prior art)

A new primary-source check closes generic UWB open-set gesture recognition as a standalone contribution on the existing cohort. Eom et al. 2025 already publish UWB impulse-radar open-set recognition. Yang et al. 2026 reports open-set experiments on this exact UWB-Gestures corpus, five held-out class splits, and cross-subject analysis. Do not train a generic unknown-class rejection, threshold, OVR inspector, or prototype variant and call it novel. The article protocol has not been confirmed equivalent to this project's eight held-user folds; scores are not directly comparable. Review: `results/cloud_20261009_open_set_prior_art/RUN_REPORT.md`.

No UWB labels/arrays were read and no training started. Current data and source snapshot are present, but the current executor is CPU-only, has restricted Zenodo egress, and has no capture hardware or synchronized external truth. Cross-dataset Pantomime transfer still has semantic/archive-license gates. No candidate currently clears both novelty and execution gates; continue only after a materially distinct, source-reviewed question is specified or new synchronized measurements become available.

---

# Current continuation gate — 2026-10-09T12:11Z

A fresh full-text check closes generic sensor-failure and view-dropout routes too:
- Xue et al. 2026 uses this exact X4 UWB-Gestures dataset for single-, dual-, and tri-view results.
- Zhao et al. 2023 evaluates incomplete multi-FMCW-radar gesture samples after one radar fails.
- Yang et al. 2018 studies angular-diversity fusion when one CW-radar view is poor.
- ModDrop already trains gesture models with randomly missing channels.
See `results/cloud_20261009_multiview_dropout_prior_art/RUN_REPORT.md`.

Do not run another generic view-subset, dropout, angular-diversity, or fusion experiment as the requested original method. No current candidate clears both novelty and execution gates. The remaining possible cross-dataset X4→Pantomime endpoint is still gated by a substantive question, independent semantic justification, archive/license association, and frozen participant-disjoint protocol. Physical truth-oriented work needs a new synchronized cohort plus capture hardware absent from this executor. Current process check found no active run; CPU-only, shell egress restricted, UWB data and full source snapshot already present. No labels/arrays were opened in this review.

The historical candidate plans below do not override these latest gates.

---

# Current continuation gate — 2026-10-09

The sections below are historical candidate plans and do not override the current branch status. Several originally listed routes have already been tested and failed; do not treat the shared/private representation plan or its listed baselines as pending work. Check STATUS.json and the latest continuation report first.

The current Pantomime cross-dataset endpoint is not yet a method/contribution: a broad swipe/circular label aggregation is only semantic metadata, with no kinematic-equivalence evidence. Exact-release license evidence is positive at the institutional-catalog level, but archive manifest/file-level association is unverified, and the cloud shell cannot reach Zenodo. Targeted prior-art review preempts generic heterogeneous gesture transfer; exact X4/UWB-Gestures to Pantomime transfer search was bounded, not exhaustive. Do not fit or train on that endpoint until a distinct question, target-label-independent semantic mapping, manifest access, and frozen participant-disjoint protocol are in hand.

Current environment check found restored UWB data and the full 789-file source snapshot, only defunct Python records, no active training, restricted shell egress, and CPU-only. No labels or arrays were opened during the 2026-10-09 review. Older positive scores on other splits or candidate screens are not validated contributions.

Latest records:
- STATUS.json
- results/cloud_20261009_endpoint_review/RUN_REPORT.md
- research/PANTOMIME_ENDPOINT_PRIOR_ART_20261009.md
- memory/CLOUD_CONTINUATION_20261009_ENDPOINT_REVIEW.md

---

# 下一轮研究：数据到达前的候选与依据

2026-09-30。状态：迁移准备与初步来源核查，尚未实施新网络或在真实数据上运行。当前云端网络允许 GitHub 及软件包域名，未开放 arXiv/OpenReview/IEEE；本轮读取的是作者仓库 README 与可访问源码，**没有把它们冒充论文全文阅读**。迁移期间可查清原信号与旧模型，访问论文后补充方法细节及最近工作的系统对比。

## 已核查来源

| 来源 | 本轮实际读取 | 可以据此判断的内容 |
|---|---|---|
| Khosla 等，Supervised Contrastive Learning（2020），[论文](https://arxiv.org/abs/2004.11362)、[作者实现](https://github.com/HobbitLong/SupContrast) | README.md、losses.py，源码快照与blob SHA见 sources/ | 标签正样本与多增强视图的监督对比学习已有成熟实现；“加对比损失”本身不是创新 |
| Zhou 等，Domain Generalization with MixStyle，ICLR 2021，[论文](https://openreview.net/forum?id=6xHJ37MVxxp)、[作者仓库](https://github.com/KaiyangZhou/mixstyle-release) | README.md及其中给出的PyTorch实现与使用说明 | 源域特征均值/方差混合可以生成域变化；不能不加验证地认定雷达信号统计变化都属于无关风格 |
| Chen 等，Domain-Generalized Gesture Recognition via mmWave Radar Signal Multi-View Learning，IEEE TIM 2025，[DOI](https://doi.org/10.1109/TIM.2025.3637962)、[作者项目](https://github.com/Tkwer/RadarStream) | 作者仓库README中的正式引用与传感器/信号信息 | “雷达＋多视图＋域泛化”已存在；需要阅读全文，区分信号表征视图与多个物理雷达，避免重复主张。仓库不是该论文训练实现，尚未复现 |

FMCW MIMO 示例能产生角度/多普勒图，不能直接推断当前 UWB 缓存也具有相同可观测量。没有原始复数与设备参数证据时，不将解析相位运动称为真实硬件测得多普勒，也不提出无依据的3D手部重建。

## 优先候选：跨用户正样本约束的共同与视角特有动作表征

已有 LTR 为93.44%，HV6约75%，用户差异仍是可以检验的空间。候选问题是：三部雷达看同一个动作，但投影不同；直接要求全部特征一致可能丢失互补运动信息。先检验将共同语义约束与视角特有残差分开，是否改善跨用户表现。

待冻结的具体机制：共享编码器输出共同动作分支与带物理视角标识的特有分支；共同分支用同一片段跨雷达配对，以及同类别跨源用户正样本约束；只在源训练用户中构造正样本。特有分支保留每路信息，真实三路融合分类。训练期以源用户身份混合低层统计作候选增强，但须检验是否破坏幅度/运动结构。测试不访问用户标签或未获得的信号。

这只是候选假设。shared/private分解、监督对比和域统计混合都已有研究；**不能仅凭组合称其首创**。在相近雷达论文及共同/特有表征论文核查完成、消融显示有特定新作用后，才确定论文贡献表述。

## 实验路线与必要对照

1. 首先恢复原预处理、质量掩码和固定 LTR 基线，确认迁移及路径适配不改变输入。留存原ID、dtype、类/用户数量、来源SHA和各折验证/测试划分。若9300映射不一致先修复，不能在不同数据上比较。
2. 同编码器、相同六人训练/一人验证/一人测试：CE三路基线、普通SupCon、跨用户正样本SupCon、共同/特有分支无新约束、完整候选。将 MixStyle 单独成一项消融，先核查增强物理合理性。不同时大规模搜索网络与增强。
3. 优先跑完整八折可行性先导；超参数只按验证用户选择，模型种子和候选网格在运行前写出。当前8用户是开发数据，所有结论标为探索。若选取失败折进行诊断，随后再跑全八折，不能只报有利用户。
4. 主指标为每用户各存在类别的macro accuracy再对8人平均，同时报告最差用户、全样本准确率、逐类混淆与配对差。统计重采样单位是用户。主机制需要相对同协议强CE对照有稳定的跨用户收益；吞吐、显存或模型规模另报。93.44%的旧数字只作线索，不能代替本次对照。
5. 有信号后用多种子、跨用户正样本的随机配对对照、移除共同/特有分支及真实雷达数量消融验证机制。随后以独立用户/独立会话确认；重复研发用户不能充当外部确认。

备选路线按证据排序：三路教师→单路学生的互补信息蒸馏；时间/距离局部运动对齐与非学习DTW基线；基于动作前缀的早期识别。蒸馏须检查单路信号信息上限与同结构CE对照；前缀识别必须将4.5秒动作完整片段切成因果前缀，不能仅把推理耗时称作提前识别。数据到达后选择一条先做，不承诺全部并行训练。

## 真实实验交付标准

至少有一条完整训练与预测链路，以及在用户真实数据上可重算的模型对照和机制消融。保存配置、输入/源码哈希、折与样本ID、验证选择、每用户预测、权重、运行命令和失败尝试。论文创新性和效果必须有证据，不保证研究一定成功；持续迭代也不能用标签泄漏或改变评价口径制造成功。
