# P1 冻结配置的追加记录

`p1_frozen.yaml` 保持原样不改，它记录的是 2026-09-06、在看到任何候选臂的外层结果
之前冻结的内容。本文件记录此后实际发生的偏离，按发生顺序，并注明**决定时已经看到
了什么**。目的是让读者能区分"事先规定的"和"事后追加的"，而不是把后者伪装成前者。

## A1. 补跑第二个 model_seed（2027）

- `p1_frozen.yaml` 写的是 `model_seeds: [2026]`，并在 `not_run_in_p1` 里列了
  `second_model_seed`（原规定：只给通过 +2 pp 闸门的候选补第二 seed）。
- **实际补跑了 seed 2027**，臂为 A-GS / A-SC / A-US / A-MV / B-GS / B-SELF / B-MV。
- 决定时已经看到：seed 2026 的全部 8 个臂 8 折结果，包括 A-MV vs A-US = −0.09 pp、
  B-MV vs B-SELF = −0.14 pp。没有任何候选通过闸门。
- 理由：补第二 seed 是为了确认**零结果**的方向，不是为了寻找正结果。事后看这个决定
  是必要的——两处方向在换 seed 后翻转（A-MV vs A-US、A-US vs A-SC），只有一个 seed
  的话会得出错误的方向性结论。
- 影响：这不是预注册确认，是事后加强。合并两 seed 的统计（含 A-SC 的 8/8 被试）
  属于此次追加分析。

## A2. A-TS（seed 2026，8 折）

- 预注册的臂表里有 A-TS（§5.6），但 P1 首批只列了 8 个臂，未含 A-TS。
- 决定时已经看到：seed 2026 的 A-US / A-MV 结果。
- 理由：拆开"教师训练时见过三视角"和"目标读取三视角"两个子机制，把 A 的负结果做完整。

## A3. C 的事后诊断

- `predictability.py --raw`（非残差版）和 `diagnostics/probe_diversity.py`
  （PCA 与内部相关性对照）都是在看到残差版结果之后写的。
- 目的是检验残差版结论的自指风险。**结论：这两个对照都不充分**，见
  [INDEPENDENT_REVIEW_20260907.md](../diagnostics/INDEPENDENT_REVIEW_20260907.md)
  与 P1_REPORT §3.4 的撤回说明。

## A4. 2026-09-07 独立复核后的代码修正（未重跑训练）

以下修改在 128 个 run 全部完成之后，因此**不影响已保存的产物**，只影响将来的运行：

1. `relation.global_within_class_derangement`：正确的实例错配对照，在整个训练集合内
   按 (类别,用户) 预生成无固定点置换。原 `shuffle_within_class` 只在 32 样本 microbatch
   内交换，实测只有 31.58% 的训练呈现次数可能被更换。已完成的 A-SHUF / B-SHUF 两个 run
   用的是旧版；`shuffle_scope='microbatch'` 与 `A-SHUF-LOCAL` / `B-SHUF-LOCAL` 保留
   只为复现它们。新增 `shuffle_changed_fraction` 写入 provenance。
2. `train_student._audit_gradients`：改为用 `torch.autograd.grad` 单独取**辅助损失**
   对编码器的梯度。原实现在合并反传后读 `.grad`，而分类 CE 本身就会让编码器梯度非零，
   因此证明不了辅助损失到达主干。已完成 run 的该字段是旧口径，不可用作证据。
   修正后发现：B 的过程损失只到达时间分支，距离分支梯度为 None。
3. `summarize.py`：seed 覆盖严格化（见 A5）。

## A5. summarize.py 的 seed 记账修正（P1 进行中）

在 seed 2027 补跑期间发现：原 `arm_summary` 会对每折已有的 seed 取平均，导致部分折
有两个 seed、部分只有一个时，头条准确率属于任何一个协议都不成立（曾一度把 A-US
显示成 75.40 而非 75.21）。已改为每行显式标注 seed 集合，且只有两个臂在该 seed 集合上
都齐 8 折才输出比较。此后所有报告数字都出自修正版。

## A6. 与 P0 阶段的两处偏离

已记录在 `p1_frozen.yaml` 的 `deviations_from_the_written_plan` 中（B-SHUF 提前、
不做帧移对齐、objectives.py 的模块归属），此处不重复。
需要补充的是：**B-SHUF 提前到 P1 的理由成立，但它当时的实现无效**（见 A4.1），
所以这条提前实际上没有产生它预期的证据。
