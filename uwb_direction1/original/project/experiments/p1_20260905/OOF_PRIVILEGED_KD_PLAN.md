# Cross-user OOF Privileged Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 检验训练用户内部折外生成的三视角软标签，能否给Right单雷达学生提供超过单视角折外教师和普通平滑的跨用户增量。

**Architecture:** 每个外层LOSO折只保留7名训练用户；再对这7人做内部留一用户，用其余6人的低维代理分类器为被留出的第7人生成折外概率。三视角代理读取L/T/R幅度和相位运动，单视角控制只读取Right；两者为外层训练样本生成逐样本软目标，部署学生始终只读取Right幅度+相位。该筛选通过后才值得训练56个深度内部教师。

**Tech Stack:** Python、NumPy、PyTorch、scikit-learn LogisticRegression、8折LOSO、用户配对bootstrap。

---

## 方案选择

1. 继续使用外层训练集内拟合的深度教师：已有教师训练准确率100%、T=2真类概率0.9842，软目标接近标签，当前实现未发现教师增量，停止。
2. 每外折训练7个深度内部教师：最接近原三视角教师，但共56个教师训练，先验成本高。
3. 先用轻量三视角折外代理筛选：严格保持被试隔离，可在几分钟内生成软目标；若不能超过Right折外代理和全局平滑，不再投入深度教师。采用此方案。

## 固定数据流

- 外折h：测试用户h完全隔离；原验证索引也不参与代理拟合或软目标生成。
- 内折s：仅在外层梯度训练索引中，用subject!=s的样本拟合StandardScaler和LogisticRegression，再预测subject==s的样本。
- 代理输入：每个90×189图按10×21网格做局部均值，得到210维；单视角为Right的幅度+相位420维，三视角为L/T/R的幅度+相位1260维。
- 软目标：从匹配全局平滑矩阵取得该外折固定真类概率m；将OOF代理的真类概率置零并归一化其余11类，仅用它决定非真类形状，q_i(y)=m，q_i(j)=(1-m)p_i(j)/sum_{k!=y}p_i(k)。学生损失保持CE+0.5T²KL(q_i||softmax(z_i/T))，T=2。single、multi与global三者的真类概率和总平滑质量完全一致。
- 两学生均为Right幅度+相位、同一初始化、相同批顺序、Adam 1e-4、weight decay 1e-4、40 epoch、patience 8、验证CE选模。

## 预设判据

三视角OOF-shape学生必须同时满足：

- 相对Right单视角OOF学生平均至少+2.00 pp且至少6/8用户改善；
- 相对已完成的最强无教师平滑基线平均至少+2.00 pp且至少6/8用户改善；
- 所有8折索引、初始化、参数量、代理来源与逐样本预测审计通过。

否则只报告“未发现三视角折外软标签的增量”，停止深度内部教师，不调权重、不挑用户或类别。

### Task 1: 折外代理与泄漏测试

**Files:**
- Create: `experiments/p1_20260905/oof_privileged.py`
- Create: `experiments/p1_20260905/tests/test_oof_privileged.py`

- [x] 写失败测试：外层validation/test索引不出现在任何fit/predict集合；每个训练样本恰有一行概率；概率有限、非负、行和为1。
- [x] 运行专项测试确认接口缺失后实现 `pooled_features(views)`、`cross_user_oof(...)` 和折外目标函数。
- [x] 重新运行专项测试和完整测试集；34项测试全部通过。

### Task 2: 生成Right与三视角折外概率

**Files:**
- Create: `experiments/p1_20260905/oof_privileged_results/proxy_h1.npz` through `proxy_h8.npz`

- [x] 对每个外折生成single/multi概率、train/val/test索引和7个内折的fit/predict索引。
- [x] 保存特征源哈希、代码哈希、代理准确率、迭代数与warning；8折无warning、无类别缺失。
- [x] 独立复算概率行和、覆盖集合和保留集合交集；训练索引各覆盖一次且validation/test交集为空。

### Task 3: 匹配学生训练

**Files:**
- Modify: `experiments/p1_20260905/oof_privileged.py`
- Create: `experiments/p1_20260905/oof_privileged_queue.py`

- [x] 写失败测试并实现固定真类质量的shape-matched OOF目标；OOF目标不反传，single/multi网络结构一致。
- [x] 实现 `sv_R_oofshape_single` 与 `sv_R_oofshape_multi`；不读取三视角深度teacher checkpoint。
- [x] GPU smoke通过，16次训练全部完成。

### Task 4: 配对审计与论文判决

**Files:**
- Create: `experiments/p1_20260905/oof_privileged_summary.py`
- Create: `experiments/p1_20260905/oof_privileged_results/RESULTS.md`
- Modify: `experiments/p1_20260905/FIRST_PART_REPORT.md`
- Modify: `experiments/p1_20260905/FIRST_PART_CHAPTER_DRAFT.md`

- [x] 对8折重算accuracy/macro-F1，核对同折测试标签、索引、初始分类头、参数量和代理来源。
- [x] 计算multi-single、multi-全局平滑、multi-CE和multi-普通KD的用户配对差值与bootstrap区间。
- [x] 按固定双判据否决三视角OOF候选：multi相对single为−0.23 pp、4/8改善；相对global为−2.87 pp、0/8改善。
- [x] 根据最终结论更新第一部分章节，明确区分相位表征、常规正则化、三视角信息和重建负结果；跨用户折外候选停止扩展。
