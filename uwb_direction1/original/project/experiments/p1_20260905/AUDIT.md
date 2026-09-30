# 第一部分：Claude 回复的独立核实

核验时间：2026-09-05。证据为当前源代码、原始运行日志、576 个 CSV、相关原始 MAT；不是对 RESULTS.md 的复述。新实验结果另见 results_clean。以下核实不预告新模型的胜负。

## 已确认的数字和误标

| 原日志 | 实际 closed_acc（6 个已知类） | 可核实的主要配置 |
|---|---:|---|
| logs/f2_loso_c0_nofusion.log | 91.71% | 三视角、CR、flatten、abs_std、fusion=False、150 epochs |
| logs/f2_loso_c0_xmv_k1nb4.log | 89.73% | 三视角瓶颈融合版本 |
| logs/sv_loso_cr_c0.log | 82.50% | Left、CR、d_key=32、150 epochs |
| logs/sv_loso_c0.log | 78.00% | Left、Raw、d_key=32、150 epochs |

91.71−82.50=9.21 个百分点是真实记录的差值，但非匹配实验，也不是 12 类全闭集的教师优势。旧日志只打印部分配置，closed_acc 只留下外层聚合；没有所有历史代码/检查点的完整快照，不能凭当前代码保证历史运行毫无问题。

RESULTS.md §23 的 82.64% 是 inspector 最佳阈值的开放集 Accuracy，不能标成 closed_acc。89.73−91.71=−1.98 个百分点是日志中的闭集变化；−1.79 是 OSCR 变化，必须分开。旧汇总文件此次保持原样，避免与别人的未提交修改混写。

## Claude 回复中需要修正的推断

1. **当前闭集头不是线性头。** model.py 的 ClosedSetHead 是 Linear→GELU→Dropout→Linear→GELU→Linear，隐藏层 2048/256。mean 是分类头前的特征聚合。
2. **closed_acc 不来自 inspector 距离集成。** evaluate_inspector 用 extract 返回的分类 logits.argmax 计算 closed_acc；per-view inspector 的距离主要决定拒识评分。所以“9.2 分完全是三个独立观察者的集成优势”缺少机制证据。
3. **F6 支持停止重复已失败的融合配置，不能给所有融合判死刑。** 它更不能预先替代新的 12 类闭集协议。concat 是合理候选，要与 mean 公平比较，且报告分类头参数增量。
4. **§49 的原文反驳了“神经 mean 特征恒盲”。** §49.2 标题就是“物理量层……但该性质不外推到特征层”。mean 特征二分类方向 82.25%，L+R 88.91% 高于 L−R 85.72%。0.00 是物理标量读出的分类准确率，不是质心数值恰好 0.00，更不是代数恒等式的实测证明。
5. **86.3% 不是零训练。** 逻辑回归需要类别标签训练。零训练符号规则的 90.6/92.4 属于另一任务；方向二分类不能当作 8 类或 12 类手势分类。缺少原 scratchpad/doppler_*.py，原 39 维定义、类集合、视角集合、正则及划分细节不能完整恢复。因此新物理基线明确称重新实现，不宣称精确复现或证伪旧 86.3。
6. **取绝对值不等于抹除所有方向信息。** 距离-时间包络移动仍携带方向，且当前相位算子输出的是有回波幅值加权、取主值的相位增量。超过 π/帧会混叠；没有速度真值，不能无条件称真实有符号径向速度。
7. **静态消除不等于对域漂移免疫。** 完整片段均值相减可去除时间不变的复信号分量，但人体、手臂、多径与动作习惯的变化仍会影响特征；后续相位是非线性量。
8. **2→1 合成平均打平支持降低优先级，不构成 1→2 必然失败的逻辑证明。** 本轮不投完整回波生成，不把未运行的一推二写成已证伪。
9. **paper_neg_mixup 属于 Stage-2 inspector 设置。** 本轮闭集 CE 不训练 inspector，它不是直接决定分类头 closed_acc 的训练项；d_key 则会改变编码器，须统一。

## 新发现：数据源本身的异常

576 个 CSV 的缓存前缀逐值一致，未发现缓存额外改写标签/数值。缓存总数 9593 不是理想的 9600。

### HV2 G11/G12：跨标签完全重复

第 2 名被试 G11 与 G12 的各 100 条样本，三视角逐点完全相同。Raw 与 ClutterRemoved 均如此；原始 MAT 的 float64 数组也完全相同，max_abs_diff=0。这是源数据标签冲突，不能归因于 Claude 训练脚本。既不能猜哪类是真，也不能把重复样本跨训练/验证拆分后忽略。

逐视角和三视角内容哈希都确认这 100 对冲突，未发现跨被试逐点完全重复。无跨被试完全重复不代表不存在所有形式的数据泄漏；本轮另以被试/动作索引隔离控制。

对逐样本确定性分类器，这 100 对相同输入、不同标签至少造成 100 个不可避免的分类错误。若在全部9593条上汇总12类预测，仅此冲突就使理论最高 pooled accuracy 约为98.96%。这个上界仅说明该输入/标签矛盾，不能外推到其它协议或修订后的数据。开集划分若将 G11/G12 分在已知/未知两侧，还会出现相同回波需要同时接受和拒绝的冲突。

### HV6 G4：视角样本数不一致

原始 ClutterRemoved MAT/CSV：Left=93、Top=93、Right=162。当前缓存直接取每路前 min=93。Raw MAT/CSV 三路为 93；MAT 与 CSV 的浮点舍入略有差异，不能把极小舍入差异误判为样本错配。缺少事件时间戳/独立同步真值，不能验证 CR 的 162 条 Right 中哪 93 条与 L/T 配对。

### 正式实验的保守处置

统一排除 HV2 的 G11/G12 共 200 条，以及 HV6 G4 的三视角单位共 93 条；不重标、不填充、不修改原缓存。留下 **9300 条、12 类、8 位被试**。HV2 测试 1000 条/10 类，HV6 1100 条/11 类，其余每名 1200 条/12 类。所有对照同一质量掩码。原先导保存在 results；正式新结果在 results_clean，二者不能混算提升。

## 可复核材料

- results/source_audit.json：全部 CSV 路径、SHA256、shape 和缓存一致性；原日志哈希和逐用户 OSCR。
- results/data_manifest.json：原缓存/依赖源码 SHA256、有限性和重复索引。
- results/quality_audit.json：原始 MAT 的冲突证据、逐视角重复索引、排除索引。
- pipeline.py、train.py、physical.py、suite.py：新流程，独立于 main.py 开集/注册流程。
- tests/test_pipeline.py：用户/事件隔离、预处理等价、相位符号与混叠、concat 顺序、教师目标梯度隔离。

## 文献边界

- [UWB-Gestures 原始数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8041886/)：数据采集、视角和样本组织的来源，不提供本轮轨迹/速度真值。
- [C²KD，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html)：说明直接跨模态蒸馏并非必然有效，不能代替 UWB 本地验证。
- [ShaSpec，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wang_Multi-Modal_Learning_With_Missing_Modality_via_Shared-Specific_Feature_Modelling_CVPR_2023_paper.html)：共享/特有特征与缺失模态已有相关工作，不把普通拼接/蒸馏本身称作首创。
