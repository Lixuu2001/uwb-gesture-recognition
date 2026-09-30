> 历史聊天文件创建记录恢复；本次未读取原文件或重新执行旧实验。Windows 路径是历史位置。

# 两条新路线：本轮研究与实验已完成

完成日期：2026-09-29（北京时间）。本轮使用同一 UWB 数据集，完成 8 用户、模型种子 2026 的预定路线筛选、条件策略试验、处理延迟测量和独立复核。

先看[最终中文研究报告](D:/uwb_gestures/experiments/p1_20260928/report_s2026_release_v2/REPORT.md)，运行入口见[实验顺序与复现说明](D:/uwb_gestures/experiments/p1_20260928/EXECUTION_AND_REPRODUCTION.md)。

## 本轮判断

- **单雷达少量动作校准**：3 类动作各示范 3 次。本轮 rank2 用户因素在未示范的已知动作上达到 68.951373%，验证选出的简单校准对照为 72.751147%；配对差为 −3.799774 个百分点，仅 1/8 用户改善。按预定规则停止该复杂机制，未触发第二种子或 adapter 扩展。结论范围是本次机制与协议。
- **多雷达预算识别**：真实三路共享模型为 93.441667%；验证选出的固定两路为 89.536932%。主级联为 91.920833%，实际平均处理 2.00461553 路，严格超出 2 路门槛。学习收益预测器为 90.863258%、2.00787311 路，未通过精度、成本和逐用户条件。停止该预测器，保留真实多雷达作为后续系统的候选基础。
- **处理延迟**：同一组 279 个片段已可用后，本机测得收益预测器中位延迟 26.3057 ms，固定三路 17.3396 ms。处理路数减少没有转化为本实现的提速。测量边界、桌面 GPU 上下文及非独占限制见报告。

A 与 B 各自主指标的查询范围不同，不能直接相减。报告另给出严格对齐的 7440 条同查询比较。固定融合的工程价值已得到本轮证据支持；本轮没有找到可作为第一部分主创新的有效新机制。8 名研发用户、单模型种子的结果仍需未来独立用户确认。

## 已完成内容

- 文献调研、实施方案、预注册与事前修订。
- 8 折 Right 基础模型及冻结激活导出。
- A：8 折 × 20 次支持抽样，完整对照、验证选择、目标评估及用户层聚合。
- B：8 折共享模型、7 种真实输入子集、固定组合、成本匹配混合、简单级联与 oracle 机会诊断。
- 条件收益预测器与类别收益表，保留负增益和联合追加动作。
- 279 个预选片段 × 7 种方法，共 1953 条实际处理路径延迟观测。
- A、B、条件策略、延迟归档四项独立审计，以及最终规格和代码质量审查。
- 最终测试 145 项通过、1 项可选 CUDA 重放检查默认跳过；6 条 PyTorch nested-tensor 提示。正式 GPU 实验与延迟测量另已执行。
- 最终报告的 28 件文件和 467 个绑定来源文件的 SHA256 全部核对通过；两张图已目检。

完整验证记录见 [FINAL_VALIDATION.json](D:/uwb_gestures/experiments/p1_20260928/FINAL_VALIDATION.json)；测试结果见 [FINAL_TEST_RESULT.txt](D:/uwb_gestures/experiments/p1_20260928/FINAL_TEST_RESULT.txt)。原始预测、权重、选择文件和审计脚本均保留在各阶段目录。

## 代码与产物入口

代码保存于工作树 `C:/Users/19252/.codex/worktrees/uwb-user-calibration/uwb_gestures`，分支 `codex/user-calibration-20260928`，本地提交 `01c9785ee47b8de784fc0efb53c307498edf5b7f`。代码工作树已核对为干净状态并保留。

- [最终报告目录说明](D:/uwb_gestures/experiments/p1_20260928/report_s2026_release_v2/README.md)
- [完整计算结果 JSON](D:/uwb_gestures/experiments/p1_20260928/report_s2026_release_v2/computed_results.json)
- [同查询比较 CSV](D:/uwb_gestures/experiments/p1_20260928/report_s2026_release_v2/cross_route_7440.csv)
- [文献调研记录](D:/uwb_gestures/experiments/p1_20260928/LITERATURE_NOTES.md)
- [原始预注册](D:/uwb_gestures/experiments/p1_20260928/PREREGISTRATION.md)

最终交付以 `report_s2026_release_v2` 为准；其余 `report_s2026*` 目录为保留的中间版本。预注册和修订文件保留冻结时的状态文字，执行完成情况以本页和 `RUN_STATUS.json` 为准。生成器的三个图注百分数对应本轮冻结结果；未来改用不同实验结果时须同步更新，详见质量审查记录。

