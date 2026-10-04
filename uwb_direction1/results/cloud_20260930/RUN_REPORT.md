## 2026-10-04T13:12:57.455143+00:00 源梯度方差固定对照全部完成；联合门槛未过

18新轨迹432轮+6旧基线共24轨迹独立审计通过，714产物SHA、所有ERM/tail验证与诊断logits逐位一致。六源主tail：旧普通92.722222%、匹配ERM92.702020%、真实方差92.869949%、类内域置换92.771465%。联合门槛未过；停止这组 lambda/EMA/warmup/seed/tail/normalizer/coordinate 网格及八折扩展。不能用次选模或移除对照改变结论，也不能把源方差机制4/6通过当作分类增益。该结果不证明所有 Fishr 设置或域泛化方法无效。

全部38逻辑包/92私有owner-only块/565203039压缩字节/595成员认证回下载及块/包/成员SHA通过，含全部432段与18完整优化器/RNG/EMA状态；索引results/cloud_20261004_source_fishr/DRIVE_BACKUP_RECEIPT.json。最终SUMMARY、AUDIT、ANALYSIS、PER_USER、SOURCE_HISTORY和FINAL_MONITOR同目录。旧恢复证据另保留在PARTIAL01–03；原数据/prepared/789源码快照不变，不重复迁移。训练598和归档607已经完成；没有下一训练进程。自动接续仍开启，目标未达。

正式Fishr主文1–9页、DomainBed附录19–24页已通过出版方GitHub PDF读到，SHA与blob记录在research/FISHR_PUBLISHED_REVIEW_RECEIPT.json。主表测试域选模不适用用户约束；作者固定EMA除数有明确目的，本版本不同且不是精确复现。未事后改变冻结实现或门槛。Close this fixed gradient-variance intervention. Next investigate a distinct radar-signal representation mechanism against original experiments and published prior art, then pre-register a train-only diagnostic and matched controls if justified. Do not reopen failed Fishr, SAM, temporal-scale, DRO or adaptation grids. No next training is currently launched.

以下保留历史执行记录；历史“运行中/主文未读”不代表当前状态。

# 云端迁移、原数据审计与首次先导

日期：2026-09-30。迁移完成；研究目标尚未完成。没有需要用户补交的数据。

## 恢复证据

- 私有 Drive 的163个分块均通过认证接口实际下载到 `/workspace/uwb-transfer/chunks/`。下载最大并发3，压缩字节总数2,006,404,772，失败0，名称、大小和SHA-256逐项吻合。
- `transfer.py verify` 成功，`restore` 写入新空目录 `/workspace/uwb-restored-20260930`。86个原文件共2,141,981,100字节，全部原文件SHA-256通过；审计脚本又独立重算一遍。
- `TRANSFER_MANIFEST.json` SHA-256：`10516baf0b49a84e6a39169ef447fffcdcb29defa2a6cc79e301dc938ddceb4d`。
- `SOURCE_MANIFEST.json` SHA-256：`8f19d50aacdcc2cfe3457e9dc4785a0cf440ef76c5777a751806b5575a9f2884`。GitHub原快照共789文件；其中28个当前所需核心文件已从不可变提交`c735e9454ba3d5168d931eb0158f2bb0fcf12808`物化，并逐个验证原字节，位于 `/workspace/uwb-source-c735e94/project/`。
- 没有更改Drive分享权限。原始数组和权重未提交到GitHub；研究分支所在仓库已确认仍为private。

机器记录：[恢复凭据](TRANSFER_RECEIPT.json)、[RESTORE_VERIFIED](RESTORE_VERIFIED.json)、[完整审计](AUDIT.json)。私有持久下载索引与原迁移清单位于 `../../migration/20260930/`，不包含短期认证链接。

## 数据与旧预测重算

X为float32，形状`[9593,3,90,189]`，全量有限值检查通过。原标签/用户数组与prepared标签/用户数组逐项一致。原质量掩码排除HV2类别10/11的200条、HV6类别3的93条，剩9300条。检测到100对完整三视角记录的精确重复，均不属于保留样本之间的重复。

八折均核验：测试用户H、验证用户H%8+1、其余六用户训练；保存的split、预测中的ID/标签/用户/角色相符；缓存文件SHA与旧结果的SHA一致；验证logits逐元素重放一致。以下是每用户现存类别macro accuracy，再对八人平均，非重新训练结果。

| 旧缓存预测 | 八用户平均 | 最差用户 |
|---|---:|---:|
| 原始Right模型 | 71.2913% | 51.3636% |
| 子集训练模型R | 71.6765% | 51.5455% |
| 子集训练模型TR | 90.5833% | 67.0000% |
| 子集训练模型LTR | 93.4417% | 75.0000% |

LTR各用户依次：96.25%、98.70%、91.25%、96.75%、98.1667%、75.00%、96.4167%、95.00%。历史八折总值93.4417%已重算吻合。

## 首次CPU先导：类内跨用户差异约束

在看到本次新方法测试结果前保存[协议](../../research/FROZEN_PROBE_PROTOCOL.json)。输入为各折原模型已保存的真实三路256维特征，训练统计量只读取六个训练用户。普通读出按类和用户平衡权重作岭回归；候选额外惩罚同一类别不同源用户特征中心在分类输出上的差异。超参数仅由该折独立验证用户选择。此为线性机制探针，不能称为新颖算法或端到端网络训练。

| 方法 | 八用户平均 | 最差用户 |
|---|---:|---:|
| 原LTR融合头 | 93.4417% | 75.0000% |
| 七子集概率等权平均 | 92.8788% | 73.3636% |
| 普通平衡岭回归融合 | 92.8254% | 75.6364% |
| 类内跨用户差异惩罚 | 92.7231% | 75.8182% |

候选相对普通岭回归为−0.1023个百分点，仅2/8用户改善；相对原LTR为−0.7186个百分点。未达运行前规定的继续门槛，停止扩大这一冻结特征约束方案。完整配置、每折验证选择、混淆矩阵和配对差在[FROZEN_PROBE_RESULTS.json](FROZEN_PROBE_RESULTS.json)。每折模型系数和样本级预测保存在 `/workspace/uwb-frozen-probe-20260930/`，可用脚本在数秒内重新生成。

没有将测试用户用于拟合、选择超参数或填补质量排除的缺失类别。八用户已有多轮研发历史，因此所有新方法比较仍属于探索性结果。缺少迁移前模型权重，本轮只重算历史缓存，不声称完整训练重现。

## 复现命令

```bash
python /workspace/uwb-cloud/uwb_direction1/tools/transfer.py verify --package /workspace/uwb-transfer
python /workspace/uwb-cloud/uwb_direction1/tools/transfer.py restore --package /workspace/uwb-transfer --out /workspace/uwb-restored-NEW
python /workspace/uwb-cloud/uwb_direction1/tools/audit_restored.py --data /workspace/uwb-restored-20260930 --package /workspace/uwb-transfer --out /workspace/uwb-audit-NEW
python /workspace/uwb-cloud/uwb_direction1/tools/frozen_probe.py --data /workspace/uwb-restored-20260930 --audit /workspace/uwb-audit-NEW/AUDIT.json --out /workspace/uwb-frozen-probe-NEW
```

所有输出须为独立空目录，不能覆盖原结果。环境：Python3.12.14、NumPy2.3.5、SciPy1.17.0；线性计算限制3线程。迁移及协议检查共14项通过。没有GPU，当前无后台训练任务。

## 研究接续

这次完成了数据接管和一项可否决的真实先导，尚未实现用户要求的“有效且有研究贡献的方法”。下一阶段应检验端到端的共同/视角特有动作表征，保留CE三路、普通SupCon和跨用户正样本约束等对照；不可把本次线性负结果推断为一切跨用户表征学习无效。先补齐相近雷达工作和共同/特有分解文献的全文核查，再冻结实现与计算预算。现阶段读到的QMF、SupCon、MixStyle是作者仓库文本/源码，不能冒称已完成论文全文调研；仅增加普通融合或已知损失也不能单独主张创新。
