## 2026-10-04T15:20:44.233381+00:00 距离带能量有向面积源内对照完成：联合门槛失败

预注册743923c39fc3db8bb117b96046e5ae71b58a5e85先正常推送回读，再成本220.805秒（2倍余量）通过1800秒预算。源用户3–8共7100质量片段，六轮四训练/一验证/一诊断；旧第24轮CNN不重训，五臂90次岭拟合、30所选读出。主signed93.079545%，普通CNN读出93.063131%，base92.771465%，unsigned92.983586%，shuffled92.968434%，旧tail92.722222%。signed对base+0.308081pp（5/6胜），对unsigned+0.095960pp（2/6），对shuffled+0.111111pp（3/6），未达+0.5/+0.25/+0.25pp联合门槛；对普通读出仅+0.016414pp。停止这组带数/级数/归一化/shuffle/正则/seed网格及据此八折扩展，不能移除强控制或换主臂包装成功。

实际提取/拟合/保存75.220秒，独立审计4.550秒：全7100原RF距离带与独立Chen递推，最大差3.192e-15；30所选读出独立sklearn重拟合、36固定原数据CNN函数式抽查、40产物SHA/角色/选择/指标/门槛通过。90候选验证预测运行时独立sklearn核对。原数据/prepared/标签/用户和核心代码/协议运行后SHA不变。全部描述/隐藏向量/ID/候选及最终预测/30读出系数/六个旧CNN检查点和tail来源/代码/日志/作者数学PDF共85成员，62014153压缩字节9个私有owner-only块认证回下载及块/包/成员SHA通过。回执results/cloud_20261004_energy_area/DRIVE_BACKUP_RECEIPT.json，报告同目录RUN_REPORT.md。

文献确证level2面积/路径签名为既有数学，原项目也已有跨雷达轨迹面积；本轮只试单雷达内部距离带能量时序，不宣称物理轨迹或整体速度不变。读到作者Reizenstein2015/2017数学说明PDF第1–3、5–7页、库文档和动作签名实现README；不是TOMS或动作论文全文。详见research/ENERGY_AREA_REVIEW.md及receipt。

当前无训练/诊断进程，无下一实验启动；目标仍未达，自动接续保留。下一步须新的雷达观测机制和先例排重；不重复当前探针或已失败局部匹配/Fishr/SAM/DRO网格。普通冻结CNN岭读出的工程改善不等于原创贡献。原789源码及原始大数组不动，无需用户本地迁移。以下保留历史状态。

## 2026-10-04T14:50:15.931014+00:00 局部回波对应源诊断完成：门槛未过，无识别训练

预注册59a999ee7eeaf65a46ef11642ddf20ff42e561e3先推送回读后执行。只处理外层fold1训练用户3–8的568条分层源片段；每条件332280对局部patch。三路相邻可靠比例均值11.221%/13.240%/11.811%，远帧4.834%/7.811%/6.750%，差6.387/5.429/5.062pp；非零可靠位移8.277%/10.754%/9.493%。三路均0/6满足相邻≥20%、比远帧≥10pp、非零≥5%的规则（须2路各4/6），故停止此窗口/位移/置信阈值网格与据此启动识别分支。不是分类准确率，也不否定全部光流方法。高相关常伴随多解，不能把匹配位移当真实手部轨迹。

提取与压缩共13.281秒，成本预算600秒；独立审计4.263秒，29536个原信号FFT/标量相关核对最大误差1.984e-13，全代价体决策/18用户视角统计/源角色与SHA通过。初次审计因NPZ循环重复解压停止，修为一次缓存后完成；只改审计I/O，真实提取/科学配置未重跑或改动。完整代价体、源ID、位移/掩码、代码/协议/日志/审计及作者幻灯片19成员，143247631字节、20个≤7MiB私有owner-only块认证回下载和块/包/成员SHA全部通过。回执results/cloud_20261004_local_echo/DRIVE_BACKUP_RECEIPT.json；原数据及789源码快照不变。

文献核查是PWC-Net作者README、UnFlow作者README/指定losses.py段及作者AAAI2018报告34页文本/公式/表格，不冒称论文主文全文；详见research/LOCAL_ECHO_REVIEW.md。相关代价体和前后向一致性有既有先例，非原创。原始全局质心/展宽试验与此次局部匹配不同，但当前可行性仍不足。

当前无训练或诊断进程；用户目标未达，自动接续保留。下次先检查最新状态，不重复此源诊断或重开Fishr/SAM/DRO/适配等失败参数网格；需要新的雷达观测依据、先例排重和前瞻协议才启动下一实验。云端数据仍在，不要要求用户本地迁移。以下保留历史状态。

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
