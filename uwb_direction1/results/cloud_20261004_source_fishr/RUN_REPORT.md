# 源梯度方差干预：完整固定对照结果

2026-10-04T13:12:57.455143+00:00。联合门槛未过；停止这组 lambda/EMA/warmup/seed/tail/normalizer/coordinate 网格及八折扩展。不能用次选模或移除对照改变结论，也不能把源方差机制4/6通过当作分类增益。该结果不证明所有 Fishr 设置或域泛化方法无效。

18条新轨迹、432轮完成；连同6条旧普通24轮基线，全24条独立函数推理、样本角色、源批顺序、选模与产物SHA审计通过。本次没有重启训练或修改科学配置。此前3次环境中断按完整状态恢复，证据保留在RECOVERY_01–03与REPLAY_01/03。用户研究目标仍未达成，没有新的训练在运行。

## 实际结果

六个反复用于研发的源用户3–8；每条四训练用户、一源验证用户、一诊断用户。指标为每用户存在类别macro accuracy，再对六人等权平均。主模型是固定13–24轮逐更新权重平均，最小验证CE端点只是次指标。不是旧外层八折93.4417%的同协议比较。

| 方法 | 主tail | 次验证CE端点 |
|---|---:|---:|
| shared128 | 92.722222% | 91.902778% |
| balanced_erm | 92.702020% | 91.378788% |
| real_variance | 92.869949% | 91.059343% |
| permuted_variance | 92.771465% | 91.156566% |

真实域约束主tail相对各对照：

| 对照 | 平均差（百分点） | 正差用户 | 单项门槛 |
|---|---:|---:|---|
| balanced_erm | +0.167929 | 4/6 | 未过 |
| permuted_variance | +0.098485 | 5/6 | 未过 |
| shared128 | +0.147727 | 3/6 | 未过 |

冻结联合门槛：对匹配balanced ERM至少+0.5pp、类内域标签置换至少+0.25pp、旧普通tail至少0pp；每项至少4/6正差，每位不得下降超过1pp。Git中的PER_USER.csv仅将换行统一为LF，数值行与备份原始CRLF文件逐项一致；备份SHA对应原始字节。ANALYSIS.json还保留逐用户差与描述性用户bootstrap区间；重叠训练集和反复研发使区间不能当独立确认。

## 方法、执行与证据

预注册提交fc19c823aec4c7d447527c9aa78d6c36da2354ca。三新臂共享初始化、265836参数CNN、源批采样/Dropout RNG、AdamW/学习率与24轮预算；每批四训练用户各16例，每轮75步。首轮CE，第二轮首批源梯度范数一次校准固定系数；后续不调整。所有臂维护真实与类内置换两套梯度方差EMA，首轮权重与源校准逐位相等已核验。旧shared128为补充强参照，采样方式不同，故匹配balanced ERM才是主要正则对照。

实际新训练epoch累计3.350252小时，不含中断等待、恢复重放、审计和传输；预先真实批成本含75%余量估计5.587792小时、预算8小时。CPU1线程，PyTorch2.14.0+cpu，Python3.12；无GPU。完整24轨迹共714项产物SHA检查，ERM/tail验证与诊断logits独立功能式前向逐位一致。

仅作源目标描述：真实约束六轨迹的“固定系数×epoch均值惩罚/epoch均值源CE”在2–6轮各轨迹中位数范围0.119388至0.629599，13–24轮为7.70466e-05至0.00436539。这不是逐批比值的均值，不是梯度范数或失败原因的因果检验；不能据此启动事后系数/时间窗搜索。SOURCE_HISTORY.json及命令化脚本保留计算，未拟合任何模型或读取诊断准确率字段。

## 文献核查与限制

本次通过出版方GitHub取得Fishr正式PDF，读主文1–9页和DomainBed附录19–24页；此前“主文未读”属于历史状态，原预注册文件不追改。详见../../research/FISHR_PUBLISHED_REVIEW.md及SHA/blob回执。Fishr主表使用测试域选模；源域选模Table9的七数据集均值67.1%对ERM66.6%，不是UWB效果或本协议。作者固定EMA除数1/(1−gamma)有明确导数补偿目的；本实验常规时间偏差校正与源范数校准是已声明的不同实现，不是精确复现或原创。论文已研究均值+方差组合，不能重复包装。

单种子、六研发用户、源集合重叠；没有外部独立用户确认。机制诊断与分类干预结果必须分开。没有测试标签拟合、选择超参或替换主指标；未改789原源码快照，原始数组未入Git。

## 完整恢复与私有备份

DRIVE_BACKUP_RECEIPT.json：38逻辑包、92个≤7MiB私有owner-only块、565203039压缩字节、595成员，全部认证回下载并验证逐块、重组包、成员SHA。含558训练成员（432轮段平均及18份完整model/AdamW/scheduler/RNG/EMA/best/tail状态、ID/预测/选模），配置、日志、审计和论文阅读证据；旧6基线复用已验证SWAD及BASELINE_STATE备份。原始大数组没有纳入本备份。旧中断恢复证据仍由历史PARTIAL回执保存。

恢复只用回执持久ID/URL，先核私有元数据，再raw fetch并立即物化校验，最多3并发；提取到新的空目录，禁止覆盖已有结果。ARCHIVE_RECEIPT.json的uploaded=false是本地打包时快照，最终远端依据DRIVE_BACKUP_RECEIPT.json。本次最后上传阶段复用90个已验证对象，新传2个对象；这些数字不代表整个实验总新增量。

下一项：Close this fixed gradient-variance intervention. Next investigate a distinct radar-signal representation mechanism against original experiments and published prior art, then pre-register a train-only diagnostic and matched controls if justified. Do not reopen failed Fishr, SAM, temporal-scale, DRO or adaptation grids. No next training is currently launched. 自动接续保留；不能把自动接续已启用说成后台模型训练正在运行。
