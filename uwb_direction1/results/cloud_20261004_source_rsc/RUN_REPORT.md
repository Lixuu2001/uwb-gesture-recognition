# RSC 源端固定对照：真实云端训练进行中

## 2026-10-04T18:26:15.949170+00:00 RSC 两臂真实云端训练已启动，首个恢复备份验证完成

冻结提交21915bdf0203984f5ded80fbf40dcca08e18609a在真实成本/候选训练前推送，13文件远端逐字回读相同。真实普通CNN首epoch损失/验证CE/样本顺序/段平均权重与旧基线逐位相同。源批成本含75%余量3.110022h，通过预定10h门槛。云端训练PID4719实际存活，当前rsc24/held3第7轮(step518)，完成0/12新轨迹；不是完整候选结果。运行根/workspace/uwb-rsc-source-runs-20261004，日志/workspace/uwb-rsc-source-train-20261004.log；先查活进程及检查点，禁止重复启动。原始科学配置/门槛未改。

第2轮/step148完整快照7,261,158字节28文件，私有owner-only Drive元数据、认证raw回下载、块/整包/全部成员SHA均通过，见results/cloud_20261004_source_rsc/START_DRIVE_RECEIPT.json。START_AUDIT独立函数式重算该快照全部验证logits/CE逐位一致；保存完整AdamW/cosine/RNG/best/tail/精确source顺序及两段权重，未评价候选诊断输入。这里只是启动备份，12条全部轨迹及最终结果尚未备份。恢复步骤RESTORE.md，精确命令LAUNCH.json。

固定RSC24与random24各六源开发折，共288新epoch；主tail13–24，相对普通tail≥+0.5pp、random≥+0.25pp且不低于既有SAMtail/强CNN岭读出，各≥4/6胜且单人不降超1pp。队列完成自动审计12新+6旧普通+6旧SAM的ERM/tail全部验证/诊断logits、角色/选择/顺序/SHA；新臂另核所有batch屏蔽记录及逐epoch段均值重建tail。强CNN读出只用前次独立重拟合证据和冻结SHA，不冒称本轮重拟合。完成后还须私有最终全部状态/日志/代码备份及认证校验；没有配置上传后台watcher。若进程中断，用同命令从完整epoch恢复，lock防重入；环境销毁不保证后台持续。

RSC是既有方法，本改编不是精确论文复现，作者课程/空间选择与DomainBed固定向量规则差异已记录；未读论文全文，不能称原创。即使源门槛通过，也仍需八折强对照/消融/独立种子及原创研究贡献证据。当前有效原创目标未达到，保留自动接续。数据/789原源码快照不变，无需用户Windows。以下为历史记录。

# 源端 RSC 固定对照：预注册准备

## 2026-10-04T18:21:13.322263+00:00 源端 RSC 固定对照已实现，先预注册再真实计时

下一机制是训练时表示自挑战，已知ECCV2020 RSC，非原创。核查作者README/源码与固定提交DomainBed RSC规则；未取得论文全文。辅助正确类logit梯度屏蔽最高1/3特征，按概率下降选择1/3批样本；辅助无dropout，最终CE保留旧dropout，三路共享CNN1728维。random24在自身状态保持选择规则和逐行屏蔽数量，只随机坐标；不同轨迹选择不保证相同。严格分位数相等项规则、与作者课程/空间配置差异详见research/SOURCE_RSC_REVIEW.md。

六源3–8四训/一验证/一诊断，12新轨迹288轮。固定24轮/seed2026/batch64/CPU1线程/AdamW .001和decay .001，主tail13–24，次最佳验证CE不能替换。相对普通tail至少+0.5pp、random+0.25pp、不低于旧SAMtail和强CNN读出；每项≥4/6胜、任一用户不降超1pp。失败关闭本组掩码/比例/seed/window网格，不据此开八折。已知方法即使通过也不直接满足原创目标。

合成NumPy分位数差分、随机控制逐行数量、辅助RNG隔离、普通AdamW和完整runner两臂断点恢复(模型/AdamW/cosine/tail/best/RNG/预测)逐位一致。当前没有真实RSC计时或训练进程、无候选识别结果。下一步正常推送并逐字回读预注册，再真实源批计时和普通基线第一轮精确复现；10h含75%余量成本门槛通过才启动。运行计划/workspace/uwb-rsc-source-runs-20261004，日志/workspace/uwb-rsc-source-train-20261004.log。先检查进程/检查点防重复。原789源码与数据不动，目标未达、自动接续保留。

