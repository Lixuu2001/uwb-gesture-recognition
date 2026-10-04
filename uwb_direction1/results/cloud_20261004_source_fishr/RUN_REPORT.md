# 源梯度方差干预：实现与固定对照

当前状态（2026-10-04）：18条新轨迹中已有6条完成并通过独立审计，144轮产物已私有备份和认证回下载校验。云端第二次中断后已从完整检查点恢复，原队列继续；整组效果与原创贡献均未判定。最新恢复/备份索引见RECOVERY_02.json、PARTIAL_DRIVE_RECEIPT_02.json、MONITOR_02.json。以下保留方法设计及执行记录。

源训练机制诊断4/6通过后，准备三组同批次/采样/优化器实验：balanced_erm、real_variance、permuted_variance。6个源轨迹各3组24轮=18条432轮；另复用6条原普通24强tail，最终共24轨迹独立审计。尚无新分类结果，不是原创主张。

四训练用户每批各16例，按各域固定打乱循环到最长域的完整步数；较少样本用户会重复，三组完全相同。首轮全部纯CE，第二轮第一个源批依据真实用户惩罚梯度校准一次固定lambda=.1*||gradCE||/||gradPenalty||，截断[1e-4,1e4]并记录是否截断；后续不再变化。真实和类内用户标签置换两套EMA在全部三组同步维护，采用常规时间偏差校正，不是作者固定1/(1-beta)；对照保留类-用户计数。历史EMA detach，当前批保留高阶梯度；同一次Dropout前向用于CE和惩罚。没有新增目标/验证标签调参。

主选模固定逐步权重平均13–24轮；次选模最早最小验证CE不替代主指标。联合门槛：真实约束对匹配ERM+0.5pp、置换约束+0.25pp、旧普通tail至少0pp；每项至少4/6位改善、每位不降超1pp。未过停止这组lambda/EMA/warmup/seed/窗口/坐标网格和八折扩展。机制通过只是训练动机，不能预言结果。

合成全架构三臂恢复、EMA/scale/RNG一致、留出输入标签隔离、独立函数推理、解析高阶梯度/有限差分、常规EMA NumPy值与导数、计数保持和普通AdamW单步等价均通过TESTS.json。归档31成员/臂、不可变复用和腐损拒绝的合成fixture通过ARCHIVE_TESTS.json。真实源批成本五步/臂，含75%余量预计5.587792小时，低于预先固定8小时预算；成本不评价候选泛化。

设计与作者Fishr的区别和限制见../../research/SOURCE_FISHR_IMPLEMENTATION_REVIEW.json；主文未读、不声称完整复现或创新。先正常推送冻结代码/协议并回读，再启动。原789文件快照及原数组不改。

实际启动：预注册fc19c823aec4c7d447527c9aa78d6c36da2354ca远端回读后，2026-10-04T08:11:53Z启动训练PID4303和本地归档watcher4319。2026-10-04T08:17:09.211507+00:00观测balanced_erm/held3第10轮，进程活动；还没有整组对照结果。原协议/helper未变，未开启其他科学配置。

第4轮300步启动快照独立验证通过，源端固定scale102.98310954123612未截断；含模型/AdamW/scheduler/RNG/EMA/best/tail和前4段平均的36成员、9380013字节、2个私有owner-only块，认证回下载逐块/整包/成员SHA通过。恢复索引START_DRIVE_RECEIPT.json；并非完整432轮备份。后台watcher仅本地归档，最终仍需私有上传验证和Git记录。

本次确认原训练4303和归档4319均不在/proc，当前容器PID1启动晚于原实验启动时间；队列遗留training不能当进程仍活动。具体终止原因未写入日志。保留原4条完整结果和real_variance/held4第6轮450步全部文件到/workspace/uwb-fishr-interruption-20261004-0934，再核对版本、冻结helper/协议、全部完成产物和检查点SHA及AdamW/scheduler/EMA步数后，按原LAUNCH命令恢复PID603，归档watcher612；未改科学配置、门槛或重跑已完成轨迹。

用保留第6轮独立重放到第9轮（76.385秒），model、AdamW、scheduler、RNG、EMA历史/计数/固定scale/校准、best/tail/选模和逐轮段平均全部逐位一致；history除墙钟耗时和序列化SHA外一致，真实源数据而非仅合成测试。系数26.974675837012917保持不变；未读取held输入作调参。RECOVERY_01.json和RECOVERY_REPLAY_01.json保存证据。

固定审计集合：balanced_erm/held3,4，real_variance/held3，permuted_variance/held3，共4条96轮。全部ERM/tail验证/诊断logits独立函数逐位相同、角色/源顺序/选模和120产物SHA通过；held3三臂首轮权重与源校准完全一致。147个完整轨迹及元数据成员，加19恢复证据成员，合计166成员、149309022压缩字节、10逻辑包/25个≤7MiB私有owner-only块，全部认证raw回下载、逐块/重组/成员SHA通过。见PARTIAL_AUDIT_01.json、PARTIAL_DRIVE_RECEIPT_01.json。备份包含96轮段平均和四份完整优化器/RNG/EMA状态；恢复包另含中断第6轮、恢复第9轮比较及重放状态。不是18轨迹全部备份。

当前观测permuted_variance/held4第2轮，完整结果5/18，进程活动。继续同一冻结队列和原联合门槛，未得整组六源效果；不得根据中途结果调整lambda/EMA/warmup/seed/窗口或扩展八折。归档和后续上传复用/workspace/uwb-fishr-final-20261004及/workspace/uwb-fishr-final-transport-20261004/PARTIAL_INDEX_01.json、records，避免重复上传已有25块。若再次中断，先检查PID，再完整checkpoint恢复；最终全24轨迹审计和其余备份完成后判定效果。原789源码和原始大数组不动。用户研究目标未达，自动接续保留。

云端环境启动后确认旧603/612进程均消失；队列文件仍training，但不能据此声称原进程活着。保留balanced_erm/held5第5轮375步全部文件到/workspace/uwb-fishr-interruption-02-20261004，核对冻结源码/helper/协议、Torch/NumPy、六条完成结果全部文件/段SHA，以及checkpoint身份、AdamW/scheduler/EMA计数和固定scale15.033946960215014后，按原argv恢复训练508和归档watcher521。具体终止原因未记入日志；当前容器启动晚于原进程。未重跑六条已完成轨迹，科学配置和门槛未变；上一次真实数据完整恢复重放证据RECOVERY_REPLAY_01.json仍有效，本次没有另声称新逐位重放。见RECOVERY_02.json。

新增real_variance/held4和permuted_variance/held4两条独立函数推理、角色/源顺序/选模审计通过。累计三臂各held3,4共六条、144轮、180产物SHA通过；此前四条独立前向引用PARTIAL_AUDIT_01且本次归档重新核全部SHA。held4三臂首轮段权重和源端校准完全一致。见PARTIAL_AUDIT_02.json。

当前备份集合14逻辑包、33个≤7MiB私有owner-only块、197926546字节、222成员，全部认证raw回下载过且本次逐块/重组/全部成员SHA重新核验；复用20已验证块，新传13块。包含186完整训练产物（144段+六份完整模型/优化器/RNG/EMA/ID/预测），24元数据成员和12个本次中断第5轮状态/段/恢复凭据成员。先前元数据和第一次恢复证据另保留在PARTIAL_DRIVE_RECEIPT_01.json；本索引不重复纳入，不删除旧备份。恢复入口PARTIAL_DRIVE_RECEIPT_02.json、MONITOR_02.json；不是整组18轨迹备份。

当前观测balanced_erm/held5第16轮，完成6/18，训练/归档进程活动。继续原固定队列，先检查实际PID再决定是否恢复，不能依赖跨环境启动的后台进程存活假设。归档/传输复用/workspace/uwb-fishr-final-20261004和/workspace/uwb-fishr-final-transport-20261004/PARTIAL_INDEX_02.json、records。完成后全24轨迹审计、补齐最终私有备份，再依冻结主tail联合门槛判断，禁止中途调参或把机制诊断当效果/创新。用户目标未达，自动接续保留。
