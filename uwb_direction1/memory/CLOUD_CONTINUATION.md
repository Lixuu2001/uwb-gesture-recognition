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

## 2026-10-04T12:10:56.723906+00:00 原队列保持活动；累计十二条完整轨迹审计与私有备份通过

实际核对原训练598与归档607仍活动，未重启或重复启动。预注册fc19c823aec4c7d447527c9aa78d6c36da2354ca源码/helper/协议SHA保持不变。当前观测real_variance/held7第3轮，完成13/18；不按中途held结果调整配置或门槛。

新增real_variance/held6、permuted_variance/held6的ERM/tail验证及诊断logits独立函数推理逐位一致，角色、源样本顺序、选模审计通过。累计三臂各held3–6十二条、288轮、360产物SHA通过；前十条前向证据沿用PARTIAL_AUDIT_03且全部归档重新核SHA。held6三臂首轮权重和源校准完全一致。见PARTIAL_AUDIT_04.json。

当前备份25逻辑包、61个≤7MiB私有owner-only块、375291305压缩字节、399成员。复用50块、新传11块；所有块曾认证回下载且本次逐块/重组/全部成员SHA核验通过。含372训练成员（288段平均、十二份完整模型/AdamW/scheduler/RNG/EMA、样本ID/预测等）及27元数据成员。旧恢复证据仍由PARTIAL_DRIVE_RECEIPT_03等历史回执保存，本集合不重复纳入；不是18轨迹最终备份。最新索引PARTIAL_DRIVE_RECEIPT_04.json、监控MONITOR_04.json。

继续原队列，不启动新科学配置；完成后全24轨迹独立审计、最终私有备份，再依主tail联合门槛判断。归档/上传复用/workspace/uwb-fishr-final-20261004及/workspace/uwb-fishr-final-transport-20261004/PARTIAL_INDEX_04.json和records。用户目标仍未达成，尚无整组效果或原创性结论，自动接续保留。以下为历史记录。

## 2026-10-04T11:41:49.790126+00:00 第三次进程中断恢复与真实尾部平均重放通过；十条完整轨迹备份

确认旧508/521进程已消失、状态文件残留training后，保留balanced_erm/held6第22轮1650步完整目录到/workspace/uwb-fishr-interruption-03-20261004，核对版本/冻结源码/协议/九条完整结果和epoch段SHA、优化器/scheduler/EMA计数后，按原argv恢复训练598和归档607。具体终止原因未记日志，容器启动晚于原进程；未改参数或门槛、未重跑已完成轨迹。恢复后该轨迹完成24轮。另从保留的22轮真实检查点独立重放23–24轮54.851秒：model、AdamW、scheduler、RNG、EMA历史/计数/scale/校准、已非零的tail平均、best/选模和epoch段张量逐位相同；history仅允许墙钟与序列化SHA差异。固定scale5.9559378958416564不变。见RECOVERY_03、RECOVERY_REPLAY_03。

本次新增三臂各held5及balanced_erm/held6四条独立前向/角色/源顺序/选模审计；累计三臂各held3–5加balanced_erm/held6十条、240轮、300产物SHA通过。前六条独立函数推理沿用PARTIAL_AUDIT_02且全部归档重新核SHA；held5三臂首轮权重和校准也完全一致。固定本次备份集合，不按中途held分数改研究配置。见PARTIAL_AUDIT_03。

当前完整备份22逻辑包、52个≤7MiB私有owner-only块、317764690字节、339成员，全部认证回下载过，逐块/整包/成员SHA通过。复用30块、新传22块；包含310训练成员（240轮段、十份完整模型/AdamW/scheduler/RNG/EMA/ID/预测）、25元数据成员和4项本次恢复重放证据，含原第22轮完整checkpoint。此前部分备份的旧元数据与恢复包仍保留在旧回执，不重复列入本集合。索引PARTIAL_DRIVE_RECEIPT_03.json，工作索引/workspace/uwb-fishr-final-transport-20261004/PARTIAL_INDEX_03.json及records供后续复用。

当前real_variance/held6第14轮，完成10/18，训练/归档进程活动；见MONITOR_03。继续原固定队列，最终全24轨迹独立审计与剩余产物私有备份后再按主tail联合门槛判定；现无整组效果/原创性结论。进程不能假定跨环境启动存活，每次先核实际PID再恢复完整状态。原789源码和原数组未变，用户目标未达，自动接续保留。以下为历史记录。

## 2026-10-04T10:05:07.684386+00:00 第二次进程中断后恢复；累计六条完整轨迹审计和备份通过

云端环境启动后确认旧603/612进程均消失；队列文件仍training，但不能据此声称原进程活着。保留balanced_erm/held5第5轮375步全部文件到/workspace/uwb-fishr-interruption-02-20261004，核对冻结源码/helper/协议、Torch/NumPy、六条完成结果全部文件/段SHA，以及checkpoint身份、AdamW/scheduler/EMA计数和固定scale15.033946960215014后，按原argv恢复训练508和归档watcher521。具体终止原因未记入日志；当前容器启动晚于原进程。未重跑六条已完成轨迹，科学配置和门槛未变；上一次真实数据完整恢复重放证据RECOVERY_REPLAY_01.json仍有效，本次没有另声称新逐位重放。见RECOVERY_02.json。

新增real_variance/held4和permuted_variance/held4两条独立函数推理、角色/源顺序/选模审计通过。累计三臂各held3,4共六条、144轮、180产物SHA通过；此前四条独立前向引用PARTIAL_AUDIT_01且本次归档重新核全部SHA。held4三臂首轮段权重和源端校准完全一致。见PARTIAL_AUDIT_02.json。

当前备份集合14逻辑包、33个≤7MiB私有owner-only块、197926546字节、222成员，全部认证raw回下载过且本次逐块/重组/全部成员SHA重新核验；复用20已验证块，新传13块。包含186完整训练产物（144段+六份完整模型/优化器/RNG/EMA/ID/预测），24元数据成员和12个本次中断第5轮状态/段/恢复凭据成员。先前元数据和第一次恢复证据另保留在PARTIAL_DRIVE_RECEIPT_01.json；本索引不重复纳入，不删除旧备份。恢复入口PARTIAL_DRIVE_RECEIPT_02.json、MONITOR_02.json；不是整组18轨迹备份。

当前观测balanced_erm/held5第16轮，完成6/18，训练/归档进程活动。继续原固定队列，先检查实际PID再决定是否恢复，不能依赖跨环境启动的后台进程存活假设。归档/传输复用/workspace/uwb-fishr-final-20261004和/workspace/uwb-fishr-final-transport-20261004/PARTIAL_INDEX_02.json、records。完成后全24轨迹审计、补齐最终私有备份，再依冻结主tail联合门槛判断，禁止中途调参或把机制诊断当效果/创新。用户目标未达，自动接续保留。以下为历史记录。

## 2026-10-04T09:43:27.725266+00:00 云端进程中断后原状态恢复；四条完整轨迹审计和备份通过

本次确认原训练4303和归档4319均不在/proc，当前容器PID1启动晚于原实验启动时间；队列遗留training不能当进程仍活动。具体终止原因未写入日志。保留原4条完整结果和real_variance/held4第6轮450步全部文件到/workspace/uwb-fishr-interruption-20261004-0934，再核对版本、冻结helper/协议、全部完成产物和检查点SHA及AdamW/scheduler/EMA步数后，按原LAUNCH命令恢复PID603，归档watcher612；未改科学配置、门槛或重跑已完成轨迹。

用保留第6轮独立重放到第9轮（76.385秒），model、AdamW、scheduler、RNG、EMA历史/计数/固定scale/校准、best/tail/选模和逐轮段平均全部逐位一致；history除墙钟耗时和序列化SHA外一致，真实源数据而非仅合成测试。系数26.974675837012917保持不变；未读取held输入作调参。RECOVERY_01.json和RECOVERY_REPLAY_01.json保存证据。

固定审计集合：balanced_erm/held3,4，real_variance/held3，permuted_variance/held3，共4条96轮。全部ERM/tail验证/诊断logits独立函数逐位相同、角色/源顺序/选模和120产物SHA通过；held3三臂首轮权重与源校准完全一致。147个完整轨迹及元数据成员，加19恢复证据成员，合计166成员、149309022压缩字节、10逻辑包/25个≤7MiB私有owner-only块，全部认证raw回下载、逐块/重组/成员SHA通过。见PARTIAL_AUDIT_01.json、PARTIAL_DRIVE_RECEIPT_01.json。备份包含96轮段平均和四份完整优化器/RNG/EMA状态；恢复包另含中断第6轮、恢复第9轮比较及重放状态。不是18轨迹全部备份。

当前观测permuted_variance/held4第2轮，完整结果5/18，进程活动。继续同一冻结队列和原联合门槛，未得整组六源效果；不得根据中途结果调整lambda/EMA/warmup/seed/窗口或扩展八折。归档和后续上传复用/workspace/uwb-fishr-final-20261004及/workspace/uwb-fishr-final-transport-20261004/PARTIAL_INDEX_01.json、records，避免重复上传已有25块。若再次中断，先检查PID，再完整checkpoint恢复；最终全24轨迹审计和其余备份完成后判定效果。原789源码和原始大数组不动。用户研究目标未达，自动接续保留。以下为历史记录。

## 2026-10-04T08:17:09.211507+00:00 源梯度方差固定三组对照已启动；效果未判定

先冻结并正常推送/远端回读fc19c823aec4c7d447527c9aa78d6c36da2354ca，再实际启动PID4303。复用原云端数据/prepared和SWAD六条旧24轮基线；没有重复下载。真实源批成本含75%余量预计5.587792小时，低于8小时预算。新balanced_erm、real_variance、permuted_variance各六源轨迹×24轮，共18新轨迹432轮，最终连旧基线24轨迹独立审计。四训练用户每批16例的同一采样，首轮纯CE，第二轮首个源批冻结梯度范数比校准系数，常规时间偏差校正EMA.95；与作者原版不同，非原创宣称。

训练目录/workspace/uwb-fishr-source-runs-20261004；日志/workspace/uwb-fishr-source-train-20261004.log；归档watcher4319，状态/workspace/uwb-fishr-archive-state-20261004.json、目标/workspace/uwb-fishr-final-20261004。当前观测balanced_erm/held3第10轮；整组效果未知，不读取中途held结果做调整。合成全架构恢复、EMA/scale/RNG一致、留出隔离、高阶梯度、独立函数前向和三臂首轮/校准一致已测试通过。

已独立截取balanced_erm/held3第4轮300步完整原子快照，当前与best的验证CE独立函数重算一致，源角色/顺序、scheduler/AdamW步数、两套EMA/步数/固定scale核对通过。系数102.98310954123612（第76步校准，无截断），源端计算，非效果结果。完整model/AdamW/scheduler/RNG/best/tail/EMA及前4段平均等36成员，9380013字节2个私有owner-only块，认证raw回下载逐块/重组/成员SHA通过。见START_AUDIT.json、START_DRIVE_RECEIPT.json及START_MONITOR.json；只是启动快照，不是全实验备份。

继续时先检查上述PID/queue/checkpoint，禁止重复启动；代码/helper/协议保持冻结，只有确认死亡后按LAUNCH.json原命令完整状态恢复。主tail13–24固定，须同时对匹配ERM+0.5pp、置换+0.25pp、旧tail至少0pp，且各至少4/6正差、每位不降超1pp；次选模不得替代。未过停止lambda/EMA/warmup/seed/窗口网格与八折扩展；过门槛仅允许另行冻结八折/种子验证，不等于研究目标达成。完成后核对AUDIT/SUMMARY/归档、私有上传认证回下载，依最新远端父正常提交结果。watcher只本地归档，不自动Drive上传。论文主文未读、不声称完整复现或原创；用户目标未达，自动接续保留。以下为历史记录。

## 2026-10-04T07:34:12.302046+00:00 源梯度方差机制诊断4/6通过；不是分类效果结果

协议033a1521c89bb7d2daf09e17e40b601a6f66e48b先正常推送/远端回读，再成本估计771.851秒（含100%余量）通过900秒预算，最后才实际运行。源轨迹3–8的18个旧第1/6/24轮段平均，39168条源样本-模型配对，诊断105.681秒。每共同类别/训练用户48例，99次类内用户置换和独立两半稳定性；未用held/验证用户信号，也未更新模型。

第1和6轮共同通过的轨迹编号4、5、6、7，达到固定至少4/6机制门槛。编号指被排除的held，不是这些用户的测试分数。编号3、8第6轮完整比值分别1.716/1.674、两半余弦.247/.210，故失败，没有用第24轮替代。第24轮lambda1000*惩罚/源CE参考量级约2.30e-7至6.04e-6，与早期大相对差异可同时成立，不能直接当梯度范数或训练超参数建议。机制通过只允许设计后续对照，不证明泛化增益、因果关系或原创性；置换只是已拟合源样本的条件噪声参照，六重叠轨迹非独立确认。

独立NumPy中心化重算全部5346置换统计、各方差和门槛，源角色/固定抽样/两半分配、首尾批独立函数前向逐位审计通过（58.580秒）。完整源ID/features/logits/方差/置换值及代码/日志共42文件，23014610字节、4个私有owner-only块，认证raw回下载并逐块/整包/成员SHA通过；索引results/cloud_20261004_gradient_variance/DRIVE_BACKUP_RECEIPT.json。旧基线权重与优化器沿用SWAD/BASELINE_STATE已验证备份；原789源码和原始大数组不动。

现无新训练进程（诊断3886已正常退出）。下一步应另行实现并冻结同批次/采样/优化器ERM对照与源梯度方差干预，检查EMA归一化和完整恢复，源端确定早期生效时段/尺度，先高阶梯度、隔离、恢复和CPU成本测试，不能直接照搬作者1500步退火或根据held结果调lambda/时间点。文献主文仍未读；known-Fishr不是原创。详见最新RUN_REPORT、SUMMARY、ROWS、AUDIT和SOURCE_GRADIENT_VARIANCE_PROTOCOL。用户目标未达，自动接续保留。以下为历史记录。

## 2026-10-04T06:42:35.300456+00:00 源SAM整组完成，联合门槛未过；全部产物私有备份通过

12条新轨迹432轮，加复用普通24轮6条，共18条独立前向/角色/样本顺序/选模审计通过，678产物SHA；原PID1398及归档1410正常完成退出，无活动训练或待运行新实验。六源用户3–8的主tail：普通24为92.722222%，SAM24为93.113636%，普通48为92.896465%。SAM分别+0.391414pp（5/6胜）和+0.217172pp（4/6胜），未达冻结+0.5pp/+0.25pp联合增益门槛。不得事后降门槛、改主选模、做rho/decay/seed/window网格或据此扩展八折。不是SAM普遍无效的证明，也不是原创贡献或独立八折确认。

最终31逻辑包、85个≤7MiB私有owner-only块，519448048压缩字节、546成员（516训练产物+30元数据），全部认证raw回下载、逐块/整包/成员SHA核验通过；复用已验证70块，新传15块。包含全部432轮段平均及12个最终AdamW/scheduler/RNG状态、样本ID/预测/日志/配置；普通24基线沿用此前已验证SWAD/BASELINE_STATE备份。完整索引results/cloud_20261004_source_sam/DRIVE_BACKUP_RECEIPT.json，计算ANALYSIS.json、SUMMARY.json、AUDIT.json及FINAL_MONITOR.json。原789源码快照及原始数组不变，无原数组入Git。

下一项仅准备：已核查Fishr（ICML2022）作者README和指定源码区间，固定提交7b8fdf1e0b15226ded9b58efd37698e74e616ab7；主文未读，不声称全文综述。已独立实现softmax线性末层解析梯度方差核，合成float64逐样本autograd/高阶导数/有限差分/批重复尺度检查通过；没有复制作者代码，未在UWB上训练。见research/GRADIENT_VARIANCE_REVIEW.md、REVIEW_RECEIPT和CORE_TESTS，以及tools/linear_gradient_variance.py、check_linear_gradient_variance.py。普通Fishr也不是原创；先冻结源训练角色内的机制/有限样本零假设诊断，再决定是否值得真实干预。不能把合成核测试说成新分类效果或后台已启动。用户目标未达，自动接续保留。以下为历史记录。

## 2026-10-04T05:47:41.180650+00:00 新增四条源SAM/计算量对照轨迹审计与备份通过

原训练1398及归档1410仍活动，观测erm48/held8第1轮，11/12完整新轨迹。预注册4c4a4c6训练代码/helper/协议未改，未重复启动，未根据中途诊断结果调整科学配置；尚无整组六源效果结论。

本次新增sam24、erm48各held6,7四条独立前向审计通过：全部ERM/tail验证和诊断logits逐位一致，角色、源样本顺序、选模及168产物SHA通过。累计两臂各held3至7共10轨迹、360epoch、420产物SHA；前6条独立前向结论引用PARTIAL_AUDIT_03，身份一致且归档产物全部重新核SHA。见PARTIAL_AUDIT_04.json。

累计430完整成员，包括360轮段平均、10份最终AdamW/scheduler/RNG状态、配置、ID和logits，432806767字节、25逻辑包、70个≤7MiB传输块，全部私有owner-only上传并认证raw回下载，逐块、重组整包与全部成员SHA通过。旧42块复用。首批传输在23个新块通过后有两次接口失败；对已上传块复用记录ID，对未返回成功的上传先精确查名确认远端无文件，再续传剩余5块，最终全部核验通过。持久索引PARTIAL_DRIVE_RECEIPT_04.json；监控MONITOR_04.json。该集合不含正在运行轨迹当前检查点，也不是12轨迹最终备份。

最终归档和上传须复用/workspace/uwb-sam-final-20261004以及/workspace/uwb-sam-final-transport-20261004/PARTIAL_INDEX_04.json、records中的已验证块和ID。继续原固定队列，完成后全18轨迹最终审计、补齐私有备份，按冻结联合门槛判定。已知SAM操作本身不是创新，尚未证明用户目标；不启动新科学配置或八折。自动接续保留。以下为历史记录。

## 2026-10-04T04:15:53.658751+00:00 前三位源诊断用户两臂完整审计与私有备份通过

原训练1398及归档1410仍活动，观测erm48/held6第3轮，7/12完整新轨迹。未改预注册4c4a4c6训练源码/helper/协议，未重启或根据中途诊断结果调整任何参数/门槛；仍无整组六源效果结论。

本次新增erm48/held5的全部ERM/tail验证/诊断独立前向审计通过，54产物SHA及48轮样本顺序/角色/选模一致。累计sam24和erm48各held3,4,5共6轨迹、216epoch，252产物SHA；前5条的独立前向结论引用PARTIAL_AUDIT_02，身份一致且全部归档产物重新核SHA，未重复运行其推理审计。见PARTIAL_AUDIT_03.json。

累计258完整成员，包括全部216轮段平均、6份最终AdamW/scheduler/RNG状态和ID/logits，259692648字节、15逻辑包、42个≤7MiB传输块全部私有owner-only上传、认证raw回下载、重组整包与成员SHA验证。旧33块复用；新传输仅本次新增轨迹。索引results/cloud_20261004_source_sam/PARTIAL_DRIVE_RECEIPT_03.json，监控MONITOR_03.json。完整集合不包括正在运行轨迹的当前状态，也不是12轨迹最终备份。

最终归档和上传须复用/workspace/uwb-sam-final-20261004与/workspace/uwb-sam-final-transport-20261004/PARTIAL_INDEX_03.json、records中的已验证文件和ID。继续原固定队列，完成后全18轨迹最终审计，再补私有备份余下产物并按冻结联合门槛判定。不启动下一科学配置或八折；目标未达，自动接续保留。以下为历史记录。

## 2026-10-04T03:42:27.438568+00:00 五条完整源SAM/计算量对照轨迹审计及私有备份通过

原PID1398/1410保持活动，观测erm48/held5第30轮，5/12完整轨迹。预注册4c4a4c6训练代码/helper/协议哈希均未变，无重复启动、无中途参数或门槛调整，尚无整组六源效果判断。

固定本次审计集合：sam24/held3,4,5与erm48/held3,4共5条、168epoch。全部ERM/tail验证和诊断logits独立函数前向逐位相同，198产物SHA及角色、源样本顺序、选模通过。完整203成员（包含全部168轮段平均与五份最终model/AdamW/scheduler/RNG、ID/logits/配置），共204381081字节，12逻辑包、33个≤7MiB块，已私有owner-only上传并认证raw回下载，逐块/重组整包/成员SHA通过。原首条5个已验证块复用，未重复上传。它不含正在运行轨迹当前检查点，也不是全部12轨迹最终备份。

持久索引results/cloud_20261004_source_sam/PARTIAL_DRIVE_RECEIPT_02.json、审计PARTIAL_AUDIT_02.json、观测MONITOR_02.json。归档/workspace/uwb-sam-final-20261004和transport/PARTIAL_INDEX_02、records/可供最终备份复用。继续原固定队列；完成后全18轨迹独立审计、补齐私有最终备份，再按冻结联合门槛判定。原始数据无需迁移，789源码快照不改；目标未达、自动接续保留。以下是历史记录。

## 2026-10-04T01:59:56.962967+00:00 首条完整SAM轨迹独立审计和私有备份通过

当前云端原PID1398/1410均仍活动，未中断、未重启；观测erm48/held3第15轮，1/12完整轨迹。仍保持预注册4c4a4c6的模型/协议/helper哈希不变，不因中途分数改配置，尚无整组六源效果判断。

sam24/held3全24轮已完成，ERM/tail全部验证/诊断logits由独立函数前向重算逐位一致，30产物SHA、源角色/24轮精确样本顺序/选模核验通过。31个完整产物（含最终AdamW/RNG、全部segment、ID/logits）分2包共31241969字节、5个≤7MiB传输块，owner-only私有上传、认证raw回下载后逐块/重组/成员SHA全部一致。不是仅epoch2启动快照，也不是全部12轨迹备份。见results/cloud_20261004_source_sam/PARTIAL_AUDIT_01.json、PARTIAL_DRIVE_RECEIPT_01.json、MONITOR_01.json。

后续最终备份复用本次5块及2个不可变归档，不重复上传；本地索引/workspace/uwb-sam-final-transport-20261004/PARTIAL_INDEX_01.json及records保存持久ID与已验证下载路径。仍等原固定队列和最终18轨迹完整审计，之后私有上传余下全部产物，按冻结门槛处理。创新目标尚未达成，自动接续保留。

## 2026-10-04T01:34:18.058690+00:00 源端SAM真实云端训练已启动并保存恢复状态

预注册提交4c4a4c65696b1510f228504e1375de3223fedbc6正常推送且18文件远端逐字回读，通过后才启动PID1398；归档watcherPID1410。观测sam24/held3第4轮，0/12完整轨迹。运行根/workspace/uwb-sam-source-runs-20261004，日志/workspace/uwb-sam-source-train-20261004.log，归档状态/workspace/uwb-sam-archive-state-20261004.json；先核实时进程，禁止重复启动。12新轨迹=6SAM24+6ERM48，432轮，复用6普通24强基线，冻结主门槛不改。

初始epoch2/step148快照7,245,703字节28文件已私有owner-only上传并认证raw下载，整包与全部成员SHA通过，持久索引results/cloud_20261004_source_sam/START_DRIVE_RECEIPT.json。完整AdamW状态、scheduler/RNG/best/tail及两轮segment/顺序均保留；START_AUDIT验证全部验证logits/CE独立逐位一致，不评候选诊断。当前只是启动备份，完整结果/全部轨迹最终备份待做。

队列完成会自动全18轨迹ERM/tail独立推理/角色/选择/产物SHA审计，本地归档watcher会核门槛和打包516新训练文件；后续仍须私有上传与认证回下载最终所有包，不能把本地archive当远端。目前没有最终效果/创新结论，已知SAM不能称原创；主论文访问限制仍在。成本初5h未过，预注册前仅将资源预算改10h，原测量8.664782h含75%余量保留，不是调整效果门槛。原789快照/数据不变。目标未达，自动接续保留。以下旧状态是历史记录。

## 2026-10-04T01:28:26.382363+00:00 源端SAM实现与真实成本核查通过，待预注册后启动

固定known-SAM24对普通ERM48近似计算量对照，加复用普通24强tail。6源用户3–8，共12新轨迹432epoch；固定rho.05/lr.001/decay.001/batch64/seed2026，CPU1线程。主tail13–24/25–48；相对24普通+0.5pp、48普通+0.25pp、各4/6胜且每人不降超1pp才推进。不是创新主张，主文仍未读，不是旧八折确认。

独立NumPy解析梯度、相同Dropout/单次RNG推进、rho0普通优化等价、零梯度/异常恢复、两个臂完整AdamW/scheduler/RNG/tail恢复及留出输入标签隔离通过；真实普通首epoch与旧模型loss/order/段平均逐位吻合。源批估计含75%余量8.664782h，原5h资源预算未过，在预注册前只改为10h并保留原成本失败；没有改科学配置或评价候选诊断。

当前尚无新训练进程，下一步先正常推送冻结协议再启动。运行根/workspace/uwb-sam-source-runs-20261004，日志/workspace/uwb-sam-source-train-20261004.log。详情results/cloud_20261004_source_sam和research/SOURCE_SAM_PROTOCOL.json。原时间轴失败不重复，数据/789原快照不动。用户目标未达，自动接续保留。以下为历史。

## 2026-10-04T00:17:20.210865+00:00 时间轴跨度研究完整结束，联合门槛失败，最终私有备份通过

12新轨迹×24轮及6复用普通CNN基线全完，18轨迹ERM/tail全部验证与诊断独立功能前向逐位相同；534产物SHA、角色/24轮样本顺序/选模通过。主固定tail：shared128 92.722222%、time2 93.500000%、range2 93.369949%。time2相对普通+0.777778pp、4/6胜，但相对range2仅+0.130051pp、3/6胜，未过预注册+0.25pp且4/6胜的门槛。联合失败，停止本假设扩张率/核长/seed/window网格和八折扩展；不得去掉距离对照包装创新。六反复研究的开发用户、单种子不是独立确认，不与旧外层八折93.4417%直接比较。

原云端执行中断已按原代码从最后epoch11完整状态恢复，真实第12轮独立重放张量/loss/step逐位一致，前11完整轨迹没有重训。PID489/512已终止（观察为Z），当前无训练。正式288轮累计7528.122727秒，不含中断未完成工作及重放。详见results/cloud_20261003_temporal_scale/RUN_REPORT.md、SUMMARY/AUDIT/ANALYSIS、RESUME_AUDIT_01。

全部26逻辑包/62传输对象/380775626字节/410成员（372训练、29元数据、9中断证据）私有owner-only上传、认证raw回下载、逐块/重组/成员SHA通过。DRIVE_BACKUP_RECEIPT.json持久ID/URL/顺序完备，FINAL_RESTORE.md含恢复布局及旧基线备份依赖。25包ARCHIVE_RECEIPT的uploaded=false仅历史打包状态，另加第26中断包。数据和当前全部结果仍在/workspace，无需重复下载，原789快照不改。

下一项仅离线源端SAM文献/实现语义核查，作者README和选定JAX源码段、非官方PyTorchREADME/sam.py已读，主文未读；见research/SOURCE_SHARPNESS_REVIEW.md及receipt。第三方封装仅存wrapper状态会漏AdamW动量，合成恢复反例已记录，非UWB结果；不能推断旧研究错误或称创新。尚未实现、预注册、启动下一项真实训练。先补全文/排重与数值、随机掩码、优化器恢复、标签隔离和成本，再冻结前瞻协议。用户研究目标仍未达成，自动接续保留。以下带时间戳均为历史记录，旧运行状态不代表当前。

## 2026-10-03T23:59:38.483267+00:00 云端环境中断后从真实检查点恢复

执行通道断开，environment曾回到starting；恢复后旧训练21027/归档21039均不存在（不是仍运行）。11条已完成轨迹的全部产物和epoch段SHA通过，最后range2/held8留存epoch11、step814完整model/AdamW/scheduler/RNG/best/tail；源码/协议/helper及Torch/NumPy版本均与原identity一致。保存中断前快照/workspace/uwb-temporal-interruption-20261003后用原命令接续，新训练PID489、新归档watcher512。原LAUNCH及日志保留，新增RESUME_01/INTERRUPTION_01/ARCHIVE_RESUME_01。没有重训前11条。

从同epoch11状态在独立目录真实重放第12轮，平均权重张量、验证损失、step元数据与主接续结果逐位相同；文件序列化SHA不同但内容等值，不声称文件字节相同。RESUME_AUDIT_01.json通过，未做诊断推理或改参数。观察时最后轨迹第19轮；尚无整组六源结论。后续首先检查PID489/512与queue，不能再用旧PID判断活动。新增中断证据包interruption-recovery.tar.gz纳入最终私有备份。

## 2026-10-03T23:29:31.246125+00:00 时序队列持续运行，前8条完整轨迹私有备份已验证

原训练PID21027/归档watcher21039仍活动；观察时9/12新轨迹完成，当前range2/held7第10轮。未修改原协议、训练脚本或任何哈希依赖，无重复启动，无中途效果选择。保持固定队列完成后统一评估门槛。

两臂held3..6共8条完成轨迹：独立功能前向重放全部ERM/tail验证/诊断logits逐位相同，240产物SHA及192轮训练样本序列、角色/选模核验通过（120.44秒）。31文件/轨迹，共248成员，16逻辑归档249,848,180字节分40个≤7MiB块。已全部私有owner-only上传、认证raw fetch(include_base64=false)立即物化，逐块SHA、重组整包SHA、248成员SHA全部通过；最大并发3。它是前8条完整轨迹的备份，不是全部12条或正在运行轨迹的最新恢复状态，也不是效果通过。初始epoch2快照仍另存。

索引/凭据：results/cloud_20261003_temporal_scale/PARTIAL_DRIVE_RECEIPT_01.json、PARTIAL_AUDIT_02.json、MONITOR_02.json。持久ID/URL/顺序/字节/SHA都在回执。已验证本地包/workspace/uwb-temporal-scale-final-20261003；transport索引/workspace/uwb-temporal-scale-final-transport-20261003/PARTIAL_INDEX_01.json，逐块records/含远端ID、已下载路径与verified=true。后续全量备份必须复用相同SHA的这40块，补剩余轨迹和最终metadata包，禁止重复上传已验证部分。完整final watcher尚在waiting。

下一步先检查queue/proc/最终SUMMARY与AUDIT：完成后按原门槛处理，失败不搜dilation/kernel/seed/window；通过也仍需边界控制/八折/独立种子及创新证据。当前没有整组六源结论，用户研究目标未完成，自动接续保留。

## 2026-10-03T22:10:05.020638+00:00 时序研究运行完整性核验（没有整组效果结论）

训练PID21027、归档watcher21039均活动；观察时1/12新轨迹完成，当前range2/held3第22轮。保持原协议/训练源码与所有哈希依赖，不重复启动、不据中途诊断结果挑参数。首条time2/held3已完成并独立重放ERM/tail全部验证及诊断logits逐位相同，30个artifact SHA、24轮源顺序、角色及选模审计通过，耗时11.79秒。只证明完整性，不提供整组效果结论。

真实完成轨迹31文件已由现有归档器打包、重读成员SHA并验证重复调用不改变包；2包31,229,182字节位于/workspace/uwb-temporal-scale-final-20261003，可由最终watcher复用。目前这些完整轨迹包仅本地，未上传；不能混同已验证的初始epoch2私有备份。见PARTIAL_AUDIT_01.json、MONITOR_01.json。全文文献访问限制仍是package_managers预设，arxiv/openreview主文未读，不绕过代理或虚报已读。自动接续继续等原固定队列/最终审计，之后备份全部产物；用户目标未完成。

## 2026-10-03T21:54:35.487159+00:00 时序轴跨度研究：云端真实启动、初始恢复状态私有备份通过

预注册提交09239d6a54b82d22972b945c6a551c78ea0daf21已正常推送、15文件从远端拉回逐字校验。GitHub连接器读取间歇超时，原生git fetch/push可用；使用/workspace/uwb-temporal-publish-20261003独立detached工作树，最新父88455c7，无force，无原789快照改动。后续优先原生Git，但先fetch最新远端父；不要从旧工作树盲推。

训练PID21027实际存活，当前time2/held3完成epoch11，完成轨迹0/12。运行路径/workspace/uwb-temporal-scale-runs-20261003，日志/workspace/uwb-temporal-scale-train-20261003.log；每次接续先核对/proc与queue/检查点，禁止重复启动。归档watcher PID21039，state=/workspace/uwb-temporal-scale-archive-state-20261003.json；最终完成后仍需私有Drive上传并重新认证下载校验，不能把本地archive当远端备份。

初始epoch2完整model/AdamW/scheduler/RNG/best/tail、两轮segment、全24轮样本顺序及代码/协议等26文件，7,245,346字节档案，已在私有Drive owner-only保存并认证下载逐SHA校验；START_DRIVE_RECEIPT.json含持久ID与每文件SHA。这只是初始快照，不是完整12轨迹备份。START_AUDIT独立功能前向重算全部验证logits及CE逐位一致，训练角色/样本序列一致；未读取诊断输入作评测，尚无新准确率。用户创新/完整验证目标仍未完成，自动接续保留。

## 2026-10-03T21:46:03.516506+00:00 时序轴跨度干预：已实现/已验证，等待预注册提交后启动

- 最新已完成组DRO负结果及完整备份不变，不重复其配置。
- 新假设仅是当前小CNN的轴向卷积采样尺度：time2对range2和原CNN。三臂265836参数、相同初始张量/RNG；只改每块第二stride1卷积的dilation/padding。纯卷积跨度71x43/43x71/43x43，不是完整网络RF，不是新算法或完整TCN复现。
- 已补读作者TCN README/tcn.py、旧vendor全部及Classifier调用。主文仍未读，旧模型已有全序列注意力，不能说首次长时序建模。
- 合成断点恢复、留出输入/标签扰动隔离、功能前向逐位一致均通过；真实普通CNN首轮loss/order/segment权重逐位复现旧基线。CPU估计3.196h含75%余量，预算4h内。成本探测不提供效果证据。
- 固定研究TEMPORAL_SCALE_PROTOCOL.json：源3–8四训练/一验证/一诊断，12新轨迹各24轮，复用6旧轨迹。主结果fixed-tail13–24，验证CE选模仅次结果。时间臂vs基线需+0.5pp、vs距离臂+0.25pp，各至少4/6胜且每位差值≥−1pp。失败停止扩张率/核长/seed/window搜索；成功也须边界控制、多种子和同协议八折，不直接称创新。
- 运行路径/workspace/uwb-temporal-scale-runs-20261003，日志/workspace/uwb-temporal-scale-train-20261003.log。此记录生成时正式训练未启动；必须以之后LAUNCH/PID和检查点证据为准，避免重复进程。数据无需下载。

## 最新：正则化组DRO完整负结果及私有归档（2026-10-03T20:41:33.167710+00:00）

18轨迹/432轮全部完成，训练PID19340与归档PID19639均已结束（僵尸不是后台活动）。六源用户3–8主验证CE选端点：uniform_base91.5821%、uniform_reg91.5821%、dro_reg91.5391%；旧强tail92.7222%。主DRO对两个匹配对照均−0.0429pp、2/6获益；对旧tail−1.1831pp、最大单用户下降4.5pp，全部冻结门槛失败。停止本decay/eta/种子/窗口网格，不扩当前八折，不用次要tail92.8434%替换主失败。仅开发六源，不是旧LTR八折或独立最终测试。不能由一个固定设置否定所有正则化DRO。

完整18轨迹ERM/tail全部验证与诊断logits独立函数重放逐位相同；954产物SHA、训练角色/样本顺序/逐步q/验证选模审计通过，432轮累计11759.94秒。37逻辑包563,083,660字节，972训练文件+25元数据共997成员，91个≤7MiB传输块均私有owner-only，全部认证回下载后分块/重组整包/每成员SHA一致。间歇上传Internal error逐项查无远端对象后重试，未重训或改选择。见results/cloud_20261003_regularized_groups/RUN_REPORT.md、DRIVE_BACKUP_RECEIPT.json、FINAL_RESTORE.md。数据/prepared/旧源tail另依旧已验证备份；当前磁盘齐全，不重复恢复。

下一项仅准备时序卷积尺度核查：InceptionTime作者README、完整inception.py/nne.py/receptive.py及旧多视角03_model.py/时间诊断报告已读，主文和图片未读。默认作者40/20/10多尺度核、6模块、5初始化集成都是已有技术，不是原创。当前CNN纯卷积跨度43但GN统计/池化/头有跨时段依赖，禁止声称网络只看43帧；旧双路径另有90时间token。旧温和时间增强、DTW/15阶段监督失败不能换名重做。见research/TEMPORAL_SCALE_REVIEW.md及blob回执；下一实现/预注册/分类实验尚未启动，不能称后台运行。先查旧vendor/调用等价性、明确同容量对照和源内预算再决定。用户创新/完整八折目标未达，自动接续保留。


以下为历史记录。

## 2026-10-03训练期间审计与自动归档（2026-10-03T17:02:57.606602+00:00）

原训练PID19340活动，3/18完成；当前uniform_base/held4第17轮。首先看实时queue_status和进程，不能根据历史观察重复启动。尚无整组六源结果，未调整冻结模型、采样、参数或选择规则。

前三条完成轨迹（held3全部三臂）独立函数前向重放所有验证/诊断logits逐位一致，159个产物SHA、所有源训练ID和逐步q重新构造通过，31.09秒。仅完整性审计，不挑选中途效果。见PARTIAL_AUDIT_01.json。

后处理代码af96196f102dc70f4f1948d3082ff4df67f316ab，tools/archive_regularized_groups.py已提交回读。真实单轨迹54文件（24段权重、24步序列文件、完整optimizer/RNG及最终预测等）打包/重读SHA和不可变复用通过，损坏/源变更/危险路径/未完成研究拒绝检查通过；2包31,277,041字节只在本地，尚未上传。损坏拒绝测试最初未捕获tarfile.ReadError，修正测试异常类型后通过，训练无影响。完整18轨迹finish分支须等队列结束后真实执行，不能声称已通过最终归档。

自动归档PID19639实际处于waiting，每45秒观察训练，8小时上限；状态/workspace/uwb-regularized-group-archive-state-20261003.json，日志/workspace/uwb-regularized-group-archive-20261003.log。训练complete且全部18轨迹审计通过后，独立复算三对照分数与原冻结门槛，保存全部972训练文件（432段权重、432步序列、18完整恢复状态等）以及元数据。输出/workspace/uwb-regularized-group-final-20261003，只写本地，不自动上传Drive。既有源基线仍依SWAD历史备份。

接续先检查两个PID/状态；完成后私有Drive上传最终所有包、认证重下载校验整包/成员SHA并提交真实报告。归档器不会启动训练或修改阈值，失败保留现场。训练中的源码/协议及其哈希依赖不得改。最终创新/八折目标未达，自动接续保留。


以下为历史记录，以时间较新的观察及实时状态为准。

# 最新：二阶源先导完整失败（2026-10-03T12:10:20.979142+00:00）

021bd332afcbc311b19dbbe384bc81f904c7295e先冻结后真实执行，六源四训练/一验证/一诊断，共72候选岭、24选定读出。原编码器固定第24轮段平均、48→16固定Gaussian QR投影。grid92.8586%、diag86.1705%、cov89.1124%、主sqrt89.2879%、同第24轮CNN92.7083%、旧尾平均CNN92.7222%。sqrt比diag+3.1174pp但比grid−3.5707pp、比tail−3.4343pp，联合门槛失败；不扩投影/维度/lambda/种子/八折。不能换raw或挑HV6包装成功；不是旧完整八折或端到端iSQRT-COV复现，也不证明所有二阶信息无效。

数值15例cov/SciPy sqrtm（含零和秩亏）、投影正交、空间置换、3组独立岭及角色隔离检查通过，奇异输入SciPy警告保留。实际每折train/val/diagnostic首32样本独立卷积/协方差/矩阵根通过，72候选全量验证与sklearn重拟合一致；后审计全部ID/24文件SHA/选择/分数/门槛通过。完整保存预测、ID、选定权重与scaler、投影、配置/日志。六轨迹106.28秒，CPU1线程，未重训encoder；无训练/适配进程。

results/cloud_20261003_second_order为完整报告。7包13,043,101字节47文件已私有Drive认证回下载，整包和成员SHA一致；回执在DRIVE_BACKUP_RECEIPT.json。原第24轮checkpoint依SWAD旧私有备份恢复，本包不重复存原大encoder；数据/prepared仍在云端，禁止重复下载/覆盖。

下一项只读Tent作者README+tent.py、SAR README+sar.py，主文与sam.py未读完。Tent原版只BN不适配本GN模型；SAR支持GN，但Dropout、空筛选、reset和更新前输出需明确处理。旧new_routes校准使用带标签support，与无标签在线更新不同。见research/UNLABELED_ADAPTATION_REVIEW.md及blob回执。未实现/运行TTA，不声称原创或成功；先补SAM/许可/全文/排重，冻结无标签API、源内验证、先预测后更新、顺序/批次、用户重置及成本。不得用真实目标类别分批或未来批数据，不可把此新部署设定暗称固定模型同协议提升。研究目标仍未完成，自动接续保留。以下历史运行状态已过期。

# 最新：配对源诊断完成未过门槛（2026-10-03T11:27:52.753223+00:00）

预注册ef52f195874bde2e4d265aef5be22f98973a5e5f后完成源配对诊断；没有训练/诊断进程运行。旧六源模型固定第24轮段平均，累计28400训练样本行（重叠源集合，不是28400独立样本），两置换种子×三雷达×同/跨用户得到72读数。L/T/R同用户翻转0.0683/0.1027/0.0788%，跨用户0.2846/1.5904/0.3067%；最高4.0966%。三路均0/6满足冻结5%翻转且较同用户多2pp、两种子同时成立的规则；停止本配对增强候选，不能看Top敏感再挑增强强度。不报告测试准确率或因果用户风格结论。

耗时55.21秒，真实CPU成本预估285.63<900秒。全部源池双射、无固定点、类/用户条件、池外标签隔离和全量identity/joint控制通过；14条件×64样本×6模型的独立原信号功能前向最大logit误差实际0，非全量原信号重放。另一NumPy统计核查及POST_AUDIT对72翻转/门槛/ID/产物SHA通过。results/cloud_20261003_pairing保存报告、模型来源、统计、审计；私有7包19,335,527字节22文件认证回下载和成员SHA一致。原模型依SWAD旧备份恢复。数据/数组不用重新传输。

曾考虑单雷达特权蒸馏，查原OOF_PRIVILEGED_KD_PLAN等发现普通KD、去教师和OOF都已失败；不重复。下一项仅阅读CVPR2018 fast-MPN-COV作者README/完整MPNCOV.py、pyRiemann README/完整tangentspace.py，主文未读完，准确范围和blob见research/SECOND_ORDER_REVIEW.md及receipt。协方差平方根池化与Riemannian切空间不同，都是已有方法，不称原创。尚未实现二阶实验。先核查旧矩/encoder池化内容再冻结源内控制、投影维度、数值测试和CPU成本，严禁全数据参考或默认测试时适配。目标仍未完成；自动接续保留。以下为历史记录，旧运行状态已过期。

# 最新：共享干预完整失败，私有备份已验证（2026-10-03T10:52:25.285134+00:00）

12新轨迹+6复用基线全部完成；PID16830/17076为历史进程，当前没有训练运行。完整18轨迹的ERM/tail功能式logits重放逐位相同，522产物SHA和角色/顺序/选择核验通过。主固定尾平均：共享128 92.7222%，独立128 92.6048%，近参数共享177 93.0947%。独立相对两对照−0.1174/−0.4899pp，1/6与0/6用户改善，冻结门槛失败。停止该假设宽度/窗口/种子网格及八折扩展；六开发源先导不能与旧八折93.4417%直接比较，不能证明所有共享或梯度方法优劣。研究目标仍未完成。

完整25逻辑包490,011,185字节的360训练产物+29元数据已私有Drive上传并认证回下载，整包及389成员SHA全部验证。14整包+32个≤8MiB传输块（重组其余11包），共46对象，shared=false/owner-only。大包工具超时已通过精确分块解决，没有改模型。持久ID、顺序和SHA见results/cloud_20261003_view_sharing/DRIVE_BACKUP_RECEIPT.json，恢复见RESTORE.md。原六共享baseline另依SWAD七包+本目录三个optimizer补充包恢复。当前磁盘齐全，不重复恢复/训练。旧ARCHIVE_RECEIPT的uploaded=false仅为打包快照。

下一项是片段级配对依赖准备，未启动新真实模型实验。research/PAIRING_DEPENDENCE_REVIEW.md记录Geodesic Multi-Modal Mixup作者代码实际阅读范围，另外两库仅README，未读完整论文；现有Mixup不是创新。原B-SHUF是teacher目标重排，不能等同新雷达输入配对诊断。原global derangement罕见单固定点fallback强制反例输出[2,2,3,1]（非双射），没有证据历史真实运行触发；原789文件不改。新source_pairing.py的源池双射/跨用户可行性/池外标签隔离检查通过，只是诊断准备，不是准确率增益。

接续先冻结源训练池内固定检查点、实例/用户重配对、共同置换控制和推进门槛，再测成本并真实审计。禁止用诊断/测试标签构造训练配对，禁止把synthetic跨人组合作为真实同步、3D或因果证明。保留自动接续；未达到用户目标。以下均为有时间戳历史记录，旧“运行中”不代表当前状态。

# 最新运行核验与自动归档（2026-10-03T08:51:57.104Z）

原训练PID16830仍活动，4/12新轨迹已完成，当前untied128/held5第14轮。先查实时进程与queue，不重复启动；当前没有整组六源用户效果结论，不按中途分数调整配置。首次三个完成轨迹（held3两臂、held4独立臂）独立重放全部验证/诊断logits逐位一致、源ID/选模/87文件哈希核验通过；这只是部分审计，完整18轨迹最终审计仍由原训练队列执行。

新增纯后处理工具tools/archive_view_sharing.py，提交9bcd3fd69af97b9f4e43bd8ee2a33b7cea32be6a，不改训练及其哈希依赖。自动归档PID17076已实际启动且处于waiting，每45秒监视原训练，最多8小时；状态/workspace/uwb-sharing-archive-state-20261003.json，日志/workspace/uwb-sharing-archive-20261003.log。训练complete且原完整审计通过后，独立重算分数/冻结门槛，生成描述性区间，并把12新轨迹全部360文件（含288段权重、12个完整optimizer/RNG状态）分成<32MiB包，输出/workspace/uwb-sharing-final-20261003。只写本地，**不自动上传Drive**。

已用一个真实完成轨迹30文件检验分包和SHA/不可变复用；已有两个本地包40,916,990字节会直接复用，并非完整最终归档或已上传。修改源、损坏归档、危险路径拒绝检查通过。证据见results/cloud_20261003_view_sharing/{ARCHIVING.md,PACKAGING_CHECKS.json,PARTIAL_AUDIT_20261003_01.json,MONITOR_20261003_02.json}。原训练继续；无需重复跑成本/单元检查或另开模型。后续首先看两个PID和归档状态；完成后将全部包私有上传并认证重下载整包/成员SHA，然后发布完整真实报告。门槛失败停止本共享假设扩展，通过仍不等于创新成立。

以下保留启动时记录；带时间戳的旧完成数/epoch以最新实时状态为准。

# 当前接续：共享参数直接干预已启动（2026-10-03）

**云端活动训练已核实。** 预注册fd4d754b57f2aa2787d7d896eee9e26828cd234e；PID16830，/workspace/uwb-sharing-source-runs-20261003，日志同前缀.log。观测2026-10-03T08:11:55.494Z时进程R、held3/untied128第19轮1406步检查点，完成0/12新轨迹。先查实时queue_status.json、进程和progress.pt，禁止重复启动。此为有时间戳的观察，不保证以后仍活跃；后续状态以实时文件为准。

先读research/VIEW_SHARING_PROTOCOL.json以及results/cloud_20261003_view_sharing/{RUN_REPORT,TESTS,COST,LIVE_OBSERVATION,START_DRIVE_RECEIPT,BASELINE_STATE_DRIVE_RECEIPT,RESTORE}.json/md。代码tools/view_sharing_source.py、view_sharing.py、audit_view_sharing.py、test_view_sharing.py均已入Git。

直接检验解除共享是否有收益：shared128(265836参数)复用既有SWAD源先导的ERM/固定尾均值；untied128(351788)和shared177(351145)各六个源3–8轨迹，24轮，288新epoch。每轨迹下一源用户验证、其余四用户训练，诊断用户只在全部权重选择完成后评分，不新分析用户1/2。主结果固定13–24轮逐更新平均；ERM次要，SWAD失败窗口不再进入选择。两个新臂依次按held交错跑，不依据中途诊断结果改变剩余配置。

初始untied与shared128真实batch64 logits逐位相同；独立前向、独立参数存储、合成数据断点续训/诊断标签及输入隔离检查通过。新共享训练器真实首轮指标和所有段权重逐位复现旧基线，因此不另训六条共享128。源码/环境/数据/基线文件哈希匹配。估算含75%余量3.179小时通过预设4小时成本门槛；实际新训练每轮约23秒(初始观测)，最终总时长待统计，不是效果结论。

固定推进门槛：untied尾均值对shared128平均>=+0.5pp、对shared177>=+0.25pp，分别>=4/6用户正增益，最差用户分别不下降超过1pp。任一失败则不调宽度/窗口/种子、不扩八折。通过才考虑同协议八折和共享位置消融；独立encoder本身已有，创新目标仍未达成。不能用ERM临时替换尾均值主结果，也不能把源六折当旧LTR八折。

队列完成自动写AUDIT.json、SUMMARY.json；独立函数式复算18轨迹×2选择器的验证/诊断logits，并核对所有ID、选择和哈希。确认审计通过后才报告六源用户最终对照。若进程退出失败，先读错误和进度，仅修复确切实现问题、记录变更；原命令可从每轮model/optimizer/RNG恢复，身份不匹配不能强行复用。精确命令见LAUNCH.json和RESTORE.md。训练中不要改这些脚本、协议及哈希依赖。

私有启动证据包16,462,020字节，30内容文件，包括代码、成本首轮复现和untied第3轮早期续训状态；文件1wWxDl1HoFgx8o9Nv4BBBfxim-h1pxHmI，SHA af87c05f2a8ae7486ebc614f576425fb3687f339d720f00359d75adcf602099c。已认证重下载逐文件核验，**并非12轨迹最终备份**。另外补存六条旧SWAD基线最终progress.pt（原七包未包含优化器）：三个包共89,012,045字节，六原文件SHA与旧result.json完全相同，均仅owner/认证重下载核验。恢复按新RESTORE.md将三包解压到原SWAD恢复目录/runs，原七包仍必需。当前已有数据/基线完整，勿重复下载。

后续须把12新轨迹的完整检查点（含optimizer/RNG）、每轮段模型、预测、日志/配置/样本ID及独立审计按<32MiB私有包保存并重下载SHA核验，再更新GitHub状态/记忆/真实结果。已有基线六模型不能冒充新训练。保持每小时自动接续，仅实质变化通知；不要仅轮询不变状态时写重复进度。

以下是历史记录，所有“无活动训练”“尚未实现”均已被本节更新。

# 当前接续：视角梯度冲突筛查通过，尚无识别干预结果（2026-10-03）

**无活动训练。** 先读results/cloud_20261003_view_grad/RUN_REPORT.md、SUMMARY.json、POST_AUDIT.json、NUMERICS_REPAIR.json、DRIVE_BACKUP_RECEIPT.json、research/VIEW_GRADIENT_REVIEW.md和VIEW_SHARING_INTERVENTION_PLAN.md。原预注册636cf5dc；float32数值核验失败后，4742cd3c仅提高到float64并按实际次数核算成本，没有改研究阈值。不要重复运行已完成诊断。

复用六条源3–8轨迹的固定第1/12/24轮内密集平均模型，非验证选定权重；每条轨迹仅四个实际训练用户各96条，2035唯一ID、每时点2304次ID使用、三时点6912次呈现。18检查点216批，不评分相应验证/诊断用户，不分析1/2。持续冲突门槛6/6轨迹通过，单路>=80%范数主导0/216批。三时点冲突分别39/72、57/72、62/72批。是单一CE的视角链式梯度，不是三个独立任务；负内积不证明伤害或泛化改进，不能直接套PCGrad/OGM。

全部216梯度和相对误差最大1.10e-15以下；18首批独立encoder副本梯度/logits逐位一致；后审计18模型SHA、源ID、所有汇总及18首批Gram重算通过。并非216反向全部独立重跑。实际369.20秒不含输入SHA/失败/后审计，估计422.99秒低于600秒。首次float32失败及修复证据均保留。结果/workspace/uwb-view-grad64-20261003、日志同名前缀.log，无新权重。

24,851,904字节26内容文件归档已私有Drive认证重下载核整包和逐内容SHA：文件1rGW6bXazHbQdxipd2GsDkkD_ILeJWoov，仅owner，SHA=66aaaf134d842eedf084fd4813c68a41defcb6cda63e0a58df95c3e8704938d4。含逐批Gram/ID、18首批梯度、源码、失败证据、OGM补充PDF；原18段模型按SWAD私有回执定位，不重复打包。恢复用新空目录，审计--models支持路径映射并保留原索引哈希。

新文献实读：PCGrad README/TF源码，OGM-GE README/训练代码及作者公开4页补充文字/公式/图注，MMPareto README；主论文未读完、图形未视觉核验，不夸大综述完整性。源共享参数抵消与OGM的CE残差抑制、MMPareto的多损失冲突有区别。

下一项具体计划：共享encoder128头、三个独立encoder128头、共享encoder177头的源协议对照。预计参数265836/351788/351145，先实际核数、初始logits等价、匹配24轮训练与固定13–24轮平均强对照、成本门控，再冻结并启动。此项尚未实现/启动；独立encoder本身也不是原创。研究贡献目标未达成，自动接续保留，仅实质变化通知。

以下为历史记录，以本节和实时进程为准。

# 当前接续：活动区间诊断完成，能量裁剪筛查失败（2026-10-03）

**无活动训练。** 先读results/cloud_20261003_event_extent/RUN_REPORT.md、SUMMARY.json、POST_AUDIT.json、DRIVE_BACKUP_RECEIPT.json和research/TEMPORAL_EXTENT_REVIEW.md。协议/实现先于真实运行提交856951f8db8d160019a2e7958897b0ddc3d8b58a，不重新启动。

仅源用户3–8共7100条，标签只用于质量掩码。旧幅度加权绝对相位变化/幅度变化两个代理的联合90%质量最短连续区间，各用户中位需73–79原帧。区间≤60帧的用户等权比例Left3.8131%、Top4.4672%、Right6.1199%；三路均0/6用户达到预定50%比例，门槛需4/6。停止这项90%/60帧能量裁剪假设，不扩质量/长度/代理网格，不说明所有时序模型都失败，更不能把能量当真前景/背景。

20.60秒不含输入SHA/后审计；首16条活动提取SciPy独立核对最大误差8.89e-16以下。全部63900单/联合区间另一种穷举窗口实现重算、7100唯一ID、汇总与代码/协议SHA通过；不是全7100原活动提取独立重算。结果/workspace/uwb-event-extent-20261003，日志同名前缀.log，无新训练权重。

28,758,567字节归档（11内容文件）已私有Drive认证重下载核整包及逐内容SHA；文件11SsO13_3NZ5amLq0Q6xutzpOGSW3NKyp，仅owner，SHA=abeb6df54b81208d771bd0897dfa5f5c47f084130107c7125640b0805df2d93b。恢复用新空目录，详见RESTORE.md，不重复下载现存结果。

排重发现：旧15段活动/进度监督已失败，不换名重训；旧B-SHUF68.42%配对未变导致机制结论已撤回，修正100%版本没有重训，不能当现成证据。W-TALC ECCV2018本轮只读作者README/train.py/test.py，不称论文全文已读。基于标签的时段注意力/同类约束已存在，不作原创主张。完整90帧去均值/标准化后的特征不能直接截前缀宣称因果早期识别。

下一不同问题：轻量CNN三路共享编码器的视角梯度是否冲突/一支主导。先核查旧代码与多模态优化近邻，再考虑仅源训练行固定模型的梯度诊断；负内积不证明泛化受损，已有梯度平衡算法不算创新。本轮未实现/未启动该项。研究目标仍未达成，自动接续保留，仅实质变化通知。

以下为历史记录，以本节和实时进程为准。

# 当前接续：单距离单元位移诊断完成，未触发抗混叠训练筛查（2026-10-03）

**无活动训练。** 先读results/cloud_20261003_shift/RUN_REPORT.md、SUMMARY.json、AUDIT.json、POST_AUDIT.json、DRIVE_BACKUP_RECEIPT.json及research/ANTIALIAS_REVIEW.md。预冻结提交6ebb2e4f先于本轮运行。复用源3–8六个ERM模型（各四训一验证一诊断），7100条、42600个单雷达±1当前距离单元扰动，不评分真实标签，不新增训练。

边界损失≤1%子集的用户等权翻转率：Left0.9912%、Top1.1044%、Right0.7680%；分别14200/13995/13778扰动对。各用户覆盖≥200，但达到5%翻转门槛的用户全部0/6，均未过筛查。停止扩大本位移网格，不据此启动抗混叠训练；不证明不存在混叠/更大位移鲁棒，也不代表准确率高或移位物理上保持标签。

实际76.33秒，成本估计82.49秒低于300秒预算。六模型完整未移位logits逐位重放旧缓存；各首64条六种移位的独立函数式前向最大差0。全部42600边界比/JS/翻转统计独立重算，JS差≤2.47e-16；7100唯一ID与六预测SHA通过。不扩大到全部移位前向都独立重算的声明。

新归档2,920,305字节，私有Drive文件1nw3runGawSgSCnvYw8AMYfxxKTwtAH62，仅owner；认证重下载包SHA及21个内容SHA均验证。SHA=b5839c8594a22c9e4bbc8effa7792c4420d48fb17f133fe35b2a969f1b48395d。原权重不重复打包，以SWAD私有回执及本轮AUDIT逐权重SHA定位。新结果/workspace/uwb-shift-diagnostic-20261003，恢复新空目录，详见RESTORE.md。

文献实读：ICML2019 Zhang作者camera-ready主文§1–5文字/公式/表格/图注，及BlurPool README/源码；未读全补充材料或逐图视觉核查。APS CVPR2021仅作者README。BlurPool、单轴低通、自适应多相采样均已存在，不包装成创新；零填充、GroupNorm、位置敏感头使当前预测变化不能因果归为混叠。

下一不同问题：先核查旧代码与近邻文献的动作起止/静止背景处理，再诊断源用户运动事件占比和时段分布。不重复失败DTW/时间扭曲/距离矩路线，不预设裁剪保留类别语义；这项尚未实现/启动。研究目标未达成，自动接续保留，仅实质变化通知。

以下为历史记录，以本节和实时进程为准。

# 当前接续：距离高低频源内读出完成、互补门槛未过（2026-10-03）

**无活动训练。** 先读results/cloud_20261003_range_frequency/RUN_REPORT.md、SUMMARY.json、AUDIT.json、POST_AUDIT.json、DRIVE_BACKUP_RECEIPT.json及research/RANGE_FREQUENCY_REVIEW.md。54个闭式线性读出已全部结束，不重复启动。冻结代码/协议提交4a52e791e9d5c8e75c36eac8f5d9010bcb5183be先于真实评分。

仅源用户3–8共7100条、五训练/一诊断，固定截止21和256维投影；不是外层八折。低频/高频/混合各视角成绩：Left55.8939/21.5707/54.8030%，Top55.2929/22.2285/55.8687%，Right53.8510/31.7323/53.1793%。只有Right高频过信息门槛，但三路混合均未过互补门槛。Top混合+0.5758pp、3胜1平2负；浮点1e-16不能把持平计胜。停止该配置截止/alpha/投影种子网格，不据此启动全八折。不将阴性结果解释为高频只有噪声。

54模型与sklearn独立拟合一致，162保存产物SHA、7100ID、源标准化重算通过；真实16条独立余弦特征最大差9.05e-14。全特征/模型/存盘/内置审计18.49秒，另有输入SHA与post审计开销。结果目录/workspace/uwb-range-frequency-20261003。228内容文件归档8,641,948字节上传私有Drive文件1H_4mskEocuXioLCyMbIsjyMhCbVWgcMM，目录1QW19MPD9rKbwkSBoUoH1X5ZUZ_K5YZ6t，仅owner；认证重下载整包SHA及逐内容SHA均验证。SHA=3b46765ed21798e58d389aea0212992d1f787fb1bb7ef3d9a9342c92489c1314。含54权重/预测/分割与必要源码，不含可重建特征缓存/原始数组，恢复到新空目录，详见RESTORE.md。

新增文献实读：CVPR2020 HFC作者README/频率代码、作者2021博士论文第2章完整正文/表格/图注（印刷5–16页）。学位论文PDF从作者公开GitHub仓库下载并核blob与SHA，非通过受限域绕路。没有读整本学位论文或独立CVPR全文，也未做图片视觉核查。作者频带训练/原图测试结果不能直接当本诊断同频带训练/测试证据；归一化与高频关联也不证明雷达机制或原创性。

下一不同问题：现有CNN的步长下采样是否对小距离位移产生混叠敏感性。先核查抗混叠CNN文献和原代码，再冻结源信号/冻结源模型的敏感性诊断；不能预设所有距离移位保持标签，不能用外层测试反馈选择。该项尚未实现/启动，不继续失败线性频带网格。研究目标未达成，自动接续保持，仅实质变化通知。

以下为历史记录，以本节和实时进程为准。

# 当前接续：距离压缩源内保真诊断完成（2026-10-03）

**无活动训练。** 先读results/cloud_20261003_range_compression/RUN_REPORT.md、SUMMARY.json、AUDIT.json与research/RANGE_COMPRESSION_REVIEW.md。冻结脚本/协议先于真实诊断提交7daabafe214c6d271c36a91cd60bd3dccbe894ed。原SWAD及更早队列均已结束，不重启。

仅源用户3–8的7100条信号，用户1/2不参加分析。现有90×63输入用块均值/DCT压至21/7单元的四臂均未过预冻结保真门槛；DCT21唯一失败项是Left幅度时间差分能量中位数83.33%低于90%。源内类别原型差异保留高，不等于跨用户识别有效；被丢弃部分可能是细节或噪声。本轮无分类拟合/新测试准确率/部署时延结论，不修改门槛扩大同一压缩率网格。

诊断15.60秒。独立公式、投影性质、真实首16条对照及7100源ID、340800逐片段比例/所有中位数审计通过；原型差异统计未另独立重算。完整per_clip.npz在/workspace/uwb-range-compression-20261003，SHA256=5d2a53660ad3d3717cd8bb705a32d0a800ac1ffa81eda1fa17ecb2db65d84393；二进制未另做远端备份，可由冻结脚本快速重建，勿误报已经Drive备份。

旧pipeline已有9倍时距均值池化、model已有双路径/池化；普通压缩/时距分离不能作为新贡献。新增第三方FMCW轻量项目与TinyRadar-OL仅核查README及原代码，不是论文全文核查，不将其非同协议分数作本项目比较。

下一可否决问题：Left幅度被丢弃的距离高频部分是否有跨源用户可重复的动作信息？先核查近邻，再冻结源内留用户、高/低频匹配容量与固定超参数对照，不使用外层测试标签选择。视角分配不同分辨率仅是假设，尚未实现/启动，不把能量差异直接包装为创新。用户研究目标仍未完成，自动接续保留，仅实质变化通知。

以下为历史记录，以本节与实时进程为准。

# 当前接续：SWAD源先导完成、门槛失败（2026-10-03）

私有备份已完成并认证重新下载核SHA：Drive目录1f0ETtgSnVcDt5zkuTFka_8mdOQsjkEfH，7文件/162,568,884字节，全部shared=false且仅owner权限。持久下载URL、逐包SHA及大小见DRIVE_BACKUP_RECEIPT.json（报告同目录）。恢复应下载到独立空目录后先核SHA，勿覆盖原结果。

**无活动训练。** 原PID14774为僵尸；六条24轮轨迹与18个选定模型全部完成和审计。先读results/cloud_20261003_swad/RUN_REPORT.md、SUMMARY.json、AUDIT.json、ANALYSIS.json及DRIVE_BACKUP_RECEIPT.json，勿按下方历史运行状态重启。

源用户3–8四训练/一验证/一诊断：ERM91.9028%、固定第13–24轮平均92.7222%、SWAD92.5846%。SWAD相对ERM+0.6818pp/4胜，相对固定平均−0.1376pp/2胜，完整冻结门槛失败；停止此SWAD配置的八折/扩种子/窗口网格。固定平均+0.8194pp的描述性区间也跨零，不把对照包装成可靠新机制。不与不同协议的旧八折直接比。近邻SWA作者README/train/utils已读，详见research/SWA_CONTROL_REVIEW.md；本次AdamW余弦+密集平均不是原SWA训练复现，论文全文尚未读完。

18模型独立函数式logits及6组作者LossValley重放逐位通过，174产物SHA/ID/选轮次/指标通过。144更新epochs计时3998.85秒；无需重复训练/审计。原目录/workspace/uwb-swad-source-runs-20261003保留optimizer/RNG/队列；/workspace/uwb-swad-archive-20261003有6个held包及metadata.tar.gz，共162,568,884字节，含全部144区间模型和最终18权重、预测/ID/配置/审计/源码，无原始数组与optimizer progress。远端持久ID和认证重新下载SHA以DRIVE_BACKUP_RECEIPT为准；ARCHIVE_RECEIPT uploaded=false是打包前快照。

下一步具体范围：核查距离轴压缩/时距分离卷积在雷达手势中的近邻论文及原789文件快照是否已做；若已有相同机制，作为既有基线处理。先建立源用户内的信号保真/类别信息/计算成本可否决假设，再冻结匹配参数和固定平均对照，不能仅因118倍参数减少就宣称新算法。不要改失败SWAD的平均起点或窗口再读取诊断分数；本轮没有实现/启动新候选。用户目标仍未完成，自动接续继续，只有实质变化通知。

以下是历史记录，旧进程/等待状态已过期。

# 当前接续：SWAD既有方法源用户先导（2026-10-03）

先读results/cloud_20261003_swad/RUN_REPORT.md与LAUNCH_VERIFIED.json，再查/workspace/uwb-swad-source-runs-20261003/queue_status.json和实际进程。新代码/协议已在训练前提交58b6e1e2e13fcf1abf3d091347705521b50cc583。本轮不重跑时序矩或圆周失败配置。

源先导仅用户3–8：循环一人诊断、一人验证、四人训练，共6条24轮轨迹。各轨迹同源梯度同时保留ERM验证最优、固定第13–24轮密集平均、作者LossValley SWAD。官方默认窗口3/6、比例0.3，epoch作为检查区间；1线程、seed2026。原3线程最终探针3.054h未过3h预算，改为1线程1.558h过门槛，保留全部成本证据，不改质量门槛。四项检查（36组官方数值/队列恢复、Dropout优化器中断恢复、诊断标签隔离、真实CNN合成全流程审计）通过，实际源梯度探针与平均模型独立前向通过；都不是识别有效性结论。

训练目录/workspace/uwb-swad-source-runs-20261003；日志/workspace/uwb-swad-source-train-20261003.log。进程自行依序训练6折后审计18个检查点：验证/诊断独立函数式前向逐位、官方LossValley从保存epoch区间模型重放、SHA/ID/选轮次/指标，再写AUDIT.json与SUMMARY.json。没有另启收尾进程。保存全部区间平均与progress中的optimizer/RNG/平均队列，切勿按中途分数改方案或重复启动。

下次先核验实时进程与进度，非僵尸才算活动；complete时才读完整SUMMARY/AUDIT，不要另跑重复审计。failed时读日志并保存现场，必要修复另立身份/目录。若确认原进程已退出但仅运行中状态，可在代码/协议/源SHA一致后同命令恢复（有文件锁），不得与已有进程并发：
```bash
python /workspace/uwb-cloud/uwb_direction1/tools/swad_source_pilot.py train --data /workspace/uwb-restored-20260930 --prepared /workspace/uwb-circular-prepared-20260930 --out /workspace/uwb-swad-source-runs-20261003 --cost /workspace/uwb-swad-cost-threads1-20261003.json
```

冻结继续门槛：SWAD平均比ERM+0.5pp、比固定平均+0.25pp，分别至少4/6改善，最差用户不比ERM低超过1pp。失败停止该配置八折/种子/网格；通过才另冻结原协议八折与强对照。源六用户内成绩不能和旧八折直接相减。SWAD是已有算法，该实验不是原创贡献；论文全文仍未读完。用户目标尚未完成，自动接续继续，只在有实质变化通知。完成后备份最终模型/预测/ID与必要区间模型到私有Drive，认证重下载核SHA，不把大数组/权重进Git。

以下为历史记录，旧“无活动训练”不覆盖本节和实时状态。

# 当前接续：时序矩CNN八折完成、门槛失败、私有备份已核验（2026-10-03）

**当前无活动训练。** PID14004/14201为结束后的僵尸记录，不可称后台训练；16run及自动收尾完整结束。先读results/cloud_20261002_moment_guided/RUN_REPORT.md、SUMMARY.json、FINAL_AUDIT.json、DRIVE_BACKUP_RECEIPT.json。下方历史运行/待审计/待上传状态已过期。

最终八用户宏平均：legacy94.1843%、static94.3155%、ordered93.9231%。ordered比static−0.3924pp、比legacy−0.2612pp，两者均仅3/8改善、描述性用户bootstrap区间跨零。冻结门槛失败，停止此配置的扩种子/相对矩/网格，不改门槛。源先导通过仅说明矩描述含动作信息，不证明其与CNN互补；static小增益也未确立有效新机制。单训练种子、八用户多轮研发，无独立确认数据；用户最终研究目标尚未达成。

16模型独立函数式前向验证/测试logits逐位一致；64产物SHA、源标准化、样本ID/分割、逐轮顺序、选最早最小验证CE、指标通过。298更新epochs计时6403.16秒，不含测试/写检查点/审计。无需重复训练或重跑审计。

模型包/workspace/uwb-moment-guided-final-models-20261002.tar.gz，18,568,289字节，SHA256=e6e05f1c00051533cfc87d52d721adfb9d96201affdc9b5a4de32556c083814d；已上传私有Drive文件101EUg3JZQ6dfT_zh5lVOK6H_MUha9xlb，父目录1Gtwre9JCxyaeRIjdmLaFY6VwS5F1j-ZA，shared=false仅owner。重新认证下载核SHA通过。67训练产物及审计/必要源码已归档，无原始数组/优化器progress。progress仍在/workspace/uwb-moment-guided-runs-20261002；旧ARCHIVE_RECEIPT uploaded=false仅上传前快照，以DRIVE_BACKUP_RECEIPT为准。

下一步：按research/SWAD_REVIEW.md核查官方默认窗口参数，实现CPU可恢复的密集区间平均与LossValley，并与官方实现数值对照；以legacy训练轨迹保留ERM与固定区间平均控制，先冻结源先导/预算再运行。当前只完成作者README及相关代码核查，没有全文核查、没有实现SWAD、没有启动新训练。SWAD是已有算法，效果若改善仅建立更强基线，不能自动宣称原创贡献。自动接续保持开启，仅实质变化通知，不再重复迁移或本轮审计。

以下为历史记录。

# 当前接续：距离矩先导通过，匹配CNN八折队列（2026-10-02）

## 20261002_2311 例行接续核查

训练PID14004与收尾PID14201均非僵尸存活，观测fold7/ordered第14轮，13/16run完成、52个完成产物SHA通过；训练/协议/帮助文件SHA未变。仍等待整项完成，勿重启或并发审计，无完整新分数。记录results/cloud_20261002_moment_guided/MONITOR_20261002_2311.json。新增research/SPATIAL_EXPECTATION_PRIOR_ART.md实际核查作者账户的空间期望坐标草稿相关章节，明确稿件标题不一致、非DSAE正式论文、非雷达证据；不据此声称全文核查或创新成立。

先读results/cloud_20261002_moments/RUN_REPORT.md和results/cloud_20261002_moment_guided/RUN_REPORT.md。源用户3–8的7100条内部留一先导已完成：静态40.4798%、绝对时序89.2336%、相对时序90.8990%、排序边缘58.4975%。按先于拟合的b70a946dad77699f1cee574955263fd82310f338协议通过门槛；24模型/72产物SHA、预测重放、960帧独立矩公式及源统计量审计通过。**这不是原八折成绩，也不是创新成立。** 源先导二进制在/workspace/uwb-range-moments-20261002，可约半分钟重建，尚未另做Drive备份。

后续CNN模型与协议在训练前提交213cd840871d32dbdd611f03937f16b9c60ed310。static与ordered各8折、同27.39万参数/初始化/优化器，另复用已审计legacy94.1843%作强基线（参数26.58万，不误称三者全同容量）。源六用户拟合矩分支标准化，原输入不变，训练所有CNN参数；验证/测试无拟合。完整新实验尚无结论。5项检查（含中断恢复逐位一致）和两臂独立前向检查通过，不能冒称完整16模型审计。

## 实际运行与防重复
- 队列/workspace/uwb-moment-guided-runs-20261002；日志/workspace/uwb-moment-guided-train-20261002.log。
- PID及观测见results/cloud_20261002_moment_guided/LAUNCH_VERIFIED.json。每次检查真实ps状态及queue_status.json，不能把僵尸或仅Popen回执当活动训练。
- 矩缓存/workspace/uwb-moment-guided-prepared-20261002；信号缓存仍/workspace/uwb-circular-prepared-20260930；已校验原数据与输入SHA，无需重下载。
- 成本初探8.236h超过6h门槛；只把成本探针大批矩提取改为32条分批后，相同5步loss逐值不变，外推4.644h过原门槛。两个成本文件都保留，不更改质量门槛，实际时长可能波动。
- 完整运行命令：
```bash
python /workspace/uwb-cloud/uwb_direction1/tools/moment_guided_signal.py train --data /workspace/uwb-restored-20260930 --prepared /workspace/uwb-circular-prepared-20260930 --moments /workspace/uwb-moment-guided-prepared-20261002 --out /workspace/uwb-moment-guided-runs-20261002 --baseline /workspace/uwb-circular-runs-20260930 --cost /workspace/uwb-moment-guided-cost-batched-20261002.json
```
只在确认原进程退出后才考虑同命令恢复，代码/协议身份必须一致，不覆盖冲突现场。

## 自动收尾已启动（2026-10-02 本轮接续）

原训练PID14004继续，观测fold1/ordered第18轮、1/16run完成；已完成run的4产物SHA及训练/协议/帮助代码SHA通过。未查看中途成绩决定方案，不重复启动。另启动不训练模型的收尾PID14201，状态/workspace/uwb-moment-guided-finalizer-state-20261002.json，日志/workspace/uwb-moment-guided-finalizer-20261002.log；实测waiting_for_existing_training且非僵尸。新增finalize_moment_guided.py自动等待原队列16run完成，再完整独立审计、汇总描述性用户bootstrap和归档模型/预测/ID。它不恢复或重启训练，原进程异常退出会记录failed。实测未完成队列拒绝汇总且不生成输出。

下次先查训练与收尾两个进程/状态，避免重复审计。收尾complete时读/workspace/uwb-moment-guided-final-audit-20261002.json、/workspace/uwb-moment-guided-summary-20261002/SUMMARY.json和ARCHIVE_RECEIPT.json；归档预定/workspace/uwb-moment-guided-final-models-20261002.tar.gz。当前这些完整产物尚未生成，不能称已打包/备份；完成后仍需通过私有Drive认证上传、重新下载验SHA并保存持久回执，不改变共享权限。

## 完成后的必须步骤（仅在收尾未在运行且需人工接续时）
queue_status为complete_pending_independent_audit后，运行一次：
```bash
python /workspace/uwb-cloud/uwb_direction1/tools/audit_moment_guided.py --data /workspace/uwb-restored-20260930 --prepared /workspace/uwb-circular-prepared-20260930 --moments /workspace/uwb-moment-guided-prepared-20261002 --runs /workspace/uwb-moment-guided-runs-20261002 --out /workspace/uwb-moment-guided-final-audit-20261002.json
```
审计拒绝少于16run，输出绑定identity.json及RESULTS.json的SHA。完整后保存真实RESULTS.json、逐用户/类统计和描述性不确定性、权重/预测/ID/配置，私有Drive备份并重新认证下载核SHA。优先让上面的既有收尾进程完成，不并发启动第二份审计。

冻结继续门槛：ordered相对legacy>=1pp、相对static>=0.5pp、6/8胜static、最差用户不低于legacy。失败就停止本配置网格；通过才独立训练seed、相对矩/纯矩消融及最近邻全文审查。现阶段只读OpenCV矩教程/代码和DSAE第三方实现，不能说完成原论文复现或全文核查。用户目标仍未完成；已知模块拼接本身不是新算法。自动接续继续，仅有实质变化时通知。

下方是历史记录，旧“无活动训练”等状态由本节及实时进程覆盖。

# 当前接续：24模型完成、私有模型备份验证；圆周机制不扩展

2026-09-30 09:54UTC核查。先读results/cloud_20260930_circular/RUN_REPORT.md、SUMMARY.json、FINAL_AUDIT.json。训练和收尾已完成，PID7419/7926仅是已退出的defunct记录，**不是正在训练**。24模型独立函数式前向、源输入、分割/ID、最早最小验证CE、预测指标及96产物SHA全部通过；无需重复训练或审计。

## 已冻结的真实结果与停止决定

| 方法 | 八用户宏平均 |
|---|---:|
| 历史LTR | 93.4417% |
| legacy轻量CNN | 94.1843% |
| angle_control | 94.1581% |
| circular | 94.4492% |

圆周版比旧LTR+1.0076pp，但相对legacy仅+0.2650pp、angle_control仅+0.2911pp且5/8改善；未过原门槛。**停止圆周机制的第二seed/池化消融/参数网格，不更改门槛。** 所有bootstrap区间跨零，一训练seed，八用户已有多轮研发历史；不能称可靠新机制或独立泛化证明。保留26.6万参数legacy作为可复现工程基线，它不是已确立论文创新。

整个24run含438更新epochs，墙钟9956.38秒。新CPU测量见CPU_INFERENCE.json：legacy前向1.8060ms，研究版完整预处理+前向11.4366ms；原架构随机权重分别26.8137/27.8788ms。仅本CPU、batch1、3线程；不能声称旧训练权重/GPU时延重现，也不能把参数缩小118.54倍当成全过程加速倍数。

## 代码、权重和恢复位置

- 原数据仍/workspace/uwb-restored-20260930；训练输入/workspace/uwb-circular-prepared-20260930。
- 完整训练目录/workspace/uwb-circular-runs-20260930，保留所有progress.pt/最优权重/预测/样本ID/历史。
- 汇总/workspace/uwb-circular-summary-20260930；原始训练日志/workspace/uwb-circular-train-20260930.log。
- 最终包/workspace/uwb-circular-final-models-20260930.tar.gz，27,013,156字节，SHA256=f2ea2bf0aad1a56487d440c08446a2f28122d5935b84196f29944d27ed5837d1。
- 已上传并认证重新下载核SHA：私有Drive文件ID1aWIw5HiXcrqOyNBqzd79J5jQNhWPBRjC，父目录1cOt20Xa69X8bk2M-uQkHJ65zO_rzjAwA。元数据shared=false，只有owner权限。稳定URL见results/cloud_20260930_circular/DRIVE_BACKUP_RECEIPT.json，不输出短期认证URL。
- 包内含24个best.pt、逐样本预测、split IDs、result历史、协议、必要源码和完整审计；不含原RF/prepared大数组。优化器progress.pt仍在原云端，不在此最终模型包。
- 若环境更换，可用Drive metadata→fetch(download_raw_file=true,include_base64=false)→download_file恢复27MB包并核SHA；不要要求用户手动上传。解压到新空目录，验证ARCHIVE_CONTENTS.json。原始数据恢复沿用既有163块索引。

旧ARCHIVE_RECEIPT.json uploaded=false是上传前快照；以DRIVE_BACKUP_RECEIPT为远端事实。旧SUMMARY.json inference_latency_measured=false生成于补做CPU推理前，以CPU_INFERENCE.json为新增时延事实；没有改写已备份归档。

## 已完成的下一候选诊断与下次具体工作

tools/diagnose_range_origin.py已对fold1六名源用户7100条执行距离中心统计，仅源信号和源标签，结果results/cloud_20260930_range_diagnostic/DIAGNOSTIC.json。类别解释中心位置约69.6–75.1%方差；控制类别后用户额外解释仅1.2–4.0%。不支持把全片段中心直接当作纯用户风格删除，**不盲目启动完整逐片段强制居中训练**。该诊断不否决所有轻微空间增强，更不是新算法效果验证。原特征和ID保存在/workspace/uwb-range-origin-diagnostic-20260930/source_centroids.npz（SHA在诊断JSON）。

已读PyTorch官方STN教程和定位/仿射采样实现（来源快照及blob SHA见该诊断报告），它是既有方法，不能宣称“加入STN”创新。qassemoquab/stnbhwd查询404未作为已读来源。最近邻论文全文与出版信息独立核验尚未完成。

下一步应以已保留的强基线为对照，在源用户内部检验**保留动作信息**的跨用户变化假设，再决定是否值得新端到端训练；不要从弱读出/圆周失败中堆叠模块，勿按测试用户定制中心或超参数。必要时先把候选转化为源用户内可否决的诊断，并检查是否已在原源码做过。当前没有新训练队列，勿声称后台继续训练已启动；自动接续保持开启，用户最终目标尚未达成。

以下是历史过程记录，旧进程/等待状态不代表当前状态。

# 最新接续：圆周表示端到端队列已实际运行（2026-09-30）

训练实现和三臂八折协议先于完整训练提交GitHub：d9f80720f46f83e61c3b2da68dedbdd3d422f6f7。2026-09-30 06:24 UTC核验进程PID7419存活，fold1/legacy已到第7轮检查点，最后一轮18.81秒；24个完整run当时完成0个。详见results/cloud_20260930_circular/RUN_REPORT.md和LAUNCH_VERIFIED.json。不要把准备探针、验证CE或部分进度称为测试成功。

## 08:57 UTC接续：22模型独立重放通过，收尾进程已就绪

训练PID7419继续，fold8/angle_control第14轮，22/24完成；独立审计22个完成模型的所有验证/测试logits逐位一致，源输入、ID、选轮次、采样顺序、指标和文件SHA均通过。尚无完整比较结论，不据中途分数改变训练。

另已启动**不训练模型**的收尾进程PID7926，其状态/workspace/uwb-circular-finalizer-state-20260930.json，日志/workspace/uwb-circular-finalizer-20260930.log。它等待现有队列complete后自动执行24模型完整审计，再汇总固定比较和描述性用户bootstrap，打包最终权重/预测/样本ID/配置/必要源码。不会重启训练或调参；训练进程提前消失会明确报failed。未完成队列的汇总入口已实际检查为拒绝，未创建输出。

**下次先查两个进程和收尾状态，避免重复审计或重复启动。** complete时读：
- /workspace/uwb-circular-final-audit-20260930.json
- /workspace/uwb-circular-summary-20260930/SUMMARY.json
- /workspace/uwb-circular-summary-20260930/ARCHIVE_RECEIPT.json
- /workspace/uwb-circular-final-models-20260930.tar.gz

归档只含最终best.pt、预测、split、result、协议/代码，不含原始RF或prepared大数组；优化器进度仍保留原云端目录。收尾脚本尚未实际完成上述整项输出，不要提前声称归档已生成/已备份远端。待收尾完成后检查包SHA和私有Drive目标元数据，再用Google Drive upload_file尝试上传归档并认证下载重验；当前查到的工具接受绝对本地路径string，但实际上传能力尚未验证。不改变分享权限。

完成后把真实完整结果、失败/通过门槛、逐用户比较、局限及归档回执写入最新远端父提交。若门槛失败，停止圆周表示网格，不包装为创新；用户最终研究目标仍未达到。新脚本tools/finalize_circular_signal.py及tools/summarize_circular_signal.py仅负责收尾，无模型选择改动。

## 07:58 UTC完整性巡检

PID7419持续运行，fold5/angle_control第11轮，13/24run完成。13个完整run的52个检查点/预测/划分产物SHA通过；训练源码、帮助函数和协议SHA未变。此项只核查完整性，独立前向仍以PARTIAL_AUDIT_0652的3run为已完成证据；最终24模型独立前向待整队列结束。无执行异常，保留原队列，不重启、不调参、无新的整项测试结论。巡检记录为results/cloud_20260930_circular/MONITOR_0800.json（内部观测时间07:58UTC）。

## 06:53 UTC例行接续核验

PID7419仍运行，fold2/angle_control第4轮；完整run已完成4/24，已独立审计先完成的fold1三臂：验证与测试logits逐位一致，输入、ID、选轮次、采样顺序、指标、产物SHA通过。只保存局部完整性证据，未据中途成绩改变方案，也没有发布整项测试结论。无需重启，保持原协议。新增近邻代码/文档核查见research/RADAR_PHASE_NEIGHBORS.md：FMCW复值手势网络和相同UWB数据的CNN均已有先例，不能把保留相位或轻量CNN本身当创新；全文和出版元数据仍未独立核验。

## 当前云端的具体接续

- 已恢复原数据：/workspace/uwb-restored-20260930。输入缓存：/workspace/uwb-circular-prepared-20260930，signals.npy SHA256=1f689bba7f10bb37faf07a557c52653843e81b5907f04d17fa92dbf4dbae7c5f。
- 队列目录：/workspace/uwb-circular-runs-20260930；日志：/workspace/uwb-circular-train-20260930.log；启动记录：/workspace/uwb-circular-launch-20260930.json。
- **先查ps与queue_status.json，仍运行就不要重复启动、不要改源码/协议。** 若环境确实停止训练，检查错误和progress.pt，再按下面原命令恢复。身份哈希不匹配应保留现场排查，不覆盖。
- 命令：`python /workspace/uwb-cloud/uwb_direction1/tools/circular_signal.py train --data /workspace/uwb-restored-20260930 --prepared /workspace/uwb-circular-prepared-20260930 --out /workspace/uwb-circular-runs-20260930`。
- PyTorch2.14.0+cpu已安装于/home/agent/.local/lib/python3.12/site-packages/torch。3线程CPU，无GPU。26.6万参数，预计整队列数小时，成本探针上限外推6.32小时仅估计。
- legacy、angle_control、circular各8折，32轮上限、最少16、patience8；最早最小验证CE。测试h、验证h%8+1、其余6训练。按原协议完成，不根据中途测试挑方法或网格。
- 完成后运行独立审计：`python /workspace/uwb-cloud/uwb_direction1/tools/audit_circular_signal.py --data /workspace/uwb-restored-20260930 --prepared /workspace/uwb-circular-prepared-20260930 --runs /workspace/uwb-circular-runs-20260930 --benchmark /workspace/uwb-circular-cost-20260930.json --out /workspace/uwb-circular-final-audit-20260930.json`。输出需新文件名。当前START_INPUT_AUDIT仅核查源输入64条，审计完整run数为0，不冒称已核查24模型。
- 全部结果冻结后，候选必须同时胜legacy1pp、angle_control0.5pp、6/8用户优于angle_control、总体至少历史LTR93.4416667%，才扩第二训练seed及无范围池化消融。失败即记录并停这条网格。
- 已知圆周坐标/复数网络不是新算法；本轮只假设相位角分支跳变在范围压缩中有代价。已读作者资料和实现，不冒称读完论文全文或最近邻综述。独立确认用户仍缺失。

29项原集成检查通过；新增独立前向后6项信号检查通过，包括中断恢复逐位一致。64条真实源输入由SciPy独立实现逐位一致。权重与输入缓存仍只保存在当前云端，未上传Git/Drive；若换环境要核实检查点可用性，不仅检查源码。

仅在完整新结果、失败、完成、重大阻塞时通知。无可行动变化保持安静；不要重报迁移、不再做旧冻结读出网格。用户目标尚未达到，保持云端自动接续。

# 最新接续覆盖（2026-09-30，融合头四臂八折与CPU原网络探针后）

先读 `results/cloud_20260930_heads/RUN_REPORT.md` 和 STATUS.json。四臂32次训练、96个二进制产物逐SHA核验、最优权重验证/测试logits逐位重放完成，当前无后台训练。subset_ce=92.0367%，full_ce=92.5178%，full_supcon=93.1030%，full_crossuser=93.3625%；旧LTR=93.4417%。跨用户版相对完整CE+0.8447pp、仅4/8改善，相对旧LTR−0.0792pp，未达冻结门槛，不补第二种子、不扩该网格。7/32选择epoch0，勿把这些当作新训练模型改善。

本会话已安装CPU PyTorch2.14.0于 `/home/agent/.local/lib/python3.12/site-packages/torch`。原三雷达31,511,500参数网络在真实源用户原始输入上完成CPU前向/反向/Adam探针，8条批中位1.141秒，3线程，单源训练遍历粗估16.89分钟，不含验证/I/O。完整8折×40轮仅训练外推约90小时，不能宣称已执行。下一步先设计/测量CPU可承担的信号级端到端训练，不默认启动这套昂贵全队列。

注意环境路径不同：本会话源码仅28份核心文件位于 `/workspace/uwb-source-c735e94/project`；核方法会话另有完整789文件checkout。两边数据都恢复到同名 `/workspace/uwb-restored-20260930`，但不是共享文件系统。保留下面核筛查结果与全部失败记录，不重做共享/专属、等权DTW阶段监督、温和增强、低秩校准、收益选路或已否决的读出网格。用户目标尚未完成，不把当前运行通过当创新成立。

以下为先前接续记录：

# 最新接续覆盖（2026-09-30，核交互筛查后）

优先读取 `results/cloud_20260930_kernel/RUN_REPORT.md` 和 STATUS.json。新容器已经重新恢复163块/86文件并全SHA通过；数据仍在 `/workspace/uwb-restored-20260930`。checkout完整789文件已逐项SHA通过，路径 `/workspace/git-local-demo/uwb_direction1/original/project`。当前无后台训练。

八折编码器记录证明同折六源用户、验证/测试隔离和验证CE选轮次，缺旧.pt所以不是模型前向再验证。八用户历经多轮开发，不能声称是未见最终测试。

已完成新的加法核/联合核同容量验证门控、两地标种子及完整测试。联合核测试93.8080%/93.7712%，对原LTR93.4417%仅+0.3663/+0.3295pp、各4/8改善、用户区间跨零，最差用户略降；不认定可靠改善，不扩大本设置网格。只对弱加法核有一致增益。不是多编码器训练种子；权重与分数在本地`/workspace/uwb-kernel-*`，Git保存协议/代码/结果/哈希，可约50秒CPU重建两轮。

用户最新指示优先，不重做共享/专属。已发现旧时间诊断与温和增强完整报告，见最新报告；不要重复等权DTW阶段监督或原温和增强设置。下一步须以信号和旧失败证据约束候选，并先适配CPU训练/测量成本，无GPU/torch/旧权重。研究目标未达成；不要把已运行核读出称端到端训练。

以下为历史接续，若矛盾以上述更新为准：

# 云端接续任务

用户目标：继续方向一研究，数据用其UWB-Gestures，单雷达或三雷达均可，最终寻求至少一种有效、可复现且有文献支撑贡献的方法。用户已经允许私有GitHub代码/记忆更新和私有Drive下载，电脑会休眠，不再把手动本地操作作为默认前置步骤。

先读同目录PROJECT_MEMORY.md、上级STATUS.json、results/cloud_20260930/RUN_REPORT.md、research/NEXT_EXPERIMENTS.md。仓库 `Lixuu2001/git-local-demo`，研究分支 `codex/uwb-direction1-cloud-20260930`。远端分支含789文件原始快照，不能用当前不完整本地Git历史覆盖它。通过GitHub connector获取最新分支提交/树，以其为父提交，只更新本次文件，force=false；先检查其他任务是否有新提交。

私有Drive文件夹：`12AGMwT99yXCTYBQocQaX901Br-IelmOf`。完整持久下载索引已保存于仓库 `uwb_direction1/migration/20260930/CLOUD_DOWNLOAD_INDEX.json`，原索引文件ID `1HNQfj72lCbLtKW9lO7-vuAmi52HFZrG3`。这不授权改变分享权限。云端数据已恢复且校验通过，目录 `/workspace/uwb-restored-20260930`；先检查已有文件，只有环境更换/文件确实缺失时才重取。重取时读取Drive元数据后fetch(download_raw_file=true,include_base64=false)，download_file立即物化，分块并发最多3，不输出认证URL/凭据。

原SHA和恢复凭据见报告。旧LTR八折缓存重算93.4417%，新冻结特征类内用户惩罚92.7231%，已否决，不再重做。现有八用户属于探索研发数据；所有训练统计只源用户，超参数只验证用户；不得改变数据划分或引用其他协议高数字充当收益。

下一项有意义的独立工作：补齐近邻雷达/共同特有表征文献的可访问正文核查，结合真实信号检查候选是否重复；设计并实现一个可训练的共同/视角特有动作表征试验，保留同容量CE、普通SupCon和跨用户正样本等必要对照。现有源码足以重建原预处理，但prepared off/only大数组和旧模型权重没有迁移，不要声称它们已存在。CPU预算优先：先验证数据通路和小批训练耗时，再限定完整八折配置并保存。若候选未过验证，不大规模加网格、不用测试用户调参。

每次接续先检查是否已有任务/进程正在训练；有则读取状态并推进可独立工作，不能重启重复训练。需要较长计算时写进度JSON、日志和可恢复检查点，明确PID/任务ID、数据和代码哈希。没有可执行工具/GPU时，准确记录缺失能力，继续能做的文献/实现/审计，不冒称后台计算已启动。只在有实质实验结论、重大阻塞、完成或需用户选择时通知。

到达研究目标后，保存方法、训练权重、代码、全部八折/种子对照、消融和限制，再暂停自动接续任务。用户暂停/取消时立即遵从。当前没有足够证据保证创新或一定成功。


## 2026-10-03 GN无标签适配先导完成（2026-10-03T13:50:35.897881+00:00）

48条流（6源用户×2固定顺序×4臂）完成且独立审计。存储/随机：冻结92.7222/92.7222%，普通熵92.7222/92.7222%，筛选熵92.7222/92.7222%，SAR式92.7222/92.7222%。SAR主门槛False，停止当前学习率/阈值/顺序网格与八折扩展。不能据单一设置否定全部TTA，也不能切换次要臂包装成功。

预注册b45d0a064cbb1d995784c8f1cb90bd5dae4a253e；评分前勘误81081c62a165cdabaccdd6dd0433114536e7cf9b：0.0005源于作者ViT缩放，原文字误称ResNet，真实GN默认batch32=0.00025。所有数值冻结未改。先预测再更新、仅当前x、GN352参数、Dropout关闭、无目标标签拟合/选参。SAR频繁恢复只作观察，不是因果结论。主文仍未读；官方实现非创新。旧原始uwb_osr/adaptation.py已经有开放世界T3A式DualMemory，设计已列Tent/CoTTA，不得再次宣称首次引入适配。

私有备份与逐包/成员认证重下载校验见results/cloud_20261003_adaptation/DRIVE_BACKUP_RECEIPT.json。每批GN增量和优化器保存，完整初始tail重用SWAD备份。审计66文件SHA/144批独立前向/48真实恢复。所有原始数据/旧快照不变。无当前后台进程；目标未达，自动接续不暂停。下一步先整理闭集跨用户证据与未重复的新假设，不再沿失败的适配参数网格试错。


## 最新：源组风险诊断完成（2026-10-03T15:36:36.906301+00:00）

预注册027c72652bb2e2c7a67a973d670cff4e45a9284c后，18既有模型（6轨迹×1/12/24轮）、仅4训练+1验证角色共106500重复角色行完成。第12/24轮全部训练用户错误率0，验证宏错误率均值7.7222%/7.4343%；第1轮训练最差用户均值7.6048%、验证13.2917%。持续训练欠拟合门槛0/6，停止以该机制直接启动无额外正则组DRO。不能由此否定正则化Group DRO。CE与错误率趋势不同，不能当因果或具体虚假相关证明。

运行203.63秒、无新增训练、无各轨迹held信号评分。28产物/18模型SHA、1065用户类别单元独立统计、72首末批独立函数前向、角色/门槛重算通过；7100源样本反复出现在轨迹，不是106500独立样本。7私有包4,950,459字节41成员全部认证回下载SHA验证，见results/cloud_20261003_group_gap/DRIVE_BACKUP_RECEIPT.json。

Group DRO作者README（含摘要和转载附录段落）与loss.py已读，主文未读。作者警示过参数模型可使所有训练组风险近零，正则化对最差组泛化关键；这是文献线索，既有算法不是创新。当前基线AdamW weight_decay=0.001。下一项仅准备设计正则化/组风险对照：同正则同采样同预算同验证选模的普通训练是必需控制，旧强tail作为参照；处理用户组/缺类/缺组/稳定q/恢复后再冻结并跑真实先导。未实现、未预注册、未启动下一训练，不伪称后台运行。目标未达、自动接续保留，不重复迁移或已失败TTA网格。


## 最新：匹配正则化/组DRO训练真实启动（2026-10-03T16:28:01.492985+00:00）

预注册9011739140cb0686fd1f259f7e2a8dcbfcfdb78f后启动云端PID19340，18新轨迹=源3–8×uniform_base/uniform_reg/dro_reg，每条24轮。观测uniform_base/held3第14轮1050步，0/18完成。目录/workspace/uwb-regularized-group-runs-20261003，日志/workspace/uwb-regularized-group-train-20261003.log。先查进程/queue_status，绝不重复启动或据中途分数改参数。

三臂同源组平衡采样（四用户各16/批、75批/轮）、同初始化265836参数、同验证CE选端点；decay分别0.001/0.01/0.01，DRO logq步长0.01仅用源训练组均值CE。固定tail次要，旧强tail附加参照。主dro_reg必须同时胜两个匹配普通臂和旧tail均值0.5pp、各4/6获益、单用户不降超1pp才扩后续。既有算法不是创新；AdamW衰减非官方L2等价；这次测试泛化正则假设，不是重新声称源组欠拟合。

数值/采样/缺组/恢复/诊断标签隔离检查通过；CPU实际源批成本含75%余量5.082小时，已开始运行，非只有计划。第4轮300步启动快照包含1–4全部轮次产物及完整optimizer/RNG/q/best/tail，9,004,704字节29成员2私有Drive块，认证重下载重组及SHA全部通过。启动打包打印变量遮蔽报错已修复并独立重验，不影响训练；启动快照不是最终18轨迹备份。

见results/cloud_20261003_regularized_groups/START_REPORT.md、RESTORE.md、START_DRIVE_RECEIPT.json。队列完成会自动全量功能式前向、源ID/q/选择/产物SHA审计；之后仍需完整私有备份最终权重/优化器/日志，再提交真实汇总。当前没有最终识别增益/创新结论，目标未达，保留自动接续。原数据及789源码快照不变。
