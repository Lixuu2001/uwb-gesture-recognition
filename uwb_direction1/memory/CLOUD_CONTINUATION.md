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
