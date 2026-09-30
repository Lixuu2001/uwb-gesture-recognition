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
