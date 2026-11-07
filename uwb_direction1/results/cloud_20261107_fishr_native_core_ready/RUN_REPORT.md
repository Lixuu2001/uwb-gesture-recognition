# Published Fishr native classifier core: core_ready (not fitting registered)

## 2026-11-07：VREx完整六折固定配置已失败关闭；下一候选Fishr数学核心已实现并通过作者EMA/梯度数值核验，新训练未启动

最新实际结果提交cabd7e115924374e0cb349f95769708e239570fc必须保留：VREx完整六人LOSO92.8444% vsGAP93.6672%、-0.8227pp/3胜/最差配对-3.6364pp，预设全部门槛未过，确切10/500/nativeGAP64配置关闭，不救lambda/anneal/seed。76块550659116字节1980载荷私有raw/allmember和整档SHA已通过，完整64生产/各自fresh64/原生审计及唯一cohort评分和无标签推断均结束；SUMMARY/FINAL_RESULT_STATE/once锁和所有图表/CSV/成本见该提交results/cloud_20261106_vrex_loso6_all6result/report_artifacts/FINAL_REPORT.md。原P/STATE是评分前冻结快照，不据此重启训练/once；原64529已结束，无后台训练。

按用户“继续验证有文献依据的方法，不因暂不能称原创停止”推进不同的已发表候选Fishr（Ramé等，ICML2022/PMLR162，https://proceedings.mlr.press/v162/rame22a.html）。核对既有正式论文全文记录与实际固定DomainBed b93c22a1cfc3b2428398272c1a116c8de1f4139e Fishr类、hparams默认lambda1000/EMA.95/anneal1500及MovingAverage/l2_between_dicts；本轮misc.py通过GitHub取原始正文、独立Git blob SHA f9e6497a82fed9a4e1d70346ad708c2c9df2ad9f核验，保留MIT及作者源码，不声称整个作者图像benchmark复现。

历史SOURCE_FISHR_PROTOCOL是265836参数、24轮、seed2026、最后线性层CE统计、时间偏差修正EMA raw/(1-beta**t)、源梯度范数校准/剪裁lambda、无optimizerreset、固定13..24tail选模。新候选拟用完整native GAP617612/64轮/seed20261029/六人LOSO和原native全局BCE，计算全部4个MLP classifier张量18060个参数梯度，保留作者固定1/(1-beta)与detach历史、固定1000/1500，不做旧校准。协议/实现不同，不能把旧改写干预等同完整默认组件；旧失败不追改，没有使用旧用户7–8分数来选新参数。只比较协议与已读实现范围，不把文件名筛查冒称穷尽证明。

已在/workspace/uwb-fishr-native-loso6-20261107实际写出fishr_core.py/check_core.py：一次native两层128/ReLU/dropout.2/12分类器前向，解析全部每例mean12 BCE梯度并保留二阶图；source EMA人口方差/全参数拼接距离。实际合成检查seed20261029通过：每例梯度对逐例autograd最大绝对差4.163336e-17、penalty差0、对特征和全部分类器参数二阶梯度最大差1.654361e-24；直接执行固定作者AST函数，5次EMA输出逐位一致、当前统计梯度补偿/历史detach/序列化继续、缺组/单例/连通零及1500边界意图通过。作者源码一个无关regex SyntaxWarning原样保留，exit0。检查体wall.040593/CPU.039096秒，不是完整GAP训练成本或准确率证据；全部数据/标签由RNG生成，optimizer_steps=0，没读真实UWB样本或标签。

明确未完成项：完整617612骨干接入、真实train/validation loader/64轮worker、完整EMA/optimizer/RNG checkpoint、ownedPID资源监督、独立fresh64/native审计、完整native CPU成本门槛与拟合预登记/私有initial/exact release均尚未完成；native_worker_implemented=false、fitting_preregistered=false、new_background_training_started=false。当前只有候选数学核心，没有新的识别结果、训练进程或已通过完整native预算，不得误报后台训练。下一接续先读PREPARATION_STATE/CORE_CHECKS/CORE_HASHES和历史对比，复用已通过核心，再推进这些具体实现；完整科学配置/seed/全6协议/12h wallCPU门槛正式冻结后才能拟合，禁止旧held1筛选/测试救参或重复旧任务。

迁移差异已列入FITTING_PROTOCOL_DRAFT：原生BCE替代作者CE；完整native nonlinear classifier而非缩小末层；保留原B64混合source sampler，缺少当前训练组时该组EMA不更新且不计本批penalty；singleton人口方差0；未来full native AdamWLookahead整体reset1500并保留EMA。这些只是候选设计，native reset和全骨干尚未实测。未来若全6有效，还需同全局BCE/同reset1500/no-variance控制，不能用VREx等域均值/reset500代替；准确率提升证据与原创贡献证据分开，Fishr已发表算法迁移不自称创新。

保持同一已启用自动任务、频率及云环境，不新建自动任务或模型进程。原始数据/基线无需重新迁移、恢复或训练；用户7–8不用于新方法选择，历史全8人反复开发探索性、非独立新用户验证。原789源码、旧科学文件/失败、Drive私有与仓库可见性保持。研究目标未达；当前实质进展为完整VREx负结果及可运行/可核验的下一候选核心，不是抽象文献否定。

源码发布辅助程序最初仅允许.py/.js/.json/.md，拒绝无扩展名的作者MIT LICENSE；发生在任何远端写入前，已有9份本地拷贝逐字节一致。保留原guard，改用固定10个公开作者/核心文件的显式清单纳入原LICENSE，逐项核对已有字节并补发布manifest，未修改核心或重新运行模型/数值试验。CORE_PUBLICATION_OPERATOR_REPAIR.json保留错误与修复，不能把操作性发布错误当作Fishr科学失败。

