# 2026-11-02 — t-LeNet完整默认成本21.40h资源关闭；DLinear完整分类CLI来源/合成验证及成本预登记

在e7b45a25df99a4f7fab6eafdc6d0df15db55292b全部42份文件远端逐字节回读/force=false核对head之后，t-LeNet完整1000/B256/seed20261029/全10窗口只计时一次。子进程72773正常exit0，无timeout，wall7.33519s/CPU7.31107s；8train max.1877435s，4eval max.0966768s，RSS727852KiB。固定保守估算77046.4395548s=21.4017888h，超过43200s=12h，内存通过但资源门槛失败。完整确切资源配置关闭，不缩轮/批/窗口或重计时，不读真实UWB值/标签/拟合；没有准确率负结果。来源、exact lazy streaming及合成验证保留供可用资源下复现。

最新四记录确认完整默认MiniRocket9996已于10-29执行，source6均值78.5336%，本轮只核查源码/论文不等价重跑。ResNet1500此前保守估算103363.8s资源关闭，不重复profile。最新四记录及本地原789/当前研究有界搜索未见DLinear执行。因此下一项固定为Time-Series-Library DLinear完整分类迁移，commit4e938a1767106324dd753b2a44832bf870a0252e，model blob3f4d666a9ffe7fb6f58627ba43f1a9d3d9804d78；原模型/分解/optimizer/EarlyStopping AST不改。论文arXiv2205.13504选定架构与实验背景段已读，原论文研究预测任务；此为库内分类迁移，非原论文分类复现、非原创贡献。

完整CLI默认max10epochs/B32/patience3/lr.0001/moving_avg25，90x567->12任务适配，628752参数/6张量，1CPU/seed20261029；无缩减。当前分类driver实际使用RAdam，CE/clip4，无LRschedule。源早停监视validation micro accuracy，tie保存最新/清零counter。源driver把TEST复用两次；将两次均换为source2验证，不在训练或选模期间读diagnostic1/7–8标签。所有角色/输入来自固定元数据，native train shuffle/global RNG保留，10workers适配0，末批保留。合成独立NumPy全图、7次RAdam含rectification、完整freshrestore/精确nextupdate、原生权重、tie/patience3检查全部通过。

只预登记一次完整synthetic CPU成本，8train/4eval/3native/3fullstate，2*(1470*maxtrain+710*maxeval+11*maxnative+24*maxfullstate)+120，43200s/14680064KiB门槛，无缩减fallback。5项边界检查通过，52项来源/关键runtime/先前资源关闭证据冻结。全部远端逐字节回读并force=false核对head后才能运行；此提交时DLinear尚未实际cost/真实UWB值/标签/fit。成本通过仍需另登记完整真实源用户协议/runner/auditor后实际实验。

TCN完整审计/完整私有保存后一次固定gate失败92.0833%vs98.6667%、差−6.5833pp，确切20/B64/seed配置已关闭不救援。目标未达，继续准确率实验与原创证据分别评估；历史八用户开发均探索性，7–8标签不用新选择。既有接续6abc995e1620819195fcff77dbd81118保留，无Windows、重复下载/训练。原789/真实ID值/原数组/模型二进制/私有原作者源码不入Git。下一步实际CPU成本；通过则立即实现/登记/运行真实实验，失败保留确切资源结论再推进下一项有依据的方法。
