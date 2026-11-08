# Full native GroupDRO default component preparation, no UWB fit: core_preparation

## 2026-11-08：IRM完整六折真实负结果已关闭；下项完整native GroupDRO默认eta.01核心实现与生成审计/CPU预估通过，尚无新UWB训练

父提交5907ffeabc7d0fedaaa37b3db3c9a82ba27a2c47已精确18文件回读：固定IRM100/500完整617612六折89.4239898990% vs GAP93.6671717172%，-4.2431818182pp、0胜1平5负、最差paired-10.8333333333pp，私有75块/1995载荷及12组盲推断和唯一cohort评分均已完成。确切配置关闭，不修改原科学文件/旧失败结论/阈值，不运行正结果才规定的reset500归因对照，不再调用其once或旧MMD单折once；完整证据和成本在results/cloud_20261107_irm_loso6_all6_result。

按用户“完整默认规模不能等同旧缩小改写”核查历史：旧GroupDRO是265836参数、24轮、seed2026、源用户3–8、每组16循环采样75批、group-CE、AdamWdecay.01（有matched-reg对照），不是本次617612/all3 signedRMS90x567/native混合B64/64/seed20261029/nativeAdamW.001wd.0001+StableLookahead.5k6/clip2的默认组件迁移。旧失败保持，不借此重开旧eta/decay/seed网格，也不因为不能立即宣称创新就停止实现。新方法固定DomainBed GroupDRO默认eta.01：q源用户乘exp(.01*detach风险)、归一化、源均值BCE加权，全部4训练用户固定排序；原生混合批缺组仅按预先声明的present-q条件化规则，所有现存组时保持原dot(q,risks)。保留poolBN/dropout、native全局BCE另作日志与相同验证选模，非作者逐env forward/CE benchmark复现。eta不按IRM held结果选；主baseline不降低，正才做固定equal-source-mean/eta0六折对照归因。

文献Sagawa/Koh/Hashimoto/Liang ICLR2020/arXiv1911.08731，已核当前MIT DomainBed固定b93c22a1cfc3b2428398272c1a116c8de1f4139e的GroupDRO类和默认hparams，原作者README/full loss.py/Appendix C.1回顾保留；未读主文全文，不夸称系统综述/作者完整benchmark复现。已发表强基线迁移和原创贡献证据分开，不把组加权/已有模块组合称新算法。

新隔离prep/workspace/uwb-groupdro-native-prep-20261108已实现groupdro_core.py：detach q、完整native optimizer+q/计数状态、无IRM退火或reset。独立NumPy BCE/q/logit梯度、全617612参数manual-logit-gradient参考最大差1.4901161193847656e-08、float64目标差1.1102230246251565e-16，固定作者全4组方程、2/3/4组缺失、nativeLookahead slow6/disk7→exact8/model/AdamW/q/全部Python/NumPy/Torch/order RNG/训练模式恢复均通过。另一个实际open防火墙下的全B64/native617612生成更新和全部6个源组角色、held组拒绝且q不变、单一present组等价BCE及梯度、非有限风险拒绝且q不变通过。没有读取真实UWB数组/手势标签/样本ID，也没有训练真实模型；诊断标量conversion warning原样保留，不是梯度/训练失败，未为此重测预算。

在测量前PROTOCOL固定8steps/arm、前3warmup、native/GroupDRO交替顺序和maxwall/CPU比值；唯一生成profile ratio1.0635646915307622，沿既定同cloud预算公式1.25*31189.589142517*max(1,ratio)+600，投影完整6折生产+fresh重放reserve42065.182194165376秒（11.6848h）通过12h gate，12GiB上限不变。这是估计不是实际UWB耗时或准确率结果，不重测挑更好时间、不缩小结构/输入/B64/64/seed、不修改强baseline。生成core检查wall15.06026463501621/CPU15.05432966秒含profile与恢复/梯度检查；额外防火墙测试wall0.7783144349814393/CPU0.778111198秒，不相互重复计profile。所有旧实验8h级成本、网络/运维未知值保持已刊记录，不抹去。

尚未实现完整GroupDRO六折loader/worker/journal/资源监督/原生first独立审计/64+fresh64 lifecycle，也未创建actual角色、初始化私有保险或真实release；当前新训练0、后台训练false。NEXT_IMPLEMENTATION列明先补齐并生成审计/完整源码和六折协议精确预登记、私有all6未训练初态及真实roleIDs/inputSHA保险后，才允许唯一真实phase。当前prep源码冻结与生成结果不是启动授权，更不是训练完成。禁止原完成模型重训、单人测试门槛、测试调参；全6输出/模型/轨迹/审计/私有冻结后一次统一评价，仍以+.5pp/4胜/最差paired>=-2pp为固定花费规则并如实保留所有正负效应。users7–8不用方法选择、历史全部八人探索性非独立验证；原789/可见性/唯一自动任务启用频率/云环境不变，目标仍未达到但实验循环继续。

