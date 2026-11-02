# 2026-11-02 — DLinear完整分类CLI实际成本通过，真实source单次协议/完整审计预登记

成本预登记05c74f2033519ea8e6e977b1d7cf8a3326647086（父e7b45a25df99a4f7fab6eafdc6d0df15db55292b）37文件逐字节回读/force=false核对head后，只profile一次：完整DLinear CLI10/B32/patience3/RAdam.0001/clip4/average25/seed20261029/1CPU，保守397.8051855391823s、RSS420784KiB，43200s/14GiB门槛通过；TERM正常exit0无timeout。此成本未读UWB值/标签/fit，不是准确率结果。

本提交在新方法实际UWB访问之前固定真实source3–6 train4700/2val1000/1diag1200，最多10epochs/1470更新/47000训练出现，B32末28；原生shuffleTrue/globalRNG、无drop_last、0workers适配；val规范顺序32批末8，两次val均只source2。源EarlyStopping按validation micro accuracy/patience3，tie保存最新并清零counter；达到早停或10轮即停。固定完整628752参数/6张量共享线性分解与分类head不改；RAdam.0001/CE/clip4不调度、不改seed/轮数/宽度救援。输入仍完整signed per-view RMS90x567；替换UEA数据loader，不声称原论文预测任务/UEA全处理链复现。库原模型、分解、RAdam选择和早停AST保留，paper arXiv2205.13504为架构依据，分类是明确迁移实验，不称创新。

真实协议SHA f8d0ac39bebf1639c45d91fc2ea85f41e7b6b5a31c8d682de825e0a0c5f99055，FROZEN SHA f584338aed54af294be4949a1d5aaf4635a7fc567c22e6d7587351bd829a705a，共107项依赖冻结。合成完整10轮/30步producer→fresh auditor逐次ID/loss/gradnorm/双val/early状态/五组完整状态/原生权重/两组logits逐位一致；fresh epoch checkpoint恢复下一轮及3种错误checkpoint拒绝通过；另以明确合成valobserver触发4轮/4步/selected1的真实producer与auditor早停终止路径，非实际验证分数。8项协议/角色/完整性/私有scope/恢复和损坏拒绝检查通过。所有测试在冻结前完成，不在真实运行后修改/重跑。

训练和完整审计分别600wall/590soft/600hardCPU，控制器只各Popen一次、无auto retry/resume/Drive/Git/评分。独立审计将重放实际全部更新/早停/所选epoch、初始首步best/last/final权重/RAdam/模式/globalRNG/early-state、原生3份权重、两组精确ID/logits及6900source rawRMS/cache/FP64，不读diag标签。完整模型集owner-only Drive raw分块/整包/全部成员回读后才可一次固定finalize。最终预测状态为best权重+last优化器，仅推理审计；best.pt另存对应自己的优化器，不称整任务自动恢复。

同ID TimeMIL98.6667%+.5pp/1e-12门槛不变；失败关闭确切10/B32/patience3/seed配置，不扫诊断或seed/轮数；通过须另登记source6确认及后续强基线/消融。用户7–8标签不用新选择；已有八用户历史暴露，结果仍探索性非独立/八折提升/原创贡献。此提交时真实训练尚未启动；全部本次文件远端逐字节回读、latest-parent force=false及head核对后才写marker并实际Popen一次。

t-LeNet完整1000/B256资源21.40h关闭，无真实准确率结论；完整MiniRocket及ResNet1500已有历史不重复。TCN完整20诊断92.0833%、差−6.5833pp已审计私有保存并关闭。目标未达，继续可运行方法真实验证，准确率与原创性分别累积证据。既有自动接续保持，不新建/暂停；Windows未参与，原789/真实ID值/原数组/模型/私有作者源码不入Git。下次先检查已有进程/检查点，正常运行仅本地journal，完成后推进完整私有保存与一次gate，禁止重复启动。
