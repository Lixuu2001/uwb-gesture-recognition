# 源端SAM：实现和真实成本检查通过，待预注册后启动

当前没有候选诊断效果结论；本报告为启动前记录，之后以LAUNCH/START_OBSERVATION和STATUS为准。已有SAM不是原创贡献。论文主文未读，作者代码/README阅读边界见research/SOURCE_SHARPNESS_REVIEW.md。

固定三路普通SignalNet265836参数；6源用户3–8四训/一验证/一诊断。sam24为24轮rho=.05的CE-only全参数SAM，两个前后向使用相同Dropout掩码，恢复原权重后普通AdamW更新。erm48为48轮普通CNN作为近似前后向次数对照，复用原24轮普通CNN作为强基线。lr=.001/decay=.001/cosine最小1e-5/seed2026/batch64/CPU1线程。主尾平均为13–24轮与25–48轮，次验证CE选模不可替换主结论。

主SAM须较原24轮tail高至少0.5pp、较48轮tail高至少0.25pp，对两者各至少4/6用户提高、每用户下降不超1pp；否则停止该rho/decay/seed/window网格和八折扩展。通过也仍需同协议八折、多种子和领域贡献证据。六人反复研究、单种子不是独立确认，不与旧93.4417%直接比较。

TESTS.json记录解析梯度、随机性、异常恢复、完整优化器恢复、独立前向与留出隔离通过。COST_INITIAL.json记录原5h估算门槛未过；COST.json复用相同真实源批时间，在预注册前只修改资源预算至10h，估计8.664782h含75%余量。没有重新解释失败的效果门槛，没有候选诊断调参。真实普通首epoch与已完成旧基线逐位吻合。

拟运行目录/workspace/uwb-sam-source-runs-20261004，日志/workspace/uwb-sam-source-train-20261004.log。12条新轨迹共432epoch，6旧基线复用；保存全部epoch段、顺序ID、完整model/AdamW/scheduler/RNG/tail/best。最终18条轨迹全部ERM/tail验证/诊断独立功能前向、角色和SHA审计，自动本地归档，再由后续接续私有Drive认证上传回下载校验；本地归档不等于远端备份。
