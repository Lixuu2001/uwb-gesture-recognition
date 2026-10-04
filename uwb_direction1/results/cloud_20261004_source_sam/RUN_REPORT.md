# 源端SAM：云端正式训练已启动（2026-10-04）

当前没有候选诊断效果结论；本报告为启动前记录，之后以LAUNCH/START_OBSERVATION和STATUS为准。已有SAM不是原创贡献。论文主文未读，作者代码/README阅读边界见research/SOURCE_SHARPNESS_REVIEW.md。

固定三路普通SignalNet265836参数；6源用户3–8四训/一验证/一诊断。sam24为24轮rho=.05的CE-only全参数SAM，两个前后向使用相同Dropout掩码，恢复原权重后普通AdamW更新。erm48为48轮普通CNN作为近似前后向次数对照，复用原24轮普通CNN作为强基线。lr=.001/decay=.001/cosine最小1e-5/seed2026/batch64/CPU1线程。主尾平均为13–24轮与25–48轮，次验证CE选模不可替换主结论。

主SAM须较原24轮tail高至少0.5pp、较48轮tail高至少0.25pp，对两者各至少4/6用户提高、每用户下降不超1pp；否则停止该rho/decay/seed/window网格和八折扩展。通过也仍需同协议八折、多种子和领域贡献证据。六人反复研究、单种子不是独立确认，不与旧93.4417%直接比较。

TESTS.json记录解析梯度、随机性、异常恢复、完整优化器恢复、独立前向与留出隔离通过。COST_INITIAL.json记录原5h估算门槛未过；COST.json复用相同真实源批时间，在预注册前只修改资源预算至10h，估计8.664782h含75%余量。没有重新解释失败的效果门槛，没有候选诊断调参。真实普通首epoch与已完成旧基线逐位吻合。

拟运行目录/workspace/uwb-sam-source-runs-20261004，日志/workspace/uwb-sam-source-train-20261004.log。12条新轨迹共432epoch，6旧基线复用；保存全部epoch段、顺序ID、完整model/AdamW/scheduler/RNG/tail/best。最终18条轨迹全部ERM/tail验证/诊断独立功能前向、角色和SHA审计，自动本地归档，再由后续接续私有Drive认证上传回下载校验；本地归档不等于远端备份。

## 正式启动观测：2026-10-04T01:34:18.058690+00:00

冻结提交 `4c4a4c65696b1510f228504e1375de3223fedbc6` 已正常推送，18文件从远端拉回逐字核验后才启动。训练PID1398、本地归档PID1410均实际存活；当前sam24/held3第4轮，完整轨迹0/12。精确命令见LAUNCH.json/ARCHIVE_LAUNCH.json，状态见START_OBSERVATION.json。此前“待启动”为历史记录，当前运行中，无完整候选识别结果。

第2轮step148完整恢复快照（含AdamW/scheduler/RNG/model/tail/best、两轮segment及精确顺序）7,245,703字节、28成员，私有Drive owner-only上传、认证raw下载及整包/成员SHA均通过，见START_DRIVE_RECEIPT.json。START_AUDIT独立函数前向重算该快照全部验证logits/CE逐位相同，参数对应AdamW状态完整；没有评测候选诊断输入。它只是启动备份，不是12轨迹最终备份。研究目标尚未达成。
