# 源端SAM：云端正式训练已启动（2026-10-04）

当前没有候选诊断效果结论；本报告为启动前记录，之后以LAUNCH/START_OBSERVATION和STATUS为准。已有SAM不是原创贡献。论文主文未读，作者代码/README阅读边界见research/SOURCE_SHARPNESS_REVIEW.md。

固定三路普通SignalNet265836参数；6源用户3–8四训/一验证/一诊断。sam24为24轮rho=.05的CE-only全参数SAM，两个前后向使用相同Dropout掩码，恢复原权重后普通AdamW更新。erm48为48轮普通CNN作为近似前后向次数对照，复用原24轮普通CNN作为强基线。lr=.001/decay=.001/cosine最小1e-5/seed2026/batch64/CPU1线程。主尾平均为13–24轮与25–48轮，次验证CE选模不可替换主结论。

主SAM须较原24轮tail高至少0.5pp、较48轮tail高至少0.25pp，对两者各至少4/6用户提高、每用户下降不超1pp；否则停止该rho/decay/seed/window网格和八折扩展。通过也仍需同协议八折、多种子和领域贡献证据。六人反复研究、单种子不是独立确认，不与旧93.4417%直接比较。

TESTS.json记录解析梯度、随机性、异常恢复、完整优化器恢复、独立前向与留出隔离通过。COST_INITIAL.json记录原5h估算门槛未过；COST.json复用相同真实源批时间，在预注册前只修改资源预算至10h，估计8.664782h含75%余量。没有重新解释失败的效果门槛，没有候选诊断调参。真实普通首epoch与已完成旧基线逐位吻合。

拟运行目录/workspace/uwb-sam-source-runs-20261004，日志/workspace/uwb-sam-source-train-20261004.log。12条新轨迹共432epoch，6旧基线复用；保存全部epoch段、顺序ID、完整model/AdamW/scheduler/RNG/tail/best。最终18条轨迹全部ERM/tail验证/诊断独立功能前向、角色和SHA审计，自动本地归档，再由后续接续私有Drive认证上传回下载校验；本地归档不等于远端备份。

## 正式启动观测：2026-10-04T01:34:18.058690+00:00

冻结提交 `4c4a4c65696b1510f228504e1375de3223fedbc6` 已正常推送，18文件从远端拉回逐字核验后才启动。训练PID1398、本地归档PID1410均实际存活；当前sam24/held3第4轮，完整轨迹0/12。精确命令见LAUNCH.json/ARCHIVE_LAUNCH.json，状态见START_OBSERVATION.json。此前“待启动”为历史记录，当前运行中，无完整候选识别结果。

第2轮step148完整恢复快照（含AdamW/scheduler/RNG/model/tail/best、两轮segment及精确顺序）7,245,703字节、28成员，私有Drive owner-only上传、认证raw下载及整包/成员SHA均通过，见START_DRIVE_RECEIPT.json。START_AUDIT独立函数前向重算该快照全部验证logits/CE逐位相同，参数对应AdamW状态完整；没有评测候选诊断输入。它只是启动备份，不是12轨迹最终备份。研究目标尚未达成。

## 2026-10-04T01:59:56.962967+00:00 首条完整SAM轨迹独立审计和私有备份通过

当前云端原PID1398/1410均仍活动，未中断、未重启；观测erm48/held3第15轮，1/12完整轨迹。仍保持预注册4c4a4c6的模型/协议/helper哈希不变，不因中途分数改配置，尚无整组六源效果判断。

sam24/held3全24轮已完成，ERM/tail全部验证/诊断logits由独立函数前向重算逐位一致，30产物SHA、源角色/24轮精确样本顺序/选模核验通过。31个完整产物（含最终AdamW/RNG、全部segment、ID/logits）分2包共31241969字节、5个≤7MiB传输块，owner-only私有上传、认证raw回下载后逐块/重组/成员SHA全部一致。不是仅epoch2启动快照，也不是全部12轨迹备份。见results/cloud_20261004_source_sam/PARTIAL_AUDIT_01.json、PARTIAL_DRIVE_RECEIPT_01.json、MONITOR_01.json。

后续最终备份复用本次5块及2个不可变归档，不重复上传；本地索引/workspace/uwb-sam-final-transport-20261004/PARTIAL_INDEX_01.json及records保存持久ID与已验证下载路径。仍等原固定队列和最终18轨迹完整审计，之后私有上传余下全部产物，按冻结门槛处理。创新目标尚未达成，自动接续保留。

