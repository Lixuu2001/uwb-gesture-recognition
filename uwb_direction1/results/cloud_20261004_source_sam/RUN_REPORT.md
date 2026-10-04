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


## 2026-10-04T03:42:27.438568+00:00 五条完整源SAM/计算量对照轨迹审计及私有备份通过

原PID1398/1410保持活动，观测erm48/held5第30轮，5/12完整轨迹。预注册4c4a4c6训练代码/helper/协议哈希均未变，无重复启动、无中途参数或门槛调整，尚无整组六源效果判断。

固定本次审计集合：sam24/held3,4,5与erm48/held3,4共5条、168epoch。全部ERM/tail验证和诊断logits独立函数前向逐位相同，198产物SHA及角色、源样本顺序、选模通过。完整203成员（包含全部168轮段平均与五份最终model/AdamW/scheduler/RNG、ID/logits/配置），共204381081字节，12逻辑包、33个≤7MiB块，已私有owner-only上传并认证raw回下载，逐块/重组整包/成员SHA通过。原首条5个已验证块复用，未重复上传。它不含正在运行轨迹当前检查点，也不是全部12轨迹最终备份。

持久索引results/cloud_20261004_source_sam/PARTIAL_DRIVE_RECEIPT_02.json、审计PARTIAL_AUDIT_02.json、观测MONITOR_02.json。归档/workspace/uwb-sam-final-20261004和transport/PARTIAL_INDEX_02、records/可供最终备份复用。继续原固定队列；完成后全18轨迹独立审计、补齐私有最终备份，再按冻结联合门槛判定。原始数据无需迁移，789源码快照不改；目标未达、自动接续保留。以下是历史记录。


## 2026-10-04T04:15:53.658751+00:00 前三位源诊断用户两臂完整审计与私有备份通过

原训练1398及归档1410仍活动，观测erm48/held6第3轮，7/12完整新轨迹。未改预注册4c4a4c6训练源码/helper/协议，未重启或根据中途诊断结果调整任何参数/门槛；仍无整组六源效果结论。

本次新增erm48/held5的全部ERM/tail验证/诊断独立前向审计通过，54产物SHA及48轮样本顺序/角色/选模一致。累计sam24和erm48各held3,4,5共6轨迹、216epoch，252产物SHA；前5条的独立前向结论引用PARTIAL_AUDIT_02，身份一致且全部归档产物重新核SHA，未重复运行其推理审计。见PARTIAL_AUDIT_03.json。

累计258完整成员，包括全部216轮段平均、6份最终AdamW/scheduler/RNG状态和ID/logits，259692648字节、15逻辑包、42个≤7MiB传输块全部私有owner-only上传、认证raw回下载、重组整包与成员SHA验证。旧33块复用；新传输仅本次新增轨迹。索引results/cloud_20261004_source_sam/PARTIAL_DRIVE_RECEIPT_03.json，监控MONITOR_03.json。完整集合不包括正在运行轨迹的当前状态，也不是12轨迹最终备份。

最终归档和上传须复用/workspace/uwb-sam-final-20261004与/workspace/uwb-sam-final-transport-20261004/PARTIAL_INDEX_03.json、records中的已验证文件和ID。继续原固定队列，完成后全18轨迹最终审计，再补私有备份余下产物并按冻结联合门槛判定。不启动下一科学配置或八折；目标未达，自动接续保留。以下为历史记录。


## 2026-10-04T05:47:41.180650+00:00 新增四条源SAM/计算量对照轨迹审计与备份通过

原训练1398及归档1410仍活动，观测erm48/held8第1轮，11/12完整新轨迹。预注册4c4a4c6训练代码/helper/协议未改，未重复启动，未根据中途诊断结果调整科学配置；尚无整组六源效果结论。

本次新增sam24、erm48各held6,7四条独立前向审计通过：全部ERM/tail验证和诊断logits逐位一致，角色、源样本顺序、选模及168产物SHA通过。累计两臂各held3至7共10轨迹、360epoch、420产物SHA；前6条独立前向结论引用PARTIAL_AUDIT_03，身份一致且归档产物全部重新核SHA。见PARTIAL_AUDIT_04.json。

累计430完整成员，包括360轮段平均、10份最终AdamW/scheduler/RNG状态、配置、ID和logits，432806767字节、25逻辑包、70个≤7MiB传输块，全部私有owner-only上传并认证raw回下载，逐块、重组整包与全部成员SHA通过。旧42块复用。首批传输在23个新块通过后有两次接口失败；对已上传块复用记录ID，对未返回成功的上传先精确查名确认远端无文件，再续传剩余5块，最终全部核验通过。持久索引PARTIAL_DRIVE_RECEIPT_04.json；监控MONITOR_04.json。该集合不含正在运行轨迹当前检查点，也不是12轨迹最终备份。

最终归档和上传须复用/workspace/uwb-sam-final-20261004以及/workspace/uwb-sam-final-transport-20261004/PARTIAL_INDEX_04.json、records中的已验证块和ID。继续原固定队列，完成后全18轨迹最终审计、补齐私有备份，按冻结联合门槛判定。已知SAM操作本身不是创新，尚未证明用户目标；不启动新科学配置或八折。自动接续保留。以下为历史记录。

