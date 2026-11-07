# IRM complete native fold insurance: held4_insured

## 2026-11-07：IRM held4完整64生产＋fresh64/native审计及私有回读完成；已保险折[1, 2, 3, 4]，原六折任务继续，未评分

本次运维故障与修复留痕：初次13块传输中，part011上传返回RemoteProtocolError（Server disconnected without sending a response），没有文件ID；其他12块原始回读通过。保留MISSING_ID_RESPONSE和ERROR，先按精确文件名及原父目录搜索，观测零匹配，核验本地无已知ID/成功上传响应后创建一次性补传锁，仅明确补传此块一次，完成owner-only metadata/raw fetch/立即物化/SHA。未对任何已知ID重传，未改训练/科学源/模型/预算，原进程继续。最终13块/整档/415payload全成员验证均通过；不能把首次12块误说成首次全成功，也不据搜索观测零结果断言无法观测的远端绝无遗留文件。

初次队列的工具进度提示使用completed-attempt数，错误把1次失败也算入verified；实际READBACK文件数和已检查的settled receipt始终明确12成功/1失败，没有据此提前验证或发布完成。已新增纯运维upload_queue_v2.js，verified仅计成功raw-verified回执，failed/attempted/unattempted单列、不自动重试；13模拟完整pipeline/1模拟失败实测报告12 verified/1 failed，max并发3，无真实Drive调用/训练。仅运维文件改变，45科学冻结项不变。后续上传使用该修正队列，失败仍须先检查已知ID，未知ID先精确父目录/文件名搜索，再人工明确单次恢复；所有原错误与修复审计入本次结果。补传/原始回读与原训练墙钟重叠不双加，完整网络及运维CPU未测，不能记0。

先读GitHub最新父7a5d46162c518a5f9547983e9dfd66bd5c4600e1四权威和LOSO纠正两文件，同提交两个大memory正文按远端blobSHA核验。复用同一cloud/boot和原主PID92152/startticks29520187，未新建训练/自动任务、未变频率/启用/环境。用户模板中的旧MMD待补状态已由后续记录取代；已关闭MMD/VREx/Fishr固定六折失败结论、GAP93.6672%和不同预算旧混合94.6290%原失败结论保持，不重训/重评分或改旧科学文件。

本次仅补已完成held4保险：64轮生产4544更新和独立fresh64同等更新，65条epoch/journal及全steps、所选/最终native模型、验证/无held truth输出逐字一致。验证用户5选中epoch62；selectedSHA6cc9d50cd44fcb1b475972f5dee7ec09162d76cc1bb8f9aa1907ba169a66fc47、最终native stateSHA12f625bf4cf45cfcc5213d85f3418d2de85679d08b7b99e3d2ff85a07fc69702。原生optimizer/slow/model/RNG完整journal加载、semanticSHA、globalBCE/IRM/objective/有限值/计数核验通过，reset=[500]、最终native count4044。首步独立dummy-scale autograd核验全部617612参数clipped梯度最大差3.725290298461914e-09；同实现完整重放不单独排除共同实现错误，不冒称后续每步都做独立梯度证明。完整617612/all3 signedRMS/B64/64/seed20261029/globalBCE/IRM100/warm1/anneal500/native reset配置及45科学冻结项、native依赖/旧结果哈希不变；操作没有训练/推断新模型或读取任何held真值/分数。

held4归档13块94341816字节、415载荷（tar416成员），archiveSHA794f0f6fa7e9f14800c912cae3629b5d0eb661e5cd54811edbc9915631f96128。原私有Drive目录metadata唯一owner/shared=false，最多3条完整upload→metadata→raw fetch(download_raw_file=true/include_base64=false)→立即物化size/SHA pipeline，所有块/整档/每个payload原始回读通过，0已知ID盲重传，不输出认证URL或改权限。代码、配置、审计和持久索引入Git，真实ID/权重/预测/原始大数组保持私有。已完成保险held1..4合计52块377804858字节，各归档独立全成员验证；这不是完整六折，不代表以后折已训练/备份，不用于提前评分。新组合索引连接全部已完成折、旧中途快照/初态、原始86文件/163块持久索引与原GAP恢复链；本环境未实际中断恢复/解包，不把raw字节核验冒称为端到端resume已实测。

held4原阶段成本wall4815.576577325992/CPU4813.218616秒完整保留（已含生产/重放/输出）；父主阶段已含相同时段，不能再叠加。当前active held5/train、epoch24/64、1728更新；六折阶段累计wall20667.501/CPU20693.839秒、RSS1704644KiB，整个新增IRM12h wall/CPU和12GiB硬上限不因跨折/归档重置。保险审计+构建wall4.862687/CPU4.856249秒（含audit不再加），raw整档核验wall0.690218/CPU0.687347秒；运维与主任务墙钟重叠不相加，操作CPU单列，未测网络/完整运维CPU不记0。既有MMD30865.434969089023/30891.040083422995（h1子成本不双加）、VREx31157.163366941997/31189.589142517、Fishr29559.94331930898/29599.386025808墙钟/CPU秒仍另保留。

继续原进程完成剩余折64生产→fresh64逐步native审计→无held truth输出。全部六折模型、轨迹、预测、角色和验证选模/原生审计、全模型私有raw-allmember与exactrelease冻结后一次cohort评分。禁止单折评分/按held分数停止剩余折/旧MMD once/测试调参/重训恢复；固定+.5pp、>=4严格胜、最差paired>=-2pp，正负如实保留，正才匹配globalBCE/reset500/no-IRM六折对照归因，负关闭确切配置不救参。accuracy与presentmacro、support/缺类null、equal-user mean/worst/paired/wins和训练/推断/复现/保险成本分别报告。用户7–8不用选择、历史八人探索性非独立验证，研究目标仍未达到，不宣称原创或准确率提升。789原源码和仓库可见性不变。

