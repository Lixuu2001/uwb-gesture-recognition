# IRM complete six-fold native trajectories frozen, private transfer pending: all6_frozen

## 2026-11-07：IRM六折64生产＋fresh64/native全部完成并冻结；原进程已正常结束，75块全模型私有回读进行中，未评分

最新父cfec5964760c73a0226f4746bb39a5ed8342a2ea四权威及LOSO纠正两文件已读，两个大memory同提交blobSHA核验。仍在同一cloud/boot，原PID92152/startticks29520187已正常结束并为Z，STATE为all_six_method_folds_frozen_unscored_requires_complete_private_backup_before_cohort_once。没有在后台训练、没有重启或新训练/自动任务，也没有使用用户电脑。root后台状态本次改为false，原STATE/旧启动及运行观察保留原样。

六折全部64生产＋独立fresh64的模型/slow/native optimizer/全RNG、65条epoch/journal、全部逐步记录、所选模型与无held truth输出精确相同；追加只读核验all6实际role IDs/subject和6900唯一source IDs、各角色互斥及h%6+1验证、训练用户集合、初态SHA、计数/reset500/globalBCE/IRM/有限值和冻结输出SHA全部通过。生产总27776更新、独立重放再27776，不把重放称为额外训练预算。仅训练/验证选模，没有读取gesture truth或算新测试分数；独立首步dummy-scale梯度审计每折617612参数通过，完整同实现重放不能单独排除共同错误。45项科学冻结SHA857e06d91fd2da1ac1428a2716e1df541db8198245dc1d8041922f8d75ad8b6d不变，原617612/all3 signedRMS/B64/64/seed20261029/IRM100,warm1,anneal500/native reset配置不改。

|held|validation|train n|updates each trajectory|validation-selected epoch|
|---|---|---:|---:|---:|
|1|2|4700|4736|46|
|2|3|4700|4736|62|
|3|4|4500|4544|5|
|4|5|4500|4544|62|
|5|6|4600|4608|47|
|6|1|4600|4608|6|

新增IRM整个六折阶段wall29509.953316993022/CPU29548.175221702秒，约8.20h，低于12h wall/CPU硬上限；原生各阶段细分日志保留，旧MMD30865.434969089023/30891.040083422995、VREx31157.163366941997/31189.589142517、Fishr29559.94331930898/29599.386025808墙钟/CPU秒另保留不抹去（旧MMD第一折子成本不双加）。追加只读全六审计wall2.8860604310175404/CPU2.87646879秒单列。全档构建wall见FULL_ARCHIVE_BUILD_COST，记录的parent_process_CPU不含子进程构建CPU；该子CPU未聚合/未测，不当完整构建CPU或0。完整网络/运维CPU尚未测，继续明确限制。

已按原注册operator/full_archive.py一次构建全部六折完整生产/重放/native/models/roleIDs/未评分预测/依赖/初态和恢复布局：75块548348323字节、1995payload，archiveSHA3436cbcc7bb738e30e5279acc9c998c3468a10954e66aa68ad9b08440494cf10。模型/样本ID/大数组不进Git。目标仍原owner-only/shared=false私有Drive目录，元数据已核验，正在用修正upload_queue_v2逐块upload/metadata/raw fetch(download_raw_file=true/include_base64=false)/立即物化sizeSHA，最多3整条流水线，已知ID不重传、失败不自动重试。此记录是全模型已本地冻结及归档构建，不是75块远端全部原始回读完成；前四折已有各自独立完整私有保险，原held4断连和计数修复证据全部保留。

LIVE_FINALIZATION_PLAN指定：先完成全部75块、整档及每成员raw校验，构建组合恢复索引，最新父force=false精确Git回读后才绑定FULL_PRIVATE_RELEASE；再按既定protocol只做一次盲配对推断成本并逐字核验冻结logits，最后一次统一六折评分。不能先算任何held分数/扫测试/用单折挑是否继续，不能重训任何已完成模型。固定+.5pp/4胜/最差paired>=-2pp，报告per-user accuracy与presentmacro/support/缺类null/equal-user mean/worst/paired/wins和成本；正才匹配reset500/globalBCE/no-IRM六折归因对照，负关闭确切配置不救参。用户7–8不用选择、历史八人探索性非独立验证，尚无新准确率或原创结论。原789/可见性/自动频率启用/云环境不变。

