# Full LOSO6 fixed DomainBed VREx: release

## 2026-11-06：VREx六折native/资源门槛通过，117载荷未训练初始化已私有回读，待本次精确release

以990154da94367b6648be5b4db746f95e015a1027实际预登记的修复后37源manifest执行原生检查，全6角色均exit0、无资源stop；实际pinnedclass默认10/500与reset、独立解析梯度/gradcheck、真正小optimizer501次/边界500恢复、完整617612梯度/BN/优化器/全部RNG、slow6/disk7/尾28全部通过；角色/不平衡accuracy-vs-macro/缺类null及ownedPID独立session停止玩具测试通过。六组native检查及固定计时总体wall46.675557秒，检查非UWB拟合。只有既定一遍生成B64九步/臂，VREx/native总wall/CPU较大比1.014077892；按评分前冻结公式1.25*30891.040083423*max(1,ratio)+600的保守reserve=39757.400995秒（约11.0437小时），低于新完整6阶段wall/CPU各43200秒、RSS12GiB，gate通过，不用真实pilot/反复测时或缩骨干救预算。原MMD完整8.58h成本继续另计不清零。

未训练state采用新的VRExScheduledOptimizer wrapper（algorithm0/resetEvents空/fastmoment空/slow空），全6初始化同字节b2e5e8cc4edda7a20f05b03a47291bc5774ed456de3cfa5c0715dd1dc8513156，完整state语义75ae4db3d0d0a34305393532fa40c421df375762a7b3f21413bd3e92461295a0，其中模型语义2aae73c58e38ca324027e888955b130a82fbaece7167df3465522ec51986d2d3与原GAP匹配；没有把MMD优化器state错当VREx初始化。全部六折ID/角色、37源/49原生依赖、native检查日志、资源门槛和完整initial组成2块7523103字节117载荷，archive5cab4eb2ee2fafa400ecf37381d7e7fe138e2d5a7f5c5f2e794212c3bd9bfa9a、MEMBERS4badc9260e760e98d4ffdfea47e250bb67b309b85442afb54ff932f47ca8d8a7；owner-only元数据核验后rawfetch(true,false)/立即物化/SHA/整档/每载荷通过。complete_modelset_verified=false准确表示只保护未训练state，不宣称训练模型已备份或恢复完整job。实际ID/role和权重不进Git。

本次先精确发布/逐文件原始回读force=false，随后将实际commit/源manifestSHA/initialfileSHA/archiveSHA写到REAL_RELEASE，核对旧MMD/新VREx模型进程及PID/boot/startticks，才允许启动唯一新VREx完整6阶段。此提交瞬间UWB训练尚未启动，不能据此宣称后台运行。首批更新必须独立nativefirst audit精确通过后才放行训练；每折完整64生产+独立fresh64逐步/轮次/全state/RNG相等和selected-output审计，串行all6，不读任何单人测试分数中途决策。全部6完整私有模型/预测/审计恢复冻结后固定无标签推断一遍及唯一cohort评分。保留phase每秒累计资源账、失败状态/检查点，无自动生产重启。

方法仍为固定已发表DomainBed默认lambda10/anneal500的完整GAP617612/三雷达/RMS/B64/64/seed20261029/原生AdamWLookahead，测试h1..6/valnext/其他四人训练；仅训练验证选checkpoint。GAP93.6672%模型/输出全复用，无重训降低基线。VREx等域均值+完整reset与GAP不同，若主6花费门槛通过，先做评分前已固定等域BCE+同reset/no-variance联合六折对照（同预算10h），未通过不能归因variance/新颖性或直接跳8确认。原MMD6均值40.6258%/0胜/-53.0414pp固定配置失败关闭；旧held1 once和旧VREx conditional route保持未调用，旧科学文件/旧失败不追改。

accuracy、present-class macro/support/缺类null、用户等权mean/worst、配对差值/胜人数/pooled及训练/推断/复现成本分开；冻结>=+0.5pp/>=4胜/最差>=-2pp门槛不变，正负均报告。用户7–8不用于选择，全部历史八人反复开发探索性，不称独立新用户验证或已验证原创贡献。自动任务频率/启用/云环境、Drive私有、仓库可见性、原789快照不变。目标未达，按本次实际release继续唯一完整6可复现实验。

