# 时序卷积跨度：云端执行中（2026-10-03）

12条新轨迹的正式云端训练已启动。预注册提交 `09239d6a54b82d22972b945c6a551c78ea0daf21` 已推送并逐文件验证；当前训练进程与真实epoch检查点均已观察到，详情见START_OBSERVATION.json。尚无完整效果结果，不能称为成功或创新。

初始epoch2恢复快照已私有备份并认证下载校验：7,245,346字节，26成员文件，owner-only；完整训练尚未备份。START_AUDIT独立重算全部验证logits与CE逐位一致，未评价诊断输入。训练PID、精确命令见LAUNCH.json；完成后的归档watcher见ARCHIVE_LAUNCH.json。自动接续仍启用。

## 启动前固定设计与检查（历史记录）

当前状态：实现、行为测试、真实CPU测时通过；本记录时正式12轨迹训练尚未启动，无新效果结果。后续以LAUNCH.json、运行队列与检查点为准。

固定time2只改三块中第二个stride1卷积的时间dilation/padding为2；range2仅改距离轴。均265836参数，与原CNN全部初始化张量/RNG一致。纯卷积跨度分别71×43/43×71/43×43，GN/头仍有全局信息。最小整数扩张2，没有搜索或按留出错误选择。设计依据与未读论文边界见research/TEMPORAL_SCALE_REVIEW.md。

六源用户3–8四训练/下一人验证/一人最终诊断，固定24轮、seed2026、AdamW(.001/.001)、batch64、cosine。复用已审计原CNN六条轨迹，新12条。主结果固定13–24轮逐更新参数平均，验证CE最优仅次结果，不允许事后交换。时间臂相对原CNN≥+0.5pp、距离对照≥+0.25pp、各胜≥4/6、每位差值≥−1pp才推进。全体用户有研究历史，非独立确认；所有主门槛预先固定。

TESTS.json：功能前向、卷积几何、三臂初值/RNG/参数、普通训练器等价；两个扩张臂断点恢复model/optimizer/RNG/selection逐位相同；留出输入和标签更改不影响选模。COST.json：真实首训练batch和原CNN首epoch校验通过，loss/order/segment逐位重现旧强基线；估计3.196h含75%余量，4h预算内。成本探测没有评价候选诊断准确率。

每epoch保存精确样本顺序、segment、model/AdamW/scheduler/RNG/tail/best状态；最后所有验证/诊断logits由独立函数前向审计。保留source/data/config SHA。训练根/workspace/uwb-temporal-scale-runs-20261003；日志/workspace/uwb-temporal-scale-train-20261003.log。原大数组不进Git。

扩张卷积、TCN、InceptionTime均既有技术。这里不是这些完整作者模型的复现，没有创新主张。轴向边界填充比例不同仍是混杂，若通过须补边界控制、多种子和八折。失败不搜核长/扩张率/seed/window。用户目标仍未完成。

## 运行中完整性检查（2026-10-03T22:10:05.020638+00:00）

首条time2/held3完成：两种预定权重选择的全部验证/诊断logits经独立功能前向逐位一致；30产物SHA和24轮源样本顺序、角色、选模通过。31文件完整归档两包共31,229,182字节，本地重读及不可变复用通过，尚未上传。此次不汇总中途准确率或更改冻结门槛。观察时1/12完成，后台训练和归档watcher均存活，详见PARTIAL_AUDIT_01.json/MONITOR_01.json。
