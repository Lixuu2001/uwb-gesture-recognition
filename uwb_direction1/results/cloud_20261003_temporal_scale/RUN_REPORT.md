# 时序卷积跨度：预注册与可执行性（2026-10-03）

当前状态：实现、行为测试、真实CPU测时通过；本记录时正式12轨迹训练尚未启动，无新效果结果。后续以LAUNCH.json、运行队列与检查点为准。

固定time2只改三块中第二个stride1卷积的时间dilation/padding为2；range2仅改距离轴。均265836参数，与原CNN全部初始化张量/RNG一致。纯卷积跨度分别71×43/43×71/43×43，GN/头仍有全局信息。最小整数扩张2，没有搜索或按留出错误选择。设计依据与未读论文边界见research/TEMPORAL_SCALE_REVIEW.md。

六源用户3–8四训练/下一人验证/一人最终诊断，固定24轮、seed2026、AdamW(.001/.001)、batch64、cosine。复用已审计原CNN六条轨迹，新12条。主结果固定13–24轮逐更新参数平均，验证CE最优仅次结果，不允许事后交换。时间臂相对原CNN≥+0.5pp、距离对照≥+0.25pp、各胜≥4/6、每位差值≥−1pp才推进。全体用户有研究历史，非独立确认；所有主门槛预先固定。

TESTS.json：功能前向、卷积几何、三臂初值/RNG/参数、普通训练器等价；两个扩张臂断点恢复model/optimizer/RNG/selection逐位相同；留出输入和标签更改不影响选模。COST.json：真实首训练batch和原CNN首epoch校验通过，loss/order/segment逐位重现旧强基线；估计3.196h含75%余量，4h预算内。成本探测没有评价候选诊断准确率。

每epoch保存精确样本顺序、segment、model/AdamW/scheduler/RNG/tail/best状态；最后所有验证/诊断logits由独立函数前向审计。保留source/data/config SHA。训练根/workspace/uwb-temporal-scale-runs-20261003；日志/workspace/uwb-temporal-scale-train-20261003.log。原大数组不进Git。

扩张卷积、TCN、InceptionTime均既有技术。这里不是这些完整作者模型的复现，没有创新主张。轴向边界填充比例不同仍是混杂，若通过须补边界控制、多种子和八折。失败不搜核长/扩张率/seed/window。用户目标仍未完成。
