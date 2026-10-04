# 距离带能量时序面积：依据、先例与可否决问题

2026-10-04。当前为候选，不是原创声明。先核对云端与远端86115e9；未重启旧训练。

## 本轮实际阅读

- Jeremy Reizenstein, *Calculation of Iterated-Integral Signatures and Log Signatures*, May2015/revised Dec2017, arXiv:1712.02757。作者仓库 [bottler/phd-docs](https://github.com/bottler/phd-docs) 的 logsignatures.pdf 已取得，读取PDF第1–3、5–7页文本与公式，重点是引言、Chen恒等式、直线段指数及log signature；未通读全文、未声称同行评审发表。精确commit/blob/SHA见同名REVIEW_RECEIPT。
- [iisignature](https://github.com/bottler/iisignature)固定commit447577b55ad07a985c0a39b7c4fbf01ca2b142e5，README全文、doc.md1–180行。A方法明确以三角形面积计算level1/2。其README引用Reizenstein/Graham2020 ACM TOMS *Algorithm 1004: The iisignature Library*，DOI10.1145/3371237；没有冒称已读TOMS主文。作者旧iisignature.pdf是用户文档，非论文。
- [PSFDataset](https://github.com/kschlegel/PSFDataset)固定ccba6042a299c8ca0fc9aaf103f0fd34faedeb2b README全文；这是实现仓库，不据此推断是论文作者仓库。它明确依据Yang/Lyons/Ni/Schmid/Jin *Developing the Path Signature Methodology and its Application to Landmark-based Human Action Recognition* (arXiv:1707.03993)。已存在动作路径签名、延迟变换、分段与级数策略；该论文主文未读，未复现其骨架结果。
- GitHub仓库描述检索signature human action recognition、signature primer Kormilitzin、UWB gesture signature。后两者无结果不构成穷尽查新证据。

## 与原项目及失败路线的关系

原proposal_v2/07_W2判决 §2 已用跨雷达距离轨迹对的有向面积区分G9/G10，报告89.06%（不同任务，不能与12类CNN指标排名）。03/04、traj_w2_predict.py等也已有几何面积。故“首次用有向面积”“路径签名等于新方法”均不成立。

本轮是每雷达内部7个连续距离带的RMS能量轨迹，不估计3D点或跨雷达帧相位，不依赖上轮失败的NCC位移匹配。既有静态range moment、卷积通道对称协方差及跨雷达质心轨迹并不等于这个距离带路径。它仍是已知数学在当前观测上的机制筛选，不是原创贡献。若通过，需要另外核查雷达路径签名近邻及更强同协议对照。

## 前瞻假设和控制

不同距离带能量改变的先后次序可能补充冻结CNN。90帧真实RF各27距离单元平方平均开方，7带，按每片段/雷达最大值归一化；不使用全数据统计。level2反对称项A_ij=1/2 sum((p_t-p_0)_i Δp_j − (p_t-p_0)_j Δp_i)，21维/雷达。保留每带均值/标准差/首末差作63维基础；已有源训练CNN的128维隐藏表示为共同控制。

五臂：CNN读出、CNN+基础、再加signed面积（唯一主候选）、unsigned面积、固定片段ID种子的内部帧置乱面积。后两者均增加63维；置乱保留端点及基础统计，不作“纯因果”解释。全部使用相同训练用户、平衡岭回归、三个预定正则和验证选择。旧CNN tail也作为强比较，不以弱探针改善替代它。

纯面积对路径平移和保留折线的细分不变；真实速度改变可能改变散射幅度或采样，完整描述含均值/std/CNN，也没有整体速度不变性保证。7带压缩可能丢信息；level2不能识别所有路径，回溯段可抵消。此处能量轨迹不是手部空间轨迹。

## 执行纪律

ENERGY_AREA_PROTOCOL.json在任何真实评分前提交远端回读。只源用户3–8四训练/一验证/一诊断，7100条既有质量掩码，全部属于反复研发数据，非独立测试。先源训练64条测成本，预算1800秒含2倍估计余量。禁止诊断标签拟合/选择、重开已失败匹配参数、失败后换主臂/带数/级数/seed/归一化/正则网格。合成Chen递推、方向反转、平移、折线细分、增益及零输入检查先过；全部保存源ID、模型系数、隐藏矩阵、预测、哈希、日志。

主signed须相对base≥0.5pp、unsigned/shuffled各≥0.25pp、CNN读出/tail不降，所有比较至少4/6正差且任一配对用户下降不超过1pp；这是计算继续标准，非统计显著性。失败停止此固定探针；通过也只能启动另行冻结的端到端/八折/消融，不代表研究目标达成。
