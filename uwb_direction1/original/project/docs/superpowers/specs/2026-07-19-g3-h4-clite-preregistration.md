# G3/H4 c-lite 预注册(v0.2 锁版,2026-07-19)

基于 GPT 审定稿 v0.1 + 三处修正(经 07-19 对话审定)。G2-v1 FAIL 永久保留(RESULTS §45),
不重判分;G3/H4 为独立次要假设,不是 G2 补考。本预注册锁定后不得看测试结果调整。

**接口核查记录(诚实条款)**:G2 主网格路由用固定 s=1.5(`p4_g2_stream.py --s-route`
默认),判决为同 bank B=5 对 B=0,未叠加 calib-select。P/C 两门与路由尺度解耦,
不改变 FAIL;G3 起全部系统统一用逐折 s_cal(各系统用自身校准分数选择)。

## 授权范围

- 仅授权:G3-S0 因果诊断、G3-S1 污染视角定位早筛;双 PASS 后才授权正式 G3;
- 不授权:G2-v2、Adapter、主编码器更新;
- 自动采纳退出候选主方法,不再调 q80/cap/伪标签权重。

## S0 六系统(必要控制)

| 系统 | 距离聚合 | e_recon 用法 | 用途 |
|---|---|---|---|
| v1-replay | mean(D)/mean(σ) | base 列加 z(E) | 精确复现旧 G2 (回归门) |
| legacy-L1 | 同上 | base 列加 z(E) | 用 s_cal 的正式旧基线 |
| legacy-L0 | 同上 | 无加性惩罚 | 检验 λ=1 是否根因 |
| uniform-rv | mean_v(D_v/σ_v) | 无惩罚 | 隔离聚合公式变化 |
| reliability | Σ_v w_v(D_v/σ_v) | 只生成视角权重 | G3 候选方法 |
| oracle-mask | 已知污染视角权重 0 | 仅评估器可见 | 可达天花板(仅单视角) |

每系统:用自身校准分数重算 μ/σ/q;用自身校准距离选逐折 s_cal 并跨漂移条件冻结;
不共用旧阈值;测试标签只进最终评估。
**注(修正 5)**:校准数据干净 ⇒ reliability 的校准侧 w_v≈均匀 ⇒ 其 s_cal≈uniform-rv
的 s_cal——特性而非缺陷(尺度选择不受权重方案污染)。

## 可靠度定义(锁死)

m_v = 逐视角原始 masked-recon MSE;e_v = m_v / (median(m_cal,v)+1e-8);
u_v = 1/(e_v+1e-8);w_v = clip(u_v, p5(u_cal,v), p95(u_cal,v)) 后跨视角归一;
all-view alert = 三视角同时 e_v > 各自校准 p99。不看测试结果调整标准化/百分位。

## S0 三道门

1. **工程回归门**:v1-replay 对现有 B=0 dump:预测标签/预测集/covered 完全一致,
   连续分数 max err ≤1e-5。不过则停,先修运行器。
   **(v0.2.1 证据修订,07-19)**:字面 1e-5 对"另一进程产的旧 dump"不可达——
   `engine.py` 的 `cudnn.benchmark=True` 自动调优使 inspector 矩阵核跨进程漂移
   (D 均匀 ~1e-5..3e-4 波动;E 逐位稳定;校准行与 q 逐位一致;进程内重打分与
   G3 dump 逐位 0.0,证明公式与管线精确)。修订为:分数 ≤5e-4(实测漂移上界)
   **且** 决策级(pred/预测集/covered)与 G2 B=0 日志完全一致。冒烟 2 折已 PASS
   (决策零翻转,边际远大于核漂移)。
2. **根因门**(legacy-L0 vs L1,流调度口径,与 §45 数字同源可比):
   {drop, occlude_vall, domain_vall} 至少 2/3 条件 cov_b 提升 ≥5 分,每个通过条件
   ≥6/8 测试用户方向为正,clean H 降 ≤1 分。不满足则撤回"λ=1 是主要凶手",暂停 G3。
   覆盖恢复但 rejU 降 >10 分:只判"找到病因",不称"完成修复"(预期结局大概率如此)。
3. **可达性门**(oracle-mask vs uniform-rv,单视角污染,静态严重度网格 AUC):
   **atten 记为预期中性(解析恒等,§44),门槛 4/5 个非 atten 族**(修正 1)提高
   H-退化曲线面积。不过则 H4 停止。

## S1 污染视角定位门

单视角 v0/v1/v2、严重度固定 0.5(静态网格),6 族逐族:macro-AUROC ≥0.75 且 Top-1
定位准确率 ≥0.60。总门:≥4/6 族(atten 允许不过);c0、c1 各自 ≥3/6(防词表反转被
池化掩盖);clean 流 all-view alert 率 ≤5%。*_vall 只作压力测试不计门。
S1 不过:删 H4,不试 Adapter。

## 正式 G3 判据(S0、S1 双 PASS 后)

reliability vs uniform-rv,**静态严重度网格**(修正 2:B=0 冻结系统与流顺序无关,
静态网格每档 n≈1150 对流内 ramp 档 ~57;唯一例外=工程回归门用流回放):
严重度 {0,.25,.5,.75,1} 逐档 H-mean → 梯形 AUC;对 v0/v1/v2、两词表、测试用户
预先固定宏平均;ΔAUC_H>0 记该族改善。
**PASS = 非 atten 族 ≥4/5 改善**(修正 1)且 clean H 对 uniform-rv 和 legacy-L0
均降 ≤1 分。报告测试用户聚类 bootstrap 区间但不重改判据。
**安全门**(过不了则不进 G2-v2):clean rejU 降 ≤5 分;任一族 rejU 降 ≤10 分;
不得靠预测集全面变大换覆盖。

## 实施形态(修正 3:一次前向,全部后处理)

运行器只 dump 原始量:逐视角距离张量 D(n,C,V) + 逐视角 E_views(n,V) + 校准侧同款
——六系统、S1 定位、G3 判决全部为纯 numpy 后处理,无需逐系统重跑网格。
阶段:CPU 单元测试 → 合成冒烟 → c0 两折真实冒烟 → 回归门零漂移 → Phase A
(流调度 {clean,drop,occlude_vall,domain_vall} × 两词表 × 8 折 → S0 门 1/2)→
Phase B(静态单视角网格 6 族 × 3 视角 × 4 档 + clean → S0 门 3 + S1 → G3)。

## 分支

S0 或 S1 FAIL → 路线 a(论文=尺度对齐+机制负结果);G3 FAIL → 删 H4 不做 G2-v2;
G3 PASS 但安全门 FAIL → 只作机制结果;全通过 → 另写 G2-v2 预注册(主配置
confirm-only + frozen sigma + reliability),终判尽量在未参与开发的 c2-c5。
