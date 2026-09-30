# 第一部分后续实施计划

**目标：** 在已审计公开 UWB 数据上，明确相位联合输入和多视角蒸馏的单雷达跨用户收益、局限与方法贡献空间。

**执行方式：** 用户已授权规划后直接执行；本任务内顺序实施，耗时训练使用本机独立折进程。新增研究文件留在本实验目录，复用现有 train.py 和 vendor 快照。

## 当前进度

|工作|已完成证据|尚不能声称|
|---|---|---|
|数据审计与协议|9300条质量保留样本；8名用户LOSO；12输出类别；HV2/HV6类支持不完整|未经筛查完整原始数据上的12类结果|
|单雷达输入|L/T/R幅度、相位、联合对照；Right联合相对幅度两seed分别+3.84/+4.18 pp；复制幅度容量控制已跑|任意视角均受益；已证明真实径向速度；独立数据确认|
|教师与KD|三视角concat教师92.27%；Right联合KD73.42%；第二seed为73.80%；普通KD叠加平均+1.20 pp|三视角教师可恢复单视角物理缺失方向；concat纯读出因果收益（容量不匹配）|
|缺失视角补全|冻结L/T/R、端到端Left、门控Left已完成；未通过扩展门槛|三维轨迹已恢复；门控数值为已校准可信度|
|论文贡献|单雷达部署问题和多视角训练信息可用性明确|普通拼接+普通KD已构成充分方法创新|

## 本轮批次 A：立即执行的稳定性与统计复核

- [x] A1. 复核已保存预测与指标，纠正文稿中的确认性/机制过度表述。分清冻结pure_recon NMSE与joint accuracy，防止同表误配。
- [x] A2. 新增 followup_queue.py，复用原train.py，每折先训练teacher_concat seed2027（off输入，三路concat，save_teacher），随后训练sv_R_concat_kd seed2027（concat输入、Right、T=2、权重0.5）。两折并行，保留日志、失败记录和持续状态；不覆盖已有完整产物。
- [x] A3. 完成8折并检查预测/标签/训练和验证索引、教师来源哈希、同输入基准与KD分类头初始化。2027教师已重新训练；禁止直接使用2026教师，因为原split函数随seed改变训练/验证索引。
- [x] A4. 新增followup_analysis.py：分开统计concat−off、concatKD−concat、concatKD−off；按用户聚合两seed差值后bootstrap，不能把16个seed×用户当作16名独立用户；分别计算两条基线的错误新增与损失账本。分析工具已通过测试，两个seed均已齐8折并完成汇总。
- [x] A5. 写入本轮进度与下一批次条件；未齐8折不排名、不决定有效。

固定teacher_concat是依据此前所有2026折的训练内验证选择结果做的后续算法选择，不在2027测试集选择教师。原2026结果仍保留其mean/concat验证选择协议。跨seed汇总说明这项策略差异，并审计2026是否确实全为concat。

本轮用同一批8用户再次运行，称“重复稳定性检查”；seed同时改变初始化和训练/验证随机划分，不能称仅初始化敏感性，也不能称外部独立确认。

## 下一批次 B：有符号信息机制与部署边界

B1的8折已于2026-09-05 23:44完成。2026-09-06补完B2–B4：符号独立增量+1.58 pp但区间跨零；9个视角配置的类别审计完成；RTX4060 Laptop GPU上的batch1成本测量完成。当前26项测试通过。启动时原计划副本保存在followup_results/PLAN_AT_LAUNCH.md。

- [x] B1. Right无符号运动对照8折完成，accuracy70.83%；标准化前去符号，结构和初始分类头与concat匹配。
- [x] B2. 有符号concat比无符号+1.58 pp（5/8，CI跨零）；无符号比duplicate+2.97 pp。支持运动表征补充价值，暂不确认符号为主要增益来源。
- [x] B3. 汇总L/T/R现有结果与混淆矩阵，见followup_results/VIEW_CLASS_AUDIT.md；测试混淆只作事后诊断，不用于训练权重。
- [x] B4. 参数/时延/显存见followup_results/DEPLOYMENT_BENCHMARK.md；独占GPU、batch1计时，不含采集与连续提交延迟。

## 下一批次 C：方法与论文收敛（依赖A/B证据）

- [x] C1. KD叠加两seed分别+1.00/+1.40 pp，保留小幅辅助增益；未达+2 pp研发阈值，不自动把普通KD升级为主创新。
- [x] C2. 三视角信息候选收敛：混淆结构、教师必要性、全局平滑和跨用户折外三视角软标签对照均已完成；未获增量，不立为主创新。
- [x] C3. 已写FIRST_PART_CHAPTER_DRAFT.md，已纳入表征、KD、跨用户折外特权软标签、重建边界与部署限制的最终结果；第二部分陌生手势与第三部分连续决策沿用递进逻辑。

C2第一轮候选已否决：structure74.51%、teacher-uniform74.61%、无教师类别自适应74.65%、匹配全局平滑74.77%。教师相对无教师-0.04 pp，类别自适应相对全局-0.12 pp；当前收益由普通平滑即可解释。最后一项三视角候选采用训练用户内部留一用户的折外软标签，避免教师在自身训练样本上100%拟合；协议见OOF_PRIVILEGED_KD_PLAN.md，状态见oof_privileged_results/queue_state.json。

## 结果和运行入口

Python：D:\D_conda\envs\pytorch\python.exe。

```powershell
& 'D:\D_conda\envs\pytorch\python.exe' experiments/p1_20260905/followup_queue.py
& 'D:\D_conda\envs\pytorch\python.exe' experiments/p1_20260905/followup_analysis.py
```

本轮新增状态和统计保存在followup_results；正式模型仍写results_clean，以便复用原汇总和checkpoint规则。C阶段已完成并按证据收敛；第二部分陌生手势与第三部分连续决策属于后续论文实验，不冒充本轮结果。

## 文献约束

数据集原文支持三雷达及独立访问各视角：[UWB-Gestures, Scientific Data 2021](https://www.nature.com/articles/s41597-021-00876-0)。已有[C²KD, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huo_C2KD_Bridging_the_Modality_Gap_for_Cross-Modal_Knowledge_Distillation_CVPR_2024_paper.html)讨论跨模态软标签错配和选择性蒸馏，所以“加权/筛选教师软标签”本身不能被称为新颖。若论文后续重新提出C2候选，必须明确雷达单视角判别歧义的估计方式、训练/评测隔离和相对普通KD的实际增量；本轮不宣称已证明新的蒸馏方法。
