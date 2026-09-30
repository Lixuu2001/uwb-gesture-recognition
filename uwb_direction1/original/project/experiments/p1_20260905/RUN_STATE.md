# 执行状态与续跑入口

## 新增：用户要求尝试三视角重建

已完成单视角→缺失两视角992维特征补全受控实验，结果在RECONSTRUCTION_REPORT.md和reconstruction_results。13配置×8折=104次训练、11项测试通过。冻结既有折内编码器，所有新对照使用相同特征。joint vs公平单视角基线L/T/R为−1.29/+0.25/−0.33pp；pure_recon NMSE=.730/.897/.805，有部分特征可预测性但没有分类收益。未达到预定扩展闸门；不能把它说成三维轨迹重建或端到端原始回波生成完成。原始第一部分神经结果目录results_clean保持原值。

用户要求：独立核实 Claude 结论与参数，由 Codex 自行定路线并完成整个第一部分结果；只用现有公开数据。不是仅规划。用户再次说“继续”。原始开题 DOCX 不修改。未知手势留第二部分。

## 已完成运行

- 正式清理实验已完成：17 个配置、每个 8 个 LOSO 折次，输出目录为 `results_clean`；没有遗留 `p1_20260905` 训练进程。
- `results` 只保留审计证据和未过滤先导，不能混入正式结果；正式汇总为 `results_clean/RESULTS_LIVE.md`、`model_summary.csv` 和 `summary.json`。
- 追加的 `sv_R_concat_kd` 八折组合实验也已完成，用于检验右视角“幅度+相位”与三视角教师蒸馏能否叠加。
- 由于 `sv_R_concat` 通过 +2 pp、6/8 用户闸门，已补跑匹配的 `sv_R_off` 与 `sv_R_concat` seed=2027；主结果在两个 seed 上均为 8/8 用户改善（+3.84、+4.18 pp）。
- seed=2026 的 17 个配置以及 seed=2027 的 Right off/concat 均有 8/8 JSON 和预测文件；对应 `.err` 日志均为空。分析脚本和 7 项协议测试均在 PyTorch 环境中通过。

## 队列顺序

L/T/R 各 off/only/concat；Left signed 对照；三视角 off 教师 mean/concat；冻结教师→L/T/R 独立学生普通 KD。教师每外层折按训练用户内部验证 CE 选择 mean/concat，不能按测试成绩选择。

候选相对 off 平均增益 ≥2 pp 且 6/8 用户改善，才补 seed2027；concat 阳性则追加 duplicate 幅度复制容量控制；该视角任一输入或 KD 达门槛则跑 self/joint/quality 运动辅助监督。未达门槛允许负结果结案，不宣布所有同类方法都不可能。KD 阳性后仍应考虑简单迁移/标签平滑对照，不能只因普通 KD 阳性就宣称新颖。

## 已完成独立核验

- 当前历史日志：no-fusion CR 三视角 closed_acc=91.71，融合版=89.73；SV Left CR=82.50、Raw=78.00，均是 case0 的6已知类。RESULTS §23 82.64 是开集阈值 Accuracy，非 closed_acc。
- ClosedSetHead 是 2048/256 MLP，不是线性层。闭集指标来自 logits argmax，不来自 inspector 距离集成。
- §49 原文否定特征 mean 恒盲（方向2分类 mean=82.25、L+R=88.91）；0.00 是分类率，不能说物理量数值等于零。
- 576 个 CSV 缓存前缀逐值一致；源数据存在 HV2 G11/G12 各100条三视角完全重复但标签相反，Raw/CR 原始 MAT 也 max_abs_diff=0。
- HV6 G4 CR 的 L/T/R 数量93/93/162。Raw MAT/CSV为93，各格式有微小浮点舍入，不能据此说 Raw 配对错；CR 的对应93条无法独立核证。
- 统一排除 HV2 两冲突类200条和 HV6 G4的93条，共留9300。HV2测试1000条10类、HV6测试1100条11类，余6名1200条12类；保留12类输出。见 `quality_audit.json` 排除索引。后续结论要明确这种不完整性；有效候选可考虑完整六被试的敏感性验证。
- 每视角和整样本哈希只发现上述100对同用户跨标签重复，无跨用户完全重复。
- 七项测试通过：质量/LOSO划分、预处理等价、不受另一样本影响、相位正负及混叠、维度、concat顺序、教师梯度detach。

## 已完成 CPU 基线（正式清理数据）

`physical.py` 是新的明确13维/视角统计，不是找回丢失的旧39维脚本，所以不能声称精确复现或证伪旧86.3。标准化只拟合训练索引，C∈{.01,.1,1}由训练内验证选择；三种类集合单列。

12类 accuracy（8被试平均）

| 视角 | motion | amplitude（10×21粗图） | combined |
|---|---:|---:|---:|
| L |44.50|66.02|70.97|
| T |41.65|60.16|61.79|
| R |46.64|60.84|69.08|
| LTR |75.77|90.08|92.18|

`sign_probe.py` 训练用户正交滑动只校准极性，测试留出用户四个对角，两分类方向（不是12类识别）：Left水平90.66%、Top垂直93.56%、R−L水平93.91%；Top水平46.62，L垂直44.16，体现视角选择性。另存质量加权和训练均值阈值对照，不能挑最高值冒充预定主结果。

## 代码与协议

`vendor` 是当前未提交的 config/data/model 快照，哈希在 results/data_manifest.json。原 main.py、旧 RESULTS.md、用户 DOCX 保留。独立入口避免把 doppler 半接进开集/注册几十条路径。

神经：当前 dual-path dim32/depth4/heads4/d_key0/flatten；当前 ClosedSetHead。CR off/only 是189bin，concat=378bin；concat增参数必须看duplicate控制。Adam1e-4,wd1e-4,batch128,microbatch32,bf16；40epoch上限，验证CE最佳，12epoch后patience8。每外层留一用户，其他7用户每类10%验证，其余梯度训练；超参数未用测试调整。SEED2026和条件SEED2027分别控制初始化/训练内划分，配对臂同一索引和batch顺序。KD=CE+0.5*T²KL,T2。

当前 `train.py` 在结果落盘时计算源码hash。运行期间别改它以免hash和已加载代码不一致；如扩展要先存源码版本，等臂结束后改。aux quality当前直接q加权，可能混入辅助损失强度变化；若它有效，需均值权重控制或训练折归一化再归因。当前只做研究门槛，不默认创新成立。

## 已完成核对与交付

1. `summarize.py` 已生成 17 个模型和 12 个配对比较；`analyze_transfer.py` 已按视角基线修正路径映射并生成 32 个 KD 折次的迁移账本。
2. 结果推断单位是用户，不把种子当额外独立样本；小样本和多配置探索效应已写入统计说明。
3. 第二部分仍需按已知类重训，不能把 12 类闭集 checkpoint 直接当作未知类实验。
4. 原始开题 DOCX 未修改；最终中文报告应链接本目录下的审计、协议、汇总和迁移文件。

## 工具/进程

执行环境 Windows PowerShell，Python `D:\D_conda\envs\pytorch\python.exe`，torch2.7.1+cu126，RTX4060 Laptop8GB，主机RAM16GB。GPU数据按视角float16分块常驻，避免CPU一次加载2GB。保持电脑唤醒已在训练/队列脚本设置，仅当前进程寿命。

GPU 队列和 seed=2027 核验均已退出；CPU physical 已完成。训练日志保存在 `results_clean`，当前没有 `p1_20260905` 进程。无子代理、无定时自动化、无create_goal。
