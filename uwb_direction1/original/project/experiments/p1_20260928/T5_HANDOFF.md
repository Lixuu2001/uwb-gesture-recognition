# T5 报告实现交接（模型切换安全点）

2026-09-29。按主代理要求立即停止继续实现/计算，交由明确 GPT-6 Sol / xhigh 的后继代理接手。没有提交 Git，没有运行正式报告，没有使用 CUDA。

## 文件与授权边界

- 仅修改工作树 `C:/Users/19252/.codex/worktrees/uwb-user-calibration/uwb_gestures/experiments/new_routes/report.py` 和 `test_report.py`。
- 本交接说明由主代理刚刚明确要求，保存在结果根目录。
- 算法、原计划、冻结源代码均未修改。基准延迟环境补丁由另一实现者完成，最新实现权已交回报告；现再次安全交接。

## 已完成

- 已读 REPORT_REQUIREMENTS、PREREGISTRATION、AMENDMENT_001/002/003、CONDITIONAL_POLICY_PLAN、POLICY_COMPARATOR_CLARIFICATION、LATENCY_PROTOCOL、LITERATURE_NOTES、LATENCY_ENVIRONMENT_NOTE、INTERPRETATION_NOTES。
- 先写并观察失败，再实现并通过15项 CPU unittest：用户内支持重复平均、present/fixed12/未校准 F1、严格2路预算、错/重复 ID、跨路线顺序/标签、摘要非有限值/篡改、缺折/smoke阻断、哈希/路径越界、idle缺失、joint-only rescue、延迟重复/重放失败、验证首路选择、A逐用户伤害门槛、策略强对照逐一门槛、披露桌面上下文的正式CUDA环境。
- `report.py` 已写入：A全部验证网格/支持估计/目标预测CPU重放；B七子集/9档预算/原摘要重算；策略价格及因果路由重放；八用户统计、同7440查询比较、动作/空类与收益分解、来源与冻结产物哈希核验、训练history费用。
- `--audit-core` 只读不写正式文件。正式路径必须完成全部必要输入与延迟审计，验证后才建立staging，最后发布报告目录。
- 最后一次补丁已加入中文REPORT.md组装、JSON、多个CSV、PNG/SVG两图、复现README、source_snapshot及ARTIFACT_SHA256清单；这次补丁之后**尚未跑测试/语法检查/实际数组审计/渲染**。
- 最后已实际运行的测试命令（在上述最后补丁之前）结果：15 tests PASS，约0.165秒。

## 尚缺及优先核查

1. 首先运行 `test_report.py`，然后 `--audit-core`。本代理从未对真实八折产物运行此生成器，可能有真实schema/字段或微小数值兼容问题；必须基于现场证据修复，涉及科学行为时新增失败测试后修复。
2. 主代理刚告知正式 GPU 279×7=1953 条记录已完整完成并解除quiet。等待/读取主代理冻结的 `latency_s2026/FROZEN_SHA256.json`（state=frozen、files为相对路径→SHA）；当前audit_latency要求19个文件：summary/preflight/gpu_idle_checks，加8折observations/summary。其source环境补充由summary.hash_inventory覆盖。
3. `require_inputs()` 当前未将FROZEN_SHA256列入最前的存在性检查，而audit_latency最后会严格要求。可考虑补入，以便缺冻结清单时更早拒绝，避免先做耗时CPU审计。
4. 目前读取并验证了独立B审计（168963项，scope不含policy/latency）。主代理另提供 `value_policy_s2026/INDEPENDENT_AUDIT.json/md` PASS，56301 checks、0errors，重fit24模型/4800trees及208价格grid/全部路线完全一致；**报告暂未纳入这个可选独立policy审计引用**。审计重算6.28秒不能补成原始policy拟合时长。
5. 完整报告生成后必须查看两幅PNG（尤其B图底部legend和密集点）、核查中文文字/坐标/单位/可编辑SVG。当前图从真实结果生成，不含mock数据；尚无输出图片。
6. 报告默认输出 `D:/uwb_gestures/experiments/p1_20260928/report_s2026`；原工作树及结果不可覆盖。staging中README复现目标会改成最终out路径。未完成时不要声称交付完成。
7. 可检查是否需要把oracle的gesture/idle分组也补入：当前B主表/图列oracle实际期望成本，但B_groups只含可部署固定/混合/级联/两policy，oracle分组未列。
8. 最后计算report.py/test_report.py的SHA并回报主代理。为遵守本次立即停止计算指令，本交接未计算最新SHA。

## 后续命令（必须在GPU计时结束后运行）

```powershell
Set-Location 'C:/Users/19252/.codex/worktrees/uwb-user-calibration/uwb_gestures'
& 'D:/D_conda/envs/pytorch/python.exe' -m unittest discover -s experiments/new_routes -p test_report.py -v
& 'D:/D_conda/envs/pytorch/python.exe' experiments/new_routes/report.py --audit-core
& 'D:/D_conda/envs/pytorch/python.exe' experiments/new_routes/report.py --out 'D:/uwb_gestures/experiments/p1_20260928/report_s2026'
```

## 不可丢失的解释边界

- A STOP，rank2未校准68.951373% vs验证简单72.751147%，−3.7997737 pp，1/8正，CI[−9.87919263,−.20805897]。低秩数值收敛不等于机制成立，高预算centerMSE改善未转化accuracy，不将机制失败解释为所有个性化无效。
- B固定LTR93.441667%、TR90.583333%、验证选两路89.536932%。TR是看完研发结果后的固定组合描述，后续工程选择待新用户确认。
- 主cascade91.920833%，实际2.0046155303路，严格超过2，不得四舍五入通过。
- ExtraTrees90.8632576%、2.007873106路，STOP。相对cascade−1.057575758 pp，0/8正；对classgain虽+3.299 pp却多0.221826路；对候选成本内混合+2.1833 pp且6/8不能覆盖其他失败；距三路−2.5784091 pp。
- 机会oracle95.045455%，两门槛通过只触发conditionalpilot，非机制成立。oracle仅当前冻结classifier/验证首路/monotoneappend族，使用真实标签和分数随机化；图横轴必须其实际期望花费。
- 固定随机混合为精确期望；candidate/cascade为实际执行。动态仍3部硬件全安装，省处理≠省硬件/采集功率。90帧20Hz约4.5秒，测量从已采好片段可用开始。
- 跨路线只能同7440 queries；B正式accuracy门槛仍9300。HV2 idle质量排除，留缺失，不能填0。
- 279 latency不是accuracy样本替代。允许明示`desktop_context_gpu_latency`，`gpu_exclusive=false`；QQ空闲PID例外检查不保证计时全程没有桌面噪声。不要仅因`scientific_latency_claim:false`拒绝完整正式测量；该字段保守声明本身非运行失败。
- 8研发用户/1seed、fold源用户重叠、无外部/跨天/连续流确认、bootstrap探索性。policy验证用户曾用于encoder checkpoint选择，内部复用有局限；oracle机会已用targetlabels，不能称策略前从未读取目标标签。
- 原始统计/policy拟合秒未记录，诚实空缺；A适配query毫秒是缓存特征运算，非端到端。旧稿22.7%是oracle测距误差86.4→66.8mm，不是精度/处理节省。
- 两复杂候选都失败，当前只保留真实多雷达工程基础；不包装成论文新机制贡献，不看过test后继续调参找正结果。项目内容一/二避免重复；第二部分never-seen词表须另行重训encoder；不改Word。
