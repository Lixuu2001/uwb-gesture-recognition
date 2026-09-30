# 本轮实验入口与复现

本轮全部训练、校准、子集分析、条件策略及 GPU 计时已完成；下列是实际使用的入口与等价命令，不是等待执行的实验队列。完整数字、停止条件和统计口径以[最终研究报告](D:/uwb_gestures/experiments/p1_20260928/report_s2026_release_v2/REPORT.md)及对应原始结果为准；报告生成入口见[报告复现说明](D:/uwb_gestures/experiments/p1_20260928/report_s2026_release_v2/README.md)。模型种子固定为 2026；未因测试结果改参数，未启动仅在通过门槛后才允许的第二种子或 adapter 扩展。

## 环境与文件

- Python：`D:/D_conda/envs/pytorch/python.exe`，Python 3.9.23、PyTorch 2.7.1+cu126、NumPy 2.0.2、scikit-learn 1.6.1。
- 代码工作树：`C:/Users/19252/.codex/worktrees/uwb-user-calibration/uwb_gestures`，分支 `codex/user-calibration-20260928`。
- 新代码：工作树内 `experiments/new_routes`。
- 输入数据及原模型定义：`D:/uwb_gestures/experiments/p1_20260905`，真实已去杂波输入在 `D:/uwb_gestures/cache`。
- 新产物：`D:/uwb_gestures/experiments/p1_20260928`。原项目、旧实验及 Word 保留。

结果绑定源代码、输入、分区、权重及预测的 SHA256。默认目录已经包含本轮结果；脚本各自执行身份校验、恢复或拒绝覆盖。复核无需删除结果或重新训练。新计时和新报告必须使用新的空输出目录。若另开完整实验，应一致指定各脚本的 `--out`、`--baseline`、`--models`、`--budget` 等依赖，不能让新旧身份混用。

## 本轮实验顺序

在 PowerShell 中进入以下目录：

```powershell
Set-Location 'C:/Users/19252/.codex/worktrees/uwb-user-calibration/uwb_gestures/experiments/new_routes'
```

1. Right 基础模型训练与冻结激活导出。默认八折，完整模型训练及导出已经完成。

```powershell
& 'D:/D_conda/envs/pytorch/python.exe' train_baseline.py --seed 2026
```

2. A：固定支持划分、源统计模型、验证选择、目标用户查询与用户层汇总。

```powershell
& 'D:/D_conda/envs/pytorch/python.exe' run_calibration.py
```

3. B：从对应折 Right 初始化共享真实视角子集模型，训练及保存七子集预测。

```powershell
& 'D:/D_conda/envs/pytorch/python.exe' multiview.py
```

4. B：验证选择固定组合与简单级联，比较成本上限内的混合策略和不可部署 oracle。

```powershell
& 'D:/D_conda/envs/pytorch/python.exe' budget_analysis.py
```

5. 仅因事前规定的 oracle 机会门槛通过，执行已冻结的条件策略先导；不是按目标结果选择策略模型。

```powershell
& 'D:/D_conda/envs/pytorch/python.exe' value_policy.py
```

6. GPU 训练退出后测实际处理路径。以下记录的是本次实际参数；`52016` 是本次已核对身份的空闲 QQ GPU 进程，不能把该 PID 视为另一日测量的自动豁免。完整例外规则见 LATENCY_ENVIRONMENT_NOTE.md。

```powershell
& 'D:/D_conda/envs/pytorch/python.exe' benchmark_latency.py --training-finished --allow-idle-gpu-pid 52016 --policy required
```

该默认目录已有冻结结果，重复计时会被拒绝。需要真正再次计时，应先核对当前运行进程及环境，再指定新的 `--out`；不能把重放已有耗时数字称为第二次硬件实测。计时不能与其他训练或繁忙 CPU 审计同时运行。

7. 报告通过 CPU 重算原始归档生成。只读核心审计不写报告；完整报告生成命令见最终报告目录 README.md。

```powershell
& 'D:/D_conda/envs/pytorch/python.exe' report.py --audit-core
```

## 独立复核入口

A、B、条件策略与延迟分别有独立审计文件。它们重算或核对已有模型/预测/选择/计时归档，不能替代新用户验证或重新采集数据。

- A：`calibration_s2026/independent_audit_a_20260928.py`，对应 INDEPENDENT_AUDIT.json/md。
- B：`budget_s2026/independent_audit.py`。
- 条件策略：`value_policy_s2026/independent_audit.py`，还独立重拟合固定小型收益模型并逐树比较。
- 延迟：`latency_audit_s2026/independent_audit.py`；与 `latency_s2026/FROZEN_SHA256.json` 一同核对，不再调用 CUDA 测量。

核心单元与回归检查：

```powershell
Set-Location 'C:/Users/19252/.codex/worktrees/uwb-user-calibration/uwb_gestures'
& 'D:/D_conda/envs/pytorch/python.exe' -m pytest experiments/new_routes -q
```

最终测试结果及交付路径见完成清单。`smoke*` 目录只用于程序执行验证，不进入正式性能表或门槛判断。两候选机制的停止决定保持在各自 summary.json 中；不能通过替换报告指标、四舍五入成本或挑选更有利预算点把 STOP 改成 GO。
