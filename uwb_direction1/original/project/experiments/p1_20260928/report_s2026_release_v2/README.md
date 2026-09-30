# 复现与文件说明

本轮执行、冻结状态与复现边界见同根目录的[EXECUTION_AND_REPRODUCTION.md](D:/uwb_gestures/experiments/p1_20260928/EXECUTION_AND_REPRODUCTION.md)。

REPORT.md为中文独立报告；computed_results.json保留原精度（准确率为0–1，差为pp，成本为真实处理视角数）。CSV列名显式注明percent、pp、ms或seconds；A_support_repeats.csv保留0–1准确率，不能把支持重复当独立用户。

A_user_effects和B_accuracy_cost各有PNG与可编辑文字SVG；图源来自computed_results.json。A图是8用户内先平均后的效果，B图用9300条主结果与实际/期望处理成本。

使用原已验证的Windows工作树和输入目录运行；只需CPU进行报告重算，不加载CUDA。完整源/数据/权重路径与SHA256见JSON的hash_inventory。旧实验数据及模型需仍然存在，不能仅凭报告目录重新训练。

```powershell
& 'D:\D_conda\envs\pytorch\python.exe' 'C:\Users\19252\.codex\worktrees\uwb-user-calibration\uwb_gestures\experiments\new_routes\report.py' --root 'D:\uwb_gestures\experiments\p1_20260928' --out 'D:\uwb_gestures\experiments\p1_20260928\report_s2026_release_v2_reproduced'
```

快速只读A/B/策略审计（不会生成正式报告，也不会代替完整延迟验证）：

```powershell
& 'D:\D_conda\envs\pytorch\python.exe' 'C:\Users\19252\.codex\worktrees\uwb-user-calibration\uwb_gestures\experiments\new_routes\report.py' --root 'D:\uwb_gestures\experiments\p1_20260928' --audit-core
```

单元测试：

```powershell
& 'D:\D_conda\envs\pytorch\python.exe' -m unittest discover -s 'C:\Users\19252\.codex\worktrees\uwb-user-calibration\uwb_gestures\experiments\new_routes' -p test_report.py -v
```

原延迟命令（需要受控测量环境，不能与报告CPU审计并行计时）：

```powershell
& 'D:\D_conda\envs\pytorch\python.exe' 'C:\Users\19252\.codex\worktrees\uwb-user-calibration\uwb_gestures\experiments\new_routes\benchmark_latency.py' '--training-finished' '--allow-idle-gpu-pid' '52016' '--policy' 'required'
```

正式报告要求8折基础/校准/多雷达/预算结果完整；机会门槛要求条件策略时须有完整8折策略。还需真实CUDA测量的279预选片段、所有方法、对应FROZEN_SHA256.json。缺失、smoke、ID错位、摘要不一致或冻结哈希变化立即失败，不会发布REPORT.md。输出目录必须新建或为空；输入目录不能作为输出。

source_snapshot仅归档本次报告与数值审计实现及其测试，帮助审阅；复现命令仍指向原工作树，原因是实验身份绑定了原源码绝对路径和相邻计划文件。不会把移位后的归档脚本冒充同身份重跑。

ARTIFACT_SHA256.json列出本报告交付文件摘要。图像需做渲染审阅；自动生成不替代研究解释或独立实验。
