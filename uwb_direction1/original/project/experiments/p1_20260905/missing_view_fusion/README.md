# 三视角训练、单雷达部署：三条路线的第一轮实现

此目录保存 2026-09-07 获准实施的顺序筛选。所有方法在训练阶段可以使用同步 Left/Top/Right，验证和测试只接收 Right。现有主程序、旧模型和旧结果未被改写。

## 实际实现

1. **缺失视角感知融合**：同一个融合头联合处理 Right、Left+Right、Top+Right、三视角四种输入；逐步比较随机缺失、同时训练、可学习缺失标记、融合特征的监督对比。
2. **共享／专属特征分解**：共享投影保留跨视角共性，各视角专属投影保留自身特征，缺失槽位用可用共享特征替代；比较单视角、联合训练、共享对齐和视角辅助监督。
3. **原型库补全**：只在训练样本中估计类别原型；Right 特征预测软权重，组合 Left/Top 原型后分类。部署不读取真实标签，也不读取其他雷达。与 Right 原型库及可训练字典比较。

这是上述机制的冻结特征适配与低成本筛选，**不是论文原始端到端网络的完整复现**。其中共享对齐采用配对均方误差、原型采用类别均值，均属于本项目适配。

## 数据与控制

- 复用 `sv_R_label_global_s2026_h*.pt`：只用 Right 幅度＋相位输入训练的编码器／分类器，全局软目标来源已核对。
- 编码器冻结，每折只从训练用户读取额外雷达。验证／测试缓存均为单视角矩阵。
- 输入 `[B,1,90,378]`，378 为 189 幅度特征与 189 相位运动特征拼接。两路 token 分段池化为 1664 维，不跨幅度／相位边界合并。
- 9300 个质量保留样本，8 折留一人测试，12 个输出。验证集来自其余用户内部 10%，沿用 split_seed=2026。模型初始化与采样随机种子分开。
- 每个阶段都有同架构 Right-only 对照；第一阶段随机缺失也执行 4 次前向，匹配同时训练的次数。分类损失按次数平均。
- **残差版本**叠加原始 logits；**直接版本**由新头直接分类。两类输出分目录，见 `READOUT_AMENDMENT.md`。
- 冻结原分类器是单独参照，不把“击败一个更弱的新头”自动解释为提升了项目当前准确率。
- `>=2` 个百分点且 `>=6/8` 用户改善为继续投入门槛；另检查是否超过复用的原始分类器。不是显著性检验。

## 文件与结果

| 文件／目录 | 内容 |
|---|---|
| `data.py`、`cache_v2/` | 训练三视角／验证测试单视角特征、原始输入及模型内容哈希 |
| `models.py` | 三条路线的小型分类模块 |
| `runner.py` | 配套对照、逐折训练、验证选模型、导出与可恢复队列 |
| `deploy.py` | 读取导出模型，仅接受预处理后的 Right 幅度＋相位张量 |
| `report.py` | 独立重算预测、划分、选择轮次、按用户配对差值 |
| `runs/` | 初始第一路线的残差版本正式试验 |
| `runs_direct/` | 依次执行的三条直接分类路线正式试验 |
| `smoke*/` | 两轮流程测试；不计入正式效果统计 |
| `tests/` | 数据隔离、梯度、部署、配对与结果完整性测试 |

每个正式运行保存 config、history、metrics、status、predictions 和 deployment。`source_code/` 保留训练时源码；代码或配置变化必须使用新输出目录。`SUMMARY.md` 与 `audit.json` 是审计工具生成的结果，只有标记完成的运行进入统计。

## 执行命令

在 `D:\uwb_gestures` 使用现有环境：

```powershell
D:\D_conda\envs\pytorch\python.exe -m pytest experiments/p1_20260905/missing_view_fusion/tests -q
D:\D_conda\envs\pytorch\python.exe experiments/p1_20260905/missing_view_fusion/runner.py --stages 1 --readout direct --out experiments/p1_20260905/missing_view_fusion/runs_direct
D:\D_conda\envs\pytorch\python.exe experiments/p1_20260905/missing_view_fusion/report.py --runs experiments/p1_20260905/missing_view_fusion/runs_direct
```

第一阶段审计后用 `--stages 2`，再用 `--stages 3`，保持其余参数。通过门槛的候选才追加第二模型种子及对应控制；第一阶段的真正配对作用用全训练池同用户同类别错配控制进一步检验。

## 解释边界

测试额外视角没有提供新测量值；可用信息来自训练过的权重／字典。不能声称恢复了真实隐藏观测或三维手部轨迹。Right 训练的固定编码器也可能限制 Left/Top 信息的表达，因此本筛选未见提升不等于端到端多视角学习无效。共享均方误差允许退化，专属分支视角分类也可能利用分支偏置，需在实际正向效果出现后进一步验证机制。

这 8 位用户已经用于项目开发，验证用户与训练用户不独立。用户 bootstrap 描述当前用户组的差异，不能替代新数据／新用户确认。

## 方法来源

- [MICCAI 2025 缺失视角感知融合](https://papers.miccai.org/miccai-2025/0483-Paper2038.html)
- [ShaSpec：共享／专属特征建模](https://arxiv.org/html/2307.14126v1)
- [PMCH：原型加权补全](https://www.ijcai.org/proceedings/2025/281)
