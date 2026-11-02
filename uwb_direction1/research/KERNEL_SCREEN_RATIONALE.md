# 三雷达非线性交互验证筛查（2026-09-30）

## 定位与预注册

此次不重跑共享/专属分解、用户中心惩罚或收益选路。假设是：固定三路编码器包含的信息，可能需要同时匹配三路响应的非线性读出；逐路核相加作为同384维近似容量对照。联合核为标准RBF乘积形式，Nyström、核岭回归以及类/用户均衡都不是新算法。旧LTR本身有非线性融合头，所以不能把旧头称为线性模型，也不能把筛查成功自动归因于“首次加入非线性”。

协议先于真实结果提交：`bc9a152`。只用源六人拟合标准化、核尺度、地标和读出，只用验证用户选正则强度。八折验证门槛同时要求联合核相对加法核及旧LTR各至少+0.5pp，并至少6折优于加法核。失败则不生成新测试预测、不追加网格。产品核的近似秩相同不意味着相同函数空间；只控制地标数及计算阶数。此门槛只决定下一步投入，不能证明统计显著性。

## 已有负结果必须保留

- `original/project/experiments/p1_20260905/missing_view_fusion/COMPLETION.md`：共享/专属、缺失补全及原型字典均未过门槛。
- `original/project/experiments/p1_20260905/temporal_style_diagnostic/REPORT.md`：三视角等权DTW未显示真实跨人匹配稳定性优势；匹配稳定性没有阶段真值，不能当作语义准确率。
- `original/project/experiments/p1_20260905/mild_time_augmentation/COMPLETION.md`：16次训练、单种子温和增强+0.9222pp，用户bootstrap跨零，未过既定门槛。不是当前三雷达收益。
- `results/cloud_20260930/RUN_REPORT.md`：冻结特征类内跨用户约束92.7231%，停止该设置。

## 本次实际访问的文献

1. [Duvenaud, Kernel Cookbook](https://www.cs.toronto.edu/~duvenaud/cookbook/)，已读取组合核正文：相加对应可加函数，乘积要求多分量同时相似。该先例已足以排除把联合核本身称为新颖。
2. [Williams & Seeger, NIPS 2000](https://proceedings.neurips.cc/paper/2000/hash/19de10adbaa1b2ee13f77f679fa1483a-Abstract.html)：论文页及PDF可访问，但PDF文本提取乱码。本次不声称完整论文精读；使用标准Nyström公式并以正定性/XOR数值测试核查实现。
3. [Xue等, MV-GLFNet, 2026](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1864448/full)：读取正文3.4、4.1及4.3，确认三视角早融合、局部/全局分支、门控与双线性融合已有先例。其报告9600条、六人训练/两人测试；项目是9593原条目、9300质量保留、六人训练/一人验证/一人测试八折，98.29%不能与93.4417%直接比较。正文还有逐epoch测试曲线，不能据此断言其最终选择泄漏，但也不能假定它具有本项目的独立验证用户选择规则。

这些核查不是系统文献综述，没有支持“此前无人研究”的结论。
