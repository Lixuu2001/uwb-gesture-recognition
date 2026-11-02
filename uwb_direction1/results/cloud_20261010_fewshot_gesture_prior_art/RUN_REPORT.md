# 雷达手势少样本个性化与运动特征先例筛查（2026-10-10）

## 判决

本轮补查少样本用户适配与相位/多普勒特征的直接先例。新找到一篇 2026 年 IEEE Internet of Things Journal 论文，摘要明确提出以雷达手势进行未见用户的少样本个性化，并使用用户条件双原型一致性、VAE 潜空间对齐、蒸馏和 L2-SP。它使“通用 few-shot user adaptation / prototype consistency”不能作为本项目单独的新颖点。该文使用 mmWave；本轮未拿到全文，不能断言其协议等同于本项目 X4 IR-UWB 八折划分。

经典多普勒、时间频率特征及 IR-UWB 手势分类也已有直接先例，因此不把笼统的 micro-Doppler、时频图或速度特征作为候选创新。本轮仅做公开文献核查；没有读取本项目标签、数组值或检查点，也未进行拟合或训练。

## 文献证据与边界

- Tian et al., “Latent Space Consistency Learning for Few-Shot Individual Gesture Recognition With Millimeter-Wave Radar,” IEEE Internet of Things Journal, 13(13), 29776–29791 (2026), DOI [10.1109/JIOT.2026.3687630](https://doi.org/10.1109/JIOT.2026.3687630)。可检索摘要报告：源用户手势类别内的双原型一致性；面向新用户的少量标注、source-free 轻量微调；VAE 潜空间正则及 z-space 对齐，并配合 KD 和 L2-SP。书目信息由 [J-GLOBAL 记录](https://jglobal.jst.go.jp/en/detail?JGLOBAL_ID=202602273435461361)交叉核对；本轮未取得论文全文，细节以摘要级证据为限。
- “Dynamic Hand Gesture Classification Based on Radar Micro-Doppler Signatures,” IET Radar, Sonar & Navigation (2017), 可从 [UCL Discovery](https://discovery.ucl.ac.uk/id/eprint/10046950/)查到；“Micro-Doppler Gesture Recognition using Doppler, Time and Range Based Features” (2019) 亦有 [UCL Discovery 记录](https://discovery.ucl.ac.uk/id/eprint/10073829/)。
- IR-UWB 手势的方向与手部正面面积特征已有早期工作：[Hand-Based Gesture Recognition for Vehicular Applications Using IR-UWB Radar](https://pmc.ncbi.nlm.nih.gov/articles/PMC5422194/)；同一雷达家族的深度分类也已公开：[IR-UWB Inception classifier](https://doi.org/10.3390/s20020564)。
- 当前同数据集的多视角融合、视角缺失、速度尺度、开集识别、EM反演、跨雷达迁移等否决项，见先前的分支报告与 STATUS。上面这些来源进一步缩小了可声称的新颖范围；这不是完整的系统综述，也不构成“文献中不存在”的证明。

## 决策与执行

关闭“通用少样本个性化/原型一致性”“笼统微多普勒或时频特征”作为独立新方法的路线。具体地将上述论文方法迁到 X4 数据仍可作复现/协议扩展，但不能据此宣称新算法，且须先拿到全文、定义其与八个 held-user folds 可比的评估，并冻结目标标注预算。

本轮没有符合门槛且值得训练的替代候选，所以不启动训练；不重跑历史失败网格、不触碰 789 文件原始代码快照。云端硬件/算力及校验状态沿用最新 EM gate：数据恢复标记在位、CPU-only、无采集硬件、未发现活动训练。
