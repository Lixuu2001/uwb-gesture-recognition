# 独立公开跨传感器雷达手势数据复核（2026-10-09）

## 判断

本次仅查公开的一手论文/数据记录，不读任何项目标签、雷达数组、样本级元数据，不下载数据。发现独立mmWave/FMCW手势库，但没有能直接作为IR-UWB同任务外部集的数据。跨传感器的粗粒度共同标签评估可以作为后续设计线索；它不是方法结果，也没有通过新颖性门槛。

## 数据候选

- **Pantomime**：Aalto论文记录为约22,000条、41位参与者、21种手势；评估环境、速度、角度、距离。公开Zenodo包为CC-BY 4.0，约107.6 MB；表示是稀疏3D mmWave点云，不是X4 impulse CIR。具体类别和项目手势的逐类映射本轮未核实。[论文页](https://research.aalto.fi/en/publications/pantomime-mid-air-gesture-recognition-with-sparse-millimeter-wave/)；[数据页](https://zenodo.org/records/4459969)；[Aalto数据记录](https://research.aalto.fi/en/datasets/pantomime-dataset-and-source/)
- **MCD-Gesture / DI-Gesture**：作者README记录24,050样本、25人、750种环境×用户×位置组合；六个正类为push/pull/左右滑/CW/CCW，另有七类负动作。传感器为AWR1843 77 GHz FMCW，处理后为T×32×32 DRAI。raw ADC需签署访问协议。其DI-Gesture论文已明确研究跨用户/环境/位置的domain-independent识别，故通用跨域主张已被先例覆盖。[作者README](https://github.com/leeyadong/cross_domain_gesture_dataset)；[DI-Gesture DOI](https://doi.org/10.1109/TMC.2022.3207570)
- **KIT FC-FMCW**：KIT记录称2,479条、25位参与者、BGT60TR13C、七种手势+no-gesture；两部分是924条/22人/七手势与1,555条/4人/五手势+零类。分组数字未证明参与者互斥，须先核文件。CC-BY 4.0，列表包约14.7 GB。[KIT记录](https://publikationen.bibliothek.kit.edu/1000172790)
- **60GHz帧标注版本**：Zenodo报告五种正手势、12人、每序列100帧并逐帧标注；此版本对早先IEEE DataPort记录增加标签，raw radar data unchanged。压缩包列表36.5 GB。它不是一个新raw cohort，固定手势序列也不等于含自然背景的连续流，不能为在线起始误触发提供真值。[Zenodo记录](https://zenodo.org/records/15178095)

## 先例与边界

DI-Gesture已覆盖mmWave跨用户/场景/位置的通用域泛化。另有UWB↔FMCW雷达迁移用于digit识别（Jung等，2023，DOI 10.1109/JSEN.2023.3289232）。因此，不能把“雷达跨域”或“跨雷达迁移”笼统宣称为原创。UWB-Gestures的12类与外部数据标签并不相同；若建共同粗类任务，必须先仅基于数据字典冻结映射、用户/域划分和评价端点，且不能把分数称作原12类成绩。现有外部集也没有UWB连续背景、同步起始或独立手运动真值。

## 后续门槛

若继续外部数据路线，先确定完全相同且可溯源的粗粒度端点，审查直接先例，核清许可和获取途径，再冻结参与者/域隔离协议。MCD需要申请，其他数据也要核实下载/文件结构；目前不请求受限文件、不下载大包。只有以上通过才做成本估计和小型先导。本轮发现的是一个可研究设计的方向，并未验证贡献。

## 执行记录

云环境connected/running，CPU-only，无采集能力、secret或出站身份；原数据和789文件快照存在，无训练进程。未下载外部文件、未读项目标签/雷达数组、未拟合/训练，原源码子树不变。整体原创目标仍未验证。
