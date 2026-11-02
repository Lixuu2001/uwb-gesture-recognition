# 相位表示与轻量雷达网络：补充近邻核查

2026-09-30，当前三臂八折训练继续运行，本次核查不改变模型、输入、协议或继续门槛。

## 新查到的具体先例

### 复值FMCW手势网络

[Complex-Valued FMCW Radar Hand Gesture Recognition仓库](https://github.com/thetuantrinh/Complex-Valued-FMCW-Radar-Hand-Gesture-Recognition)自述对应Trinh等人的 *Complex-Valued (2+1)D Convolutional Neural Networks for Real-Time Hand Gesture Recognition on Edge Devices With FMCW Radar*，并给出IEEE TAES 2026及DOI 10.1109/TAES.2026.3709269。这些出版信息在本次只从仓库取得，未向出版社独立核验；没有把它当作已读全文。

实际读取README、data/README.md和src/cvradar/models/layers.py，获得下列可核对信息：

- README与数据说明使用77GHz AWR1243BOOST的原始I/Q，单条20帧×128chirp×64采样点×8实通道；10类、10人。8名训练、另2名验证，并在同验证用户上加入合成噪声。
- 层实现真实调用ComplexConv3D，先空间核(1,Kr,Kd)再时间核(Kt,1,1)，配复数归一化和残差。`strides`在两个因子均应用，不能把它等同于一次空间下采样。
- 这足以说明“保留相位”“实虚表示”“轻量时空分解用于雷达手势”都有明确近邻。仓库宣传的准确率/延迟未在本环境复现，不与本项目93.4417%直接排名。

与本次差别是：本项目是实值UWB去杂波RF，经Hilbert解析构造后取相邻帧乘积，使用普通实值2D卷积；不是该工作的硬件I/Q、复卷积，也不是其10类8/2协议。本项目12类、8人六训练/一验证/一测试八折；仅数据、模态或网络不同不足以构成新贡献。

### 相同UWB-Gestures上的通用神经网络

[DenissStepanjuk/UWB-Gestures-classification-with-Neural-Networks](https://github.com/DenissStepanjuk/UWB-Gestures-classification-with-Neural-Networks)的README明确使用UWB-Gestures，并列出CNN、RNN和Transformer。这里只读取README，没有推定它采用本项目的质量掩码、被试划分或评价指标，也没有宣称已复现。

因此，把UWB数据改成小CNN即使提升精度也不能单独宣称论文创新。当前legacy臂的意义是同架构强对照和工程基础，候选圆周表示需要相对angle_control的冻结比较。

## 对当前研究结论的约束

当前具体可检验的问题仍是：幅度权重一致、相同网络预算下，圆周坐标在范围压缩中是否比线性角度表示稳定、并改善跨用户识别。已有源几何诊断只显示分支附近确有样本；它既不是“首次发现”，也不是模型收益证据。

若候选通过预定八折门槛，还需独立训练种子、无范围池化消融和近邻全文核查，才能评估具体贡献。若未通过，不能转而只比较弱legacy臂、引用异协议数字或把通用复值网络重新包装为本项目新方法。

## 检索和证据边界

GitHub仓库名/描述搜索使用“UWB gesture phase”“radar gesture complex”“phase representation gesture radar”“UWB-Gestures”。检索覆盖有限，零结果不等于不存在先例；不据此声称穷尽文献或证明新颖性。

所读文本和GitHub blob SHA见sources/RADAR_PHASE_REVIEW_INDEX.json。未下载或运行外部模型；未联系论文作者；未更改正在执行的训练方案。当前仍缺论文全文和出版社元数据独立核验。
