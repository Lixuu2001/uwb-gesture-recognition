# 下采样混叠与位移一致性：作者论文及本项目边界

2026-10-03。读取Richard Zhang，Making Convolutional Networks Shift-Invariant Again，ICML2019，PMLR97；不是CVPR论文。作者公开[项目仓库PDF](https://github.com/richzhang/antialiased-cnns/blob/gh-pages/resources/camera-ready.pdf)由允许的GitHub codeload下载，4,970,902字节，Git blob7de5c4f3342b3daf5f165fa0ab0b0adbdfe9a1bf，SHA256=93755c6bf93f68ee3bb77be60d18a0beddac643703760a48a40f7fd2656743b7。已阅读主文§1–5的正文、公式、表格及图注；未逐图视觉检查、未读完参考文献与补充材料，不把本范围扩大为领域文献查全。

另实读[adobe/antialiased-cnns README](https://github.com/adobe/antialiased-cnns/blob/master/README.md) blob0d9d17a02091b0437b54da83f105763d40c0447c，以及[BlurPool源码](https://github.com/adobe/antialiased-cnns/blob/master/antialiased_cnns/blurpool.py) blob062f38d0de1df383236b4a2fddd42d2835fb5608。代码使用归一化二项式系数、反射填充和按通道卷积；包含1D版本，因此不能把“只沿一轴滤波”当新概念。作者代码许可CC BY-NC-SA4.0；本轮未复制该实现到项目，仅引用核查。

关键论据与限制：
- §3.1将平移等变与输出不变分开，特别讨论边界；CIFAR用循环卷积及循环移位以隔离边界因素。本项目原网络零填充，不能把非循环移位的预测变化全部归因为混叠。
- §3.2先密集执行原算子，再在抽取前低通；仅对原输入低通不是等价的网络抗混叠。步长卷积的示例还有非线性位置要求。标准BlurPool是既有算法，不是论文创新点。
- §4.2分类一致性是两次输出argmax相同，不是两次预测正确；本文将一致性与准确率分别测量。我们无标签诊断只测一致性/Jensen–Shannon差异。
- ImageNet实验通常跳过最高分辨率的首次步长卷积以控制成本；保持参数数目不等于计算成本不变。§4.5中更强滤波对生成质量存在取舍。

现有tools/circular_signal.py中Block首层为3×3、stride2、padding1卷积，接GroupNorm/GELU，共三个Block，然后3×4自适应池化与位置敏感的展平头。该源文件SHA485347aac0f42eba0353fb6030919b7d451d5439b9df06838b4e1e3bebe44403。原789源码快照中读取uwb_osr/model.py（blob3b37977e98dea599b0a6840a63b83d88b312dd8c），为双路径Transformer/池化，不是本轻量CNN；没有逐文件排除全部789文件中的相同机制，不能声称全项目从未做过抗混叠。

预冻结本轮：复用源3–8的六个ERM模型（各四人训练、一人验证、一人诊断），共7100条，各雷达单独±1现有距离单元，零填充、不循环、不重新标准化。一个单元对应3个原距离bin，不是已校准米数。记录预测翻转、概率JS以及该雷达丢失的边界能量；先看丢失≤1%子集覆盖，每用户至少200对且至少四用户翻转率≥5%才允许计划新的控制实验。不计算准确率、不用诊断标签调位移或挑选权重。

即使通过，也只说明存在值得检查的敏感性；输入位置、边界、非线性、GroupNorm和展平头都可能贡献。独立雷达同量距离移位也不是已证明的3D可实现运动。尚未实现/训练BlurPool候选，不宣称因果诊断或新方法有效。
