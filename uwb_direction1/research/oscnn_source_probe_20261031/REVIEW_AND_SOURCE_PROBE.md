# OS-CNN 作者源码的合成可运行性核查，尚未选择真实实验配置

ResNet CPU100真实训练已完成，既有完整重放auditor仍在运行。本目录只为之后的文献方法循环准备具体代码；不读取新的UWB值/标签，不开展新真实fit，不提前按未经审计的ResNet分数选择下一配置，成本门槛也尚未运行。

Tang等，*Omni-Scale CNNs: a simple and effective kernel size configuration for time series classification*, ICLR2022，[arXiv:2002.10061v3](https://arxiv.org/pdf/2002.10061)。本轮读摘要、方法3.1–3.4、4.3实验设置的文字/公式/图注，非完整17页或图像核验。论文以多种核长的三层OS-block覆盖感受野，另介绍残差、ensemble和逐变量扩展；UCR实验设定不能直接冒称多变量包装器的配置。本次只是已发表方法的准确率候选准备，不声称原创贡献。OpenReview返回浏览器验证页，未绕过，使用可直接获取的作者arXiv版本。

作者库Wensi-Tang/OS-CNN commit f70626d3c4aad70a76ba7b4745f556df01044279，7文件的Git blob SHA1与原UTF8字节SHA256均验证，MIT许可保留。789原源码和最新远端STATUS/continuation exact OS-CNN/Omni-scale/repo/arxiv标记有界搜索0；不构成穷尽等价或新颖性证明。不是把旧通用多核CNN称为新方法。

直接导入未修改的OS_CNN及structure builder。以作者multivariate notebook的输入通道相关规模公式8*128*C和5*128*256+2*256*128核查C567、L90、12类形状；不是错用默认单变量1024预算而令第一层输出0后自行补丁缩宽。当前wrapper的上限是min(L//4,89)=22，实际核[1,2,3,5,7,11,13,17,19]，三层输出117/225/234；含固定mask总3736380参数，可训练1870464、14组有梯度Adam state。源码few_shot=False才接分类head，完整head保留。

按源码核组合精确枚举，卷积路径的名义局部RF集合为1–38（不计BN统计依赖），GAP仍汇总完整90帧。因此没有覆盖局部RF1–90这一整套尺度，不能据此误说输入被截断或网络不使用其他帧；这是wrapper的已核行为，未悄悄放大kernel。代码在每次forward执行weight.data=weight*mask，而未对梯度做mask；Adam更新后被屏蔽位置可重新非零，下一次forward再清零（eval也会修改这些权重）。合成测试已核实这一点，不能将其静默替换成梯度掩码或仅初始化时清零。后续任何真实审计必须记录并重放eval的权重副作用。

源wrapper Adam.001/默认betas/epsilon/无decay，ReduceLROnPlateau(min,.5,50,min_lr.0001)输入是**该轮最后一批loss**，不是epoch平均。名为Best_model的文件是每50轮和最后轮覆写，并未验证最优选择；最终模型为last。普通forward输出logits。constructor默认为2000轮；multivariate示例用500并注明作者实验2000。三者不应混称，若未来CPU预算适配必须在成本测量前固定规则并声明偏离，不能据诊断扫轮数。当前未选定UWB轮数、seed、batch或准确率门槛协议，尚未跑成本或注册真实入口。

具体代码author_source_probe.py在纯合成B2/seed7上，通过原fit的未修改innerloop AST三步与独立adapter三步全模型/Adam/全部Python/NumPy/Torch RNG逐位对照、恢复后两步逐位；mask/even-SAME并行支路卷积由独立NumPy实现核对。保留完整源码，没有import stub/缩宽/替换model。该B2/seed7仅用于工程检查，不能描述为已选真实训练配置。SOURCE_PROBE.json记录实际0.742039秒脚本时间，**不是正式成本门槛或真实训练时长估计**。

下一首先让现有ResNet审计/私有完整备份/注册gate收尾；本候选若继续，须先选择并冻结一项来源明确的包装器版本（不得混合残差/ensemble/逐变量模块后声称创新），在正式合成计时之前写资源规则，再完成真实runner/auditor/ID/模型日志和远端预登记逐字读回后才fit。7–8标签不用于新方法选择，历史八用户反复开发仍探索性。不能根据未审计ResNet producer分数抢跑配置。原789快照不改，Git不放UWB原数组/maps/模型/实际样本ID。
