# 时序卷积尺度核查：准备记录（2026-10-03）

本轮匹配正则化/组DRO全部失败，不能继续搜索其衰减、eta、种子或选模窗口。下一项只做编码器时序尺度的文献/旧实现核查，尚未预注册或启动新分类实验，不能称为创新。

## 实际阅读

InceptionTime: Finding AlexNet for Time Series Classification，Ismail Fawaz等，Data Mining and Knowledge Discovery 2020，DOI10.1007/s10618-020-00710-y。已读作者仓库完整README、classifiers/inception.py、classifiers/nne.py、receptive.py；主文PDF与图片未读。可定位blob见TEMPORAL_SCALE_REVIEW_RECEIPT.json。README实验主体是UCR单变量数据，不能据它声称已在此雷达跨用户设定有效。

作者代码默认构造参数kernel_size=41，内部先减1，实际分支核为40/20/10，额外池化3+1x1分支；通道数>1时有32通道1x1瓶颈，6个模块、每3模块残差、BN/ReLU、最后时间全局平均。nne.py默认5次初始化概率平均；单个网络不能冒称完整InceptionTime集成。默认1500epoch、Adam，学习率和最佳模型监视训练loss；不能直接照搬到本研究固定验证用户选模协议。receptive.py是合成位置模式的大网格，不是UWB证据；其图中RF公式也不能直接套当前含stride/GN/池化的网络。未运行作者代码，没有复制其训练或网格。

## 当前模型与旧路线的边界

当前SignalNet有6个3x3卷积，stride依次2/1/2/1/2/1。若仅算卷积路径，时间跨度依次3/7/11/19/27/43、步距最后8。这**不是整个网络的严格感受野**：GroupNorm统计跨空间位置，后面3x4自适应池化与全连接头也联合多个时段。不得据43<90声称“模型看不到完整手势”或因果欠建模。

原多视角/Right模型是SharedDualPathEncoder双路径，保存模型有90时间token和378空间token；已读取原multiview_s2026/source/03_model.py，不能把当前小CNN限制推广成全部历史网络限制。旧temporal_style_diagnostic/REPORT.md明确：Right已学时间特征可恢复已知时间形变，比固定帧位置强；三路等权真实跨人匹配没有稳定性优势。后续温和时间增强、等权DTW和15阶段过程监督已有负结果，不能换名再做。

789文件原清单按名称检索inception/tcn/receptive/multiscale等未发现同名模型，但这只是文件名排查，不是源码等价性或原创性证明。原报告与适用架构不等同当前CPU模型。

## 可执行的下一步边界

若继续，仅考虑同容量、同源3–8角色、同种子/预算/验证选模的卷积时间跨度干预，须保留当前CNN和参数匹配控制；先审核旧vendor编码器/增强调用和新实现实际计算图，再固定唯一干预及CPU预算。不能依据既有用户错误挑选核长，也不能把Inception/扩张卷积/多尺度组合称新方法。长时间覆盖本身是已有技术，只有可复现的机制/对照结果才能决定下一步是否值得研究。

本记录没有新增准确率，没有启动新后台训练，没有改变已完成组DRO方案的主结果选择。下一候选仍处于准备阶段；完整同协议八折、创新贡献和独立验证仍未完成。
