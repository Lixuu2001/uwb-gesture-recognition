# 固定区间平均的先验来源与本次实验解释

2026-10-03。源用户3–8先导完整结果：ERM91.9028%、固定尾段平均92.7222%、SWAD92.5846%。SWAD比固定平均低0.1376pp，只2/6改善，未过事先冻结的完整门槛，停止该配置的八折/扩种子/网格。固定平均比ERM高0.8194pp、4/6改善，描述性用户bootstrap95%区间[-0.0556,+1.6944]pp，不能当成已验证稳定增益，更不能借对照的小增益把失败SWAD包装成成功。

本轮实际读取SWA作者仓库timgaripov/swa的README、train.py、utils.py。README对应Izmailov、Podoprikhin、Garipov、Vetrov、Wilson的Averaging Weights Leads to Wider Optima and Better Generalization（2018，arXiv:1803.05407）。没有读完论文全文或独立确认出版版本；以下只对所读作者实现负责。

作者实现使用SGD+momentum，学习率分段下降后保持swa_lr，swa_start后按swa_c_epochs收集epoch末模型；moving_average以1/(swa_n+1)等权更新。包含训练集BatchNorm统计重估。README把constant/cyclical学习率探索与多点平均作为方法背景。本次固定对照用AdamW、24轮单次余弦下降、第13–24轮每个优化器步密集平均，模型为GroupNorm；它不是作者CIFAR协议或原SWA完整复现。不得把本次反复开发的雷达源用户成绩与README的CIFAR表直接比较。

含义：权重平均本身是明确既有方法，不能以“尾段平均/更平坦”命名为本项目创新；本次也没有测量损失平坦性或建立其因果解释。后续任何新的表征或架构机制应带固定平均控制，避免把通用训练平滑收益误归因给新模块。不要依据本次诊断标签另选起始epoch、收敛窗口或容忍比例。

下一项研究范围应回到雷达输入表示与计算效率：先核查距离轴压缩/时距分离卷积的雷达近邻论文和已有原源码，在源用户内部做明确的保真与成本假设，再冻结新的匹配对照；现有轻量CNN只是工程基线，不能因参数少就宣称原创。此处是后续研究问题，不是已确定新算法、已启动实验或已有新结果。若近邻已有同样机制，按既有基线处理。SWAD失败配置不重启。
