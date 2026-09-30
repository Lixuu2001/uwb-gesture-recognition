# 源数据距离中心诊断：不支持盲目去掉整段中心位置

2026-09-30。圆周表示机制未通过门槛后，对下一项几何不变性假设做源数据检查，没有训练新模型，也没有依据测试用户调整参数。

只读取fold1的六名源用户3–8，共7100条保留记录。对每条去杂波RF计算绝对幅度加权的距离格中心、范围扩散和逐帧中心变化。坐标为189个快时间/距离格索引，不是米或手部真实位置；多径和散射体混合仍存在。

在源样本上拟合描述性加性回归：中心位置分别由动作类别、用户ID、二者共同解释。

| 方差比例 | Left | Top | Right |
|---|---:|---:|---:|
| 只用类别解释的R² | 75.06% | 74.67% | 69.59% |
| 控制类别后用户额外解释的总方差 | 1.20% | 4.00% | 2.05% |

中心位置与动作类别高度相关，不能直接把整个中心当作纯用户因素移除。这里是样本内统计，未控制所有动作/用户交互，不证明因果风格，也不证明轻微平移增强无效。决定是**不直接启动强制逐片段完全居中这项训练**；后续若研究空间变化，必须先证明尽量保留动作信息。

算法与来源：tools/diagnose_range_origin.py；输出DIAGNOSTIC.json；完整源ID、中心等特征保存在云端/workspace/uwb-range-origin-diagnostic-20260930/source_centroids.npz，SHA已记录。所有拟合只用源标签，未读取验证或测试输入/性能。

文献/实现边界：读取了PyTorch官方Spatial Transformer Networks教程的定位网络、仿射网格、grid_sample与单位变换初始化。它展示已有可学习几何变换机制，并不证明雷达手势对某个变换保类，也不能把直接加入STN称为新方法。教程链接：https://github.com/pytorch/tutorials/blob/main/intermediate_source/spatial_transformer_tutorial.py；blob SHA99efe41b39bab0491ae7690bc966f79b684d2c39；快照research/sources/pytorch_stn_tutorial.py.txt。教程引用Jaderberg等的Spatial Transformer Networks（arXiv:1506.02025），本轮未取得论文全文。先前尝试的qassemoquab/stnbhwd仓库返回404，未作为已读来源。
