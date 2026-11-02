# 空间期望坐标近邻核查（2026-10-02）

本轮训练队列保持冻结。本笔记补充贡献边界，不改变方法、超参数或继续门槛，也不根据中途测试成绩选择新设置。

实际读取Chelsea Finn账户下cbfinn/cs280report的report.tex（Git blob 9b2640c89e22d2ca80a0982d1aa2d846cfa659fd），重点完整读取Network Architecture和CNN Architecture Evaluation两节，另看标题与摘要。来源为作者账户草稿；TeX标题是Robust Object Pose Tracking for Robotics，PDF元数据却写End-to-End Training of Deep Visuomotor Policies，正文也讨论机器人控制。不能把该文件冒称为Deep Spatial Autoencoders for Visuomotor Learning正式论文，也不能由草稿替代出版信息核验。未把稿件全文复制进项目仓库。

架构节明确把卷积响应图通过空间softmax变成位置分布，再计算每通道二维坐标期望，与机器人状态拼接后送入全连接控制层。该方法保留位置信息，服务视觉运动控制；空间特征提取及与其他状态拼接早已有先例。因此“CNN + 位置期望 + 拼接”不能作为本项目原创点。

架构评估节对比了固定坐标期望、普通全连接和最大池化，但任务是具备机器人正运动学位置真值的视觉姿态估计，度量为厘米误差。该节同时承认替代网络参数更多，不能把其差异当成严格等容量机制证据，更不能与UWB手势的跨用户宏准确率比较。

与当前实验的区别：本项目使用实RF绝对幅度的非学习质心与展宽，在90帧上构成时间序列；它不是学习得到的视觉关键点，也没有3D手部位置真值。当前static/ordered两臂参数与初始化匹配，可检验显式时序分布矩是否提供静态矩以外的识别收益。legacy参数较少，只能作强基线参照。即便八折门槛通过，仍须最近邻雷达论文全文、独立种子和纯矩/相对矩消融，不凭模块组合宣布创新。

本轮还读取cbfinn/gps的README，其自述为guided policy search与LQG轨迹优化的重实现，不能据此推断它就是DSAE作者实现。cbfinn/cbfinn.github.io的master树读取404，没有取得该假定网站的论文。文献核查仍未完整，STATUS中的paper_fulltext_review_complete保持false。
