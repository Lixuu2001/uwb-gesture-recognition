# Published GroupDRO native full implementation: running

## 2026-11-08 GroupDRO 完整六折真实云端训练已启动，第一折原生及独立首步审计通过

完整源码预登记222fc556f54d0b90c11ddd478bee58d2579658f9，初始私有3块/136载荷精确索引ec4721b7328b1d256273f7cff9180dfca7f47182，六个不同训练域元组q/初始原生全状态/角色及原始输入SHA绑定REAL_RELEASE后，才检查全部旧PID/boot/startticks和新锁启动一个新phase。实际PID99353/startticks33123032/bootad409b48-89ce-4704-beee-3658103d5856存活，held1训练中，尚无完成折或新准确率。

真实首步相同初始模型/RNG/顺序/原生BN/dropout/完整AdamWLookahead/q状态精确重放一致。独立NumPy float64 BCE/q/logits导数→全617612原生参数梯度参考最大差7.450580596923828e-9，q差2.9802322387695312e-8，通过预登记容差；reference objective .6910376786939741，实际q加权source-BCE .691037654876709，原生globalBCE .6903143525123596。first auditor退出0，FIRST_RELEASE放行。精确重放/局部参考不意味着排除全部共享实现错误。只使用训练标签；未读取held真值或用户7-8标签。

固定eta.01/完整GAP617612/三雷达signedRMS/B64/64轮/seed20261029/原生优化器和globalBCE验证选模不变；新增真实六折生产64+每折独立fresh64及审计累计墙钟/CPU上限43200秒/RSS12GiB。生成reserve11.6848h为估计，不是实际完成用时；旧MMD/VREx/Fishr/IRM成本和负结论另计原样保留。日志/状态/精确命令与恢复链在本报告。后续先查进程和检查点，禁止重复启动；执行失败保存现场不当科学失败，不能自动重置12h预算或换seed/eta/批次。全部6折模型、预测、原生审计和私有最终原始全部回读冻结后才一次全队列评分。主六折过固定门槛后才实施已登记eta0固定源均值匹配对照归因，不能把已发表GroupDRO叫原创。现有自动接续频率/启用状态/环境保持。以下为历史。

