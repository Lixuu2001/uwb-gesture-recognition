# Published GroupDRO native full implementation: preregistered

## 2026-11-08 GroupDRO 完整六折执行器及生成审计完成，真实训练尚未启动

接续13e134d8的已发表GroupDRO默认eta=.01组件，新增独立/workspace/uwb-groupdro-loso6-20261108完整6折loader/worker/q-native检查点/首步NumPy风险与q及全617612梯度参考/逐步64+fresh64重放/角色隔离/12h资源监督/全私有后一次评分。46文件冻结；六用户顺序/缺组尾批/原生Lookahead slow6与disk7→exact8/全RNG/BN/标签firewall通过。完整生产worker/controller/journal在生成18样本fixture实际64轮+fresh64，全部65状态及逐步日志一致；独立首步最大参数梯度差7.45e-9，q差0。停止路径3个独立session toy进程全部终止。生成整套墙钟67.0622秒，整套CPU未单独测量；不冒称真实UWB结果。

沿用已预登记且只测一次成本，ratio1.0635646915307622，reserve42065.182194165376秒(11.6848h)，新增完整6折墙钟/CPU硬上限43200秒/RSS12GiB，不缩模型/输入/B64/64轮/seed，不重新测时寻找过门槛。全6模型初始化相同但q训练用户元组不同，必须分别保存和SHA绑定六个完整初始状态。完整源码远端精确登记、实际角色/输入hash及owner-only raw全部成员初始备份后才可release/启动一个真实phase。

GAP93.6672%强基线复用，eta0固定源等权BCE同预算消融仅完整主6折过冻结门槛后另行执行。旧MMD/VREx/Fishr/IRM负结果/成本原样保留；不调用旧单折once、不按held1筛后续折、不用用户7-8选方法，不主张作者benchmark复现或原创。当前没有真实GroupDRO进程/新评分，现有自动任务/频率/环境不改。以下为历史。

