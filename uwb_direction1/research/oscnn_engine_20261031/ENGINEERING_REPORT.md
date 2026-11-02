# OS-CNN 通用训练/状态/重放组件：合成对照完成，未启动真实实验

本轮ResNet原完整audit仍由52684持续计算；不读producer score来挑新配置，不重复起训练/审计。这里仅补齐下一轮文献方法实验所需工程组件，无新的UWB值/labels/fit、资源门槛、真实超参选择或远端实验预登记。

source_engine.py直接使用前一目录固定commit f70626d3c4aad70a76ba7b4745f556df01044279的未修改作者OS_CNN/structurebuilder和multivariate形状规模公式，依赖源码及MIT许可位于research/oscnn_source_probe_20261031/vendor/oscnn，未另装全局包。build必须显式传seed/shape/class数，不默认替用户选真实seed/epoch/batch。当前没有可执行的UWB数据入口/完整真实auditor；不能以组件测试冒称已经注册或启动下一训练。

保留source特有语义：Adam.001，ReduceLROnPlateau喂**每轮最后一batchloss**；源的每50轮评估train+val并覆写名为Best的文件，最后保存last，不用val选best。通用epoch接受显式评估周期，合成fixture为1以逐轮覆盖eval，未来原包装器真实配置需固定50并在预登记说明。源掩码在每次forward清零、Adam后会regrow，eval也改变权重；snapshot包含模型/Adam/BN/scheduler/模式/全部Python、NumPy、Torch RNG、identity、sampler，恢复会校验层结构，且可强制expected_identity一致。训练observer接口可记录实际第三字段ID/顺序/loss，不加入随机操作。

验证不是只重复内循环：加载作者完整fit AST，原函数body保持不变，只在末尾额外return局部model/optimizer/scheduler供只读观察。用原生DataLoader、纯合成32训练/19验证、567×90、12类，seed19、构造batch参数16使原min(n/10,16)有效B3，2轮/22更新、每轮eval。与新engine两轮比较，模型/Adam/scheduler/BN/全RNG、原周期及最终raw checkpoint加载后的模型状态逐位一致，最后loss调度和eval权重副作用均保留。源fit的requires_grad tensor到float warning保留在日志；engine传last.detach()消除此兼容warning，其数值/状态一致已对照。此警告不是fit失败，也未改作者源体。

另做独立NumPy CE方程、epoch1完整状态恢复续epoch2（含evalmask副作用）逐位；154个恒loss调度步符合独立的>50坏步触发规则；IDobserver与无observer同参数/RNG，wrongidentity在任何恢复变更前拒绝。四项readiness守卫证明：现有audit缺失/未过时，不读diagnostic summary；无固定真实配置、无匹配远端marker均拒绝。守卫还要求前ResNet完整私有模型集已验证、原route确实close；若ResNet gate pass应另注册source确认而非换方法。

合成fixture的epoch/seed/batch/评估周期只为验证控制流，不是已选真实实验配置。脚本运行时间不是正式costgate。也不声称原作者2000轮/示例500轮结果复现、准确率提升或原创。卷积局部RF1–38与GAP完整90帧区别、maskedweight副作用详见上一源码核查。

下一优先处理现有ResNet完整audit/私有完整备份/注册gate。完成后若close，才固定一项来源明确的OS-CNN variant及计时前资源规则，测CPU成本并一次选择配置，再实现真实runner/完整auditor/私有首步备份/真实IDs日志，远端预登记逐字读回后才能新fit。不能用producer未审计分数抢跑，不能把通用组件当完整真实入口。7–8标签不用来选择，八用户历史开发仍探索性；原789源码/原数组/maps/真实模型/实际ID不入Git。
