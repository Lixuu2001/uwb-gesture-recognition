# 完整状态自动归档

训练协议和训练源码不变。`tools/archive_view_sharing.py`只在训练队列complete且完整18轨迹独立审计通过后执行后处理；不读中途分数改训练、不替换选模或门槛。

等待模式每45秒核对队列和原训练PID，不重复启动训练，最多等8小时；原训练失败或PID消失/僵尸立即以失败状态结束。完成后独立按现存类别重算每用户macro accuracy、核对既定尾均值门槛，生成描述性的六用户bootstrap区间（不是新的显著性证据），汇总真实训练epoch耗时。

新12轨迹每条归档30文件：result、split、selection、逐样本预测、两选定权重、完整model/optimizer/scheduler/RNG进度及24个段平均权重，总360文件。原始RF和prepared数组不上包。每包原文件payload最多28MiB、压缩后必须小于32MiB；包含逐成员SHA清单，写后重读验证，现存包只在内容匹配时复用。总包数由实际大小决定。六条旧共享对照仍引用已验证的旧七包和三个优化器补充包，不重复复制。

已用真实已完成的untied128/held3全部30文件验证分包、整包/内容SHA、不可变复用；额外检查源文件变化、损坏归档、危险成员路径会被拒绝。该轨迹产生两个本地包26,477,524与14,439,466字节，**不是整组完成，也尚未私有上传**。详见PACKAGING_CHECKS.json。部分独立前向审计另见PARTIAL_AUDIT_20261003_01.json，只覆盖当时完成的三个新轨迹（held3两臂、held4独立臂）。

自动归档目标：`/workspace/uwb-sharing-final-20261003`；状态：`/workspace/uwb-sharing-archive-state-20261003.json`；日志：`/workspace/uwb-sharing-archive-20261003.log`。现有两个真实测试包直接复用，不删除。进程是否启动以最新MONITOR/STATUS和实际PID为准。

后续接续：先查原训练和归档两个进程。归档完成后检查ANALYSIS.json、ARCHIVE_RECEIPT.json、PER_USER.csv，核对原SUMMARY/AUDIT。将所有包通过已授权的私有Drive connector上传、读取权限/字节数元数据、认证重新下载核整个包和每个成员SHA；最多3并发且不打印认证URL。归档器只写本地，不把uploaded=false误说成已完成远端保存。传输完成后小摘要/回执写Git，大模型只存私有Drive。

预注册门槛失败就保留负结果并停止此共享干预网格；通过也只是六源用户开发筛查，需要进一步八折、共享位置消融及文献贡献审查，不能直接称创新成立。
