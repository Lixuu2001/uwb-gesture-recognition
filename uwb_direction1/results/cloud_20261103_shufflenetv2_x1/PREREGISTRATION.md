# 完整官方ShuffleNetV2 x1：成本通过；真实40轮实验独立预登记

截至2026-11-03T02:02:57.201187+00:00，先前全层ResNet18的96.9167%一次性诊断失败已关闭并在95b4f9436a35fae5db215ae3f370a96cfe3ca3e8持久化完整私有模型索引，不能重启/调参救回。下一项是在旧诊断解封之前已固定的单一完整canonical ShuffleNetV2 x1.0预训练迁移方法，不是宽度/seed/学习率网格。原官方[4,8,4]块/[24,116,232,464,1024]通道、原1000头2278604参数严格加载，换12分类头后1265904参数全训练；原架构未缩减。Ma等2018 arXiv1807.11164支持原架构，Zhang/Lan/Zhang2021 DOI10.3390/s21010259原始可索引内容支持雷达手势预训练ShuffleNetV2迁移。雷达论文报告1.37M与本x1原2.278604M不同，原文完整训练段读取受阻；不称作者雷达规模/超参严格复现。RMS三雷达range-time到224图像适配及40epochs/B15/SGDM3e-4/.9/decay1e-4是明确项目迁移协议，普通已发表方法迁移不称原创。

正式cost在95b4f9436a35fae5db215ae3f370a96cfe3ca3e8全部40变更精确回读/head确认后才启动，worker97560完成exit0/noOOM：9.219390708秒墙钟/14.876465CPU，ru_maxRSS817068KiB。六次完整B15更新最高.311976550秒，最终B5 .095615548秒；三次B15评估最高.058646520秒；完整模型/BN/优化器/动量/RNG checkpoint10362327字节，三次磁盘save/load精确，临时合成文件按登记删除并保留SHA。固定公式覆盖producer+独立重放各40轮/12560更新、各40次验证及selected2200预测，保守8348.676818124秒=2.319076894小时/33394.707272498CPU，RSS含2GiB映射余量2914220KiB；完整模型与流式归档额外空间166728044字节，实测剩余329469952。通过12h/172800CPU/14GiB预算，独立verify_result仅核算原测量/TERM/IO证据，不重测。尚未实际UWB拟合，不是新的准确率结果。

完整原1000logits独立channel索引重排driver以及B15 checkpoint恢复连续2步精确检查已通过。新的真实实现另做全参数两步独立SGD动量/decay公式精确检查、301056像素独立FP64仿射几何/zero-fill解析界检查；流式跨多块归档全部成员及乱序/缺块/损坏/路径穿越阻断检查、缺远端预登记/部分重放/资源停止不能解封标签的边界检查全部通过。完整mock上传→metadata→rawtrue/base64false→物化SHA→相同传输块原子替换通过，真实Drive调用0。这些是新adapter/context必要验证，没有重跑已关闭旧网络训练或测试。

真实实验22文件冻结，FROZEN_HASHES SHA=1616f2fe7c642d721d9e2af984f203aa3f2ddd7889659eb0dbb2dc75ed61129b，全部科学源码、权重、数据、运行库、成本依赖SHA通过。新backup明确保留原官方ShuffleNet源码/9.2M权重、seeded初始12头、best完整模型、final模型/优化器/动量/order/aug/TorchRNG、实际ID及全step日志/40验证及selected2200输出，并包含复用的预处理/原生SGD辅助源码和SOURCE_LAYOUT映射。原数据/严格依赖分支历史/哈希运行库另行恢复，不宣称全操作系统或所有历史运行时恢复。FP64基础像素参考函数已原样物化入新冻结源，不在审计执行时从旧engine读取；公式AST来源/hash留痕，无数学变化。

真实训练代码和本次变更全部精确远端回读/head确认后，才允许controller.py Popen一次。固定训练用户3–6/4700、验证用户2/1000最高present-class macro/最早并列选择、完整40epochs/B15/12560更新，无早停/超参grid；CPU4/f32/seed20261029，CE无类权重/标签平滑；共享三通道仿射增强只用于训练。所有层及BN学习阶段train，验证/诊断阶段eval。每epoch原子保存final_full完整优化器状态，改进时保存best完整模型；全程实际ID、每step顺序/增强RNGhash/loss、全40轮状态hash和验证预测保留。受控总12h/172800CPU、每阶段6h/86400CPU、14GiB RSS，文件锁/存在断言和TERM防止重复启动/超限放行，无自动resume/retry。

只有producer正常完成才自动新进程完整40轮重放，逐step/全40验证/selection/所有模型/优化器/RNG及2200 selected输出精确相同，再完整6900 rawRMS/FP64和1038643200基础像素解析float32界。仍用同源Torch内核，非第二套卷积实现。之后必须完整owner-only私有分块和所有成员实际恢复验证，再finalize_once排他锁一次解封诊断用户1。用户7–8标签禁止，既有八用户开发探索性；固定参考.9866666666666667/+ .005/tol1e-12不变。失败关闭确切x1/40/B15/seed配置，不能换宽度/seed/轮数救回；通过也仅另行登记source6确认，不立即扩八折或声称原创。

同时透明记录旧ResNet18 PROTOCOL.preprocessing的继承文案错误：其中末尾残留frozenBNeval；当时固定all-layer配置、原生make/train代码、独立BN更新检查和完整40轮重放实际均是BN train学习、eval验证诊断。旧冻结PROTOCOL/代码/模型/结果一字不改，不重拟合/重评分/改门槛；DOCUMENTATION_ERRATA.json记录原SHA和正确解释。新未冻结阶段已清理这一继承文案，并使新runtime_parity/architecture/weights/preprocessing字段一致。此为元数据勘误，不是数值审计或模型算法修补。

接续目录 `/workspace/uwb-shufflenetv2-x1-pilot-20261103`。先读STATE/LAUNCH/supervisor01或audit_supervisor01的PID/TERM和/procstart_ticks，不能重复启动或下载。complete()全部守卫通过后，backup.py build一次；操作键ShuffleNetX1CompleteDir/Manifest为complete_backup路径/MANIFEST，Pipeline为已冻结upload_stream_pipeline.js，先核验父文件夹owner-only元数据，最多3条完整流水线，原始下载立即SHA/原子替换。出现既有响应或ID必须先复用/核查，不能盲目重传；按序汇总READBACK_INPUT，再backup.py verify全流式整体/成员/receipt/index，然后finalize_once一次。当前提交是实际启动前预登记，不能声称后台真实训练已经启动。准确率提升和原创贡献仍未验证，目标未达、自动接续继续；789源码、用户README、仓库可见性及Drive私有性保持。
