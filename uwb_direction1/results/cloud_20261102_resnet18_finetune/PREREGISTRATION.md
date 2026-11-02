# 完整ResNet18全层微调：成本通过，真实实验独立预登记

截至 2026-11-02T19:28:14.339728+00:00，前一冻结ImageNet ResNet18+C1配置已在e609b4122d7e5d6baaf23435e2fb5f7dd4f61d09公开真实失败：验证82.2%、诊断76.6667%对固定参考98.6667%，差−22pp，完整私有恢复后一次评分，无救回/重评分。这里预登记的是文献支持的全层SGDM微调，不是旧分类头网格。Cho/Park2022 DOI10.3233/THC-212827出版社原文§4.2–4.4已核查；40完整epochs是项目对原文40 training iterations的固定解释，momentum .9与Torch全参数weight decay是明确实现选择。并非作者MATLAB逐更新复现，迁移到三通道雷达图和12手势也不构成原创贡献证据。

e609b4122d7e5d6baaf23435e2fb5f7dd4f61d09全部41变更精确回读/head确认后，正式合成成本94045实际执行完成exit0/noOOM：16.257457494秒墙钟、38.453333CPU、采样RSS975272KiB/ru_maxRSS1021824KiB。六次完整11182668参数B15前反向/SGD更新为1.133638857/0.995669063/0.925076035/0.904958625/1.038941415/0.960293821秒，最终B5一次.354069980秒；三次B15预测最高.327088315秒；完整模型/优化器/动量/随机状态checkpoint89,571,787字节，三次真实磁盘save/load语义精确，临时合成副本按登记删除并保留字节/SHA。不是实际UWB fit、不是缩小网络试验，也没有40轮训练结果。

冻结公式覆盖producer+独立audit两次完整40×314=12560更新、两遍40×67=2680验证批次、2200 selected held预测及checkpoint/180秒余量，保守估计30506.880974411秒=8.474133604小时/122027.523897644CPU；RSS含2GiB映射余量3118976KiB；额外磁盘443532966字节，测时剩余494768128字节。12h/172800CPU/14GiB预算通过，独立verify_result不重测，仅核算原测量/退出/IO证据。此时真实UWB微调尚未启动。

已完成原成本要求的流式归档实现：直接tar/gzip到7MiB块，不复制payload、不生成单体tar/重组副本。每块完整流水线先metadata owner-only/size/parent，rawtrue/base64false立即物化SHA；确认下载与原块字节完全相同后，以远端原始字节原子替换对应本地传输块，最多3个7MiB额外下载并发。最终有序流式读取所有确认过的远端块，校验整体压缩SHA/字节、每个成员SHA/字节/唯一安全路径，不解包模型大文件。私有scope包含共享ImageNet初始权重、seeded初始12头、best完整模型、final完整模型/优化器/动量/两类generator/TorchRNG/epoch更新状态、实际ID/全部step日志/40个验证预测和selected2200预测、完整源码/成本/audit。原数据/运行库另行恢复，不宣称全操作系统或全历史运行时恢复。保留最终checkpoint可手工恢复，自动resume/retry关闭。所有上传响应先净化落盘；发现既有响应或ID时自动上传阻断，后续必须复用/核查，不能因空搜索就重传。

合成流式测试3项覆盖跨多块原子下载替换+全部成员、缺块/乱序/损坏阻断、路径穿越阻断；完整mock上传/metadata/raw/物化/SHA/原子替换流水线通过，真实Drive调用0。首次异常路径有filehandle warning，保留attempt01源码/日志后修复；首次JS解析重复check变量也保留attempt01并在任何真实Drive调用前修复。完整B15原生SGD两次更新对全部参数按独立动量/.9/decay1e-4公式精确一致；301056像素独立FP64仿射逆几何含zero-fill，最大误差4.284787237884835e-5，在按尺寸/float32解析γ32界.0055858173644805775内，非经验阈值拟合。2项completion边界检查确保缺远端预登记、完整40轮replay缺证或资源停止不能解封诊断标签。原next2update模型/优化器/RNG精确恢复检查同样保留。

真实实验25文件冻结，FROZEN_HASHES SHA=69b12b2e244cd95d6c249237459d61dad91ef023d1ed19bc6c23f529cabc2886，依赖源码/权重/数据/运行库/成本SHA逐项核验通过。所有新变更精确远端回读及head确认后，只启动一次controller.py。固定原官方全18层ImageNetV1网络，12头/11182668参数全训练；CPU4/f32/seed20261029，40完整epochs/B15/SGD lr3e-4/momentum.9/decay1e-4，CE无类权重/标签平滑；共享三雷达训练仿射缩放.8–1.2和平移±32，验证/诊断无增强。train3–6/4700、val2/1000选择最高presentclassmacro/最早并列checkpoint，完整40轮/12560更新，无早停/参数网格。best_model.pt原子保存所选完整模型；final_full.pt每轮原子更新完整模型+优化器+动量+order/aug/TorchRNG，完成时必须epoch40。日志保存每step loss/顺序hash/RNGhash、每轮完整状态hash及验证预测，split保存实际IDs。

controller有文件锁/目录存在断言，producer后只有exit0/noOOM/无资源停止才运行新的独立进程完整40轮重放。总43200墙钟/172800CPU、单阶段最多21600墙钟/86400CPU/14GiB；实际remaining budget限制audit，TERM/RSS/cgroup证据留存。审计逐step损失/顺序/增强RNG、全40验证score/selection、最佳及最终完整模型/优化器/RNG、2200 selected输出精确一致，再核验全6900 rawRMS/FP64及全部1,038,643,200基础图像像素的已明确解析float32数值合同。独立进程仍使用相同作者网络和Torch内核，不伪称第二套卷积实现。原严格固定2e-5失败已永久保留，新方法自始即登记解析界，未针对任何新诊断标签调容差。

诊断用户1标签保持封存到完整新审计及private所有块/所有成员恢复验证后，finalize_once.py用排他lock只允许一次。用户7–8标签不读、不用于新方法选择；固定参考.9866666666666667/+ .005/tol1e-12不变；失败关闭确切配置，不调C/seed/轮数/宽度/视角；通过也必须另行登记source6确认，不据单个diagnostic立即扩八折。此前八用户反复开发已探索性，不是独立验证。准确率与原创贡献两类证据均未新增，目标未达、自动接续仍启用。

接续先看B/STATE.json、supervisor01/PID.json/TERMINATION.json和/proc实际PID，严禁重复启动/下载。完整producer+audit结束后，只有B/COMPLETE_AUDIT.json pass且complete()全部SHA/TERM守卫通过，才执行`python /workspace/uwb-resnet18-finetune-pilot-20261102/backup.py build`一次。新操作存储键ResNet18FT84Dir指B/complete_backup，ResNet18FT84Manifest为其MANIFEST，ResNet18FT84Pipeline为冻结upload_stream_pipeline.js。先重新核验私有父文件夹元数据，最多3条完整流水线；遇有响应/ID错误先核查并复用，禁止盲重传。按MANIFEST顺序汇总READBACK_*.json到READBACK_INPUT.json，执行backup.py verify完整流式SHA/all-member及receipt/index，然后仅finalize_once.py一次。若失败或中断保留所有日志/checkpoint，不自动重启或改配置；按实际状态修复并单独留痕。

该提交是实际训练前的完整方案与实现，不能将预登记或合成成本宣称为后台真实训练已经启动。启动后会另写PID/初始真实产物与Git状态。全部新路径限uwb_direction1，789源码、用户README、仓库当前可见性/Drive私有性不变，无权重/实际ID/大数组上传Git。
