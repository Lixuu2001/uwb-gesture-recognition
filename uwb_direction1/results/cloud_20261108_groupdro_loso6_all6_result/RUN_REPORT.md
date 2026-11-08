# Published GroupDRO native full implementation: all6_result

## 2026-11-08 GroupDRO完整六折结果已唯一评分：93.565404% vs GAP93.667172%，固定配置关闭

预登记222fc556/全6冻结c87d6af0/最终私有精确索引c8cebdc830a5f4406791818ddd84e5dfef1099fd依序完成后，77块/2010载荷（2011归档成员）/562618273字节全部owner-only metadata+raw回下载+块/包/成员SHA通过，失败0/自动重试0，archiveSHA5459865e09bc4ce393d810273b0e738c337e25e3aa29bc1e7e9e699df140e7e1。再完成12组无标签盲推断（输出逐位一致、参数/BN不变），最后才由原冻结scorer一次打开held真值配对；本次once已消耗，不得重训/重评/换eta或seed补救。

用户等权present-class macro recall为93.5654040404% vs原GAP93.6671717172%，-0.1017676768pp；3胜0平3负，均值+0.5pp/≥4胜未满足，最差paired-1.8pp条件满足但整体门槛失败。h1..6 GroupDRO accuracy依次98.583333、96.900000、89.666667、94.000000、97.333333、84.909091%，原GAP98.416667、98.700000、90.000000、94.250000、96.000000、84.636364%；paired +0.166667/-1.800000/-0.333333/-0.250000/+1.333333/+0.272727pp。最差用户84.909091%略好于84.636364%，不能据此称整体提升；局部正效果保留。pooled accuracy独立为93.594203% vs93.652174%。当前每人现有类别各100，因此accuracy和present-macro仅数值相等，定义不混淆；h2缺10/11、h6缺3，recall=null。逐类support/recall、全部逐人指标/成本见FINAL_REPORT/SUMMARY。

固定eta.01/fullnativeGAP617612/全部三雷达RMS/B64/64/seed20261029/nativeAdamWLookahead/globalBCE验证选模，6生产64+6独立fresh64及全轨迹/原生/q/角色/输出审计均通过。完整新phase wall29556.896173048997/CPU29595.509859692秒(<12h)，补充只读审计6.444258/6.442113秒，最终归档29.168755wall/29.160023waitedCPU秒，原始回读工具窗口1612.910秒含provider等待，盲推断完整进程35.659789wall/35.999821CPU秒，唯一评分完整进程6.222610wall/6.633296CPU秒。原GAP held2..6训练320epoch12416.149607wall/12416.359542CPU秒，h1成对实验孤立cost未记录不补造；MMD首折5022.569200/5019.332419已在MMD6合计内，不重复加。旧MMD/VREx/Fishr/IRM及历史部分保护/未知操作成本保留，不当0或抹去。

/usr/bin/time缺失的exit127启动故障已保存原stderr并用Python标准库计量修复，失败时未执行归档，无模型/科学配置重跑。全46冻结文件及1886产物SHA不变；精确重放、局部公式参考和只读核查不保证排除全部共同实现错误。此结果关闭确切配置，route=`close_exact_GroupDRO_LOSO6_eta0p01_nativeGAP64_seed20261029_no_rescue`，不执行仅主门槛通过才规定的eta0归因对照，不把旧缩小GroupDRO结论改写或否定全部GroupDRO版本。

下一路线在held评分前已写入并随c8cebdc8精确登记：若GroupDRO失败，推进SWAD LossValley默认3/6/.3在完整617612/native64/seed20261029上的实现审计；历史SWAD为users3..8/24轮/seed2026/legacyGN不同骨干，旧失败保留。已读取既有官方LossValley/AveragedModel/config并解析官方当前不可变commit252190eb881169e736beccf51af5caaa5c5cf666，接下来核查trainer/BN接线并具体实现、生成原生检查/成本/完整6fold执行器/源和私有initial release后才可新fit；本条时没有SWAD真实训练或完整runner。发表方法可作强基线/迁移实验，不因非原创停止实验；不把组件迁移称新算法。用户7-8不用于新选择，历史8人多次开发仍探索性，不称独立新用户验证。现有自动任务/频率/环境不变，无新自动任务/重复训练，目标未达继续。以下为历史。

