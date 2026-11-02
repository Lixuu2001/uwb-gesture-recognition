# 2026-11-02 — 官方完整 TCN20/B64 实际CPU成本通过，完整真实训练器/审计器联调通过并预登记

最新父提交 e3b9002acce8e57fc73accbcac15d4e495a2a2c7 已记录 Encoder 完整审计、40块/113成员私有验证及固定诊断93.3333%失败，确切配置关闭。本次未重复旧训练、下载或评分。

完整发布TCN20/B64成本在51份远端逐字节回读后执行一次。原成本子进程71616正常退出、无超时；wall5.1807209679391235s/CPU5.129033000000001s，RSS449104KiB。固定全轨迹资源公式结果996.7140554799698s（16.6119分钟），通过12h/14GiB资源门槛。实际成本文件及独立重算证明保留；没有TCN真实UWB值/标签或拟合，成本不是准确率结果。

真实训练器、全部更新审计器和私有恢复工具已实现，研究目录 /workspace/uwb-tcn-pilot-20261102。官方源码2f8c2b817050206397458dfd1f5a25ce8a32fe65完整8x25/kernel7/16次因果扩张weightnorm卷积、普通Dropout.05、末时刻Linear/logsoftmax、180162参数/52参数张量，均保留。源码发布CLI与论文表格/历史运行时的差异已披露，不静默修改legacy weightnorm初始化缓存，也不称原创模型或论文原数字复现。

固定20轮/B64/seed20261029/单CPU，Adam.002/NLLLoss，无gradientclip/permutation/shuffle。native Torch DataLoader沿canonical源清单行序，shuffleFalse/drop_lastFalse/num_workers0/generatorNone，其迭代器对全局Torch RNG的base-seed消耗保留。训练3–6/4700，验证2/1000，诊断1/1200；74训练批(末28)/16验证批(末40)/19诊断批(末48)，1480更新/94000训练出现。第10、20轮验证后才将LR除10，固定第20轮模型，无best-validation挑选。full signed RMS90x567仅转置为Torch567x90，12类，无追加归一化、resize或增强。

保留每批实际ID/损失/更新号、所有轮指标/LR，以及initial/first/epoch10/rollinglast/finalinference完整state_dict/Adam/modes/Torch+NumPy+PythonRNG、原生initial/last权重及两组冻结未评分log probabilities。原生weightnorm的重复state_dict键保留，派生缓存由forward hook重建，不能把单独native权重文件当作完整优化器/RNG恢复。生产/回放使用训练与验证标签，不提供诊断标签加载入口。

8项协议/角色/完成/备份边界测试通过。完整20轮/60更新合成生产->审计逐项对齐全部5状态、2原生权重、ID/损失/指标/LR/两组logprob，损坏ID及user8角色拒绝。另20轮/40更新engine证明LR10/20时序、epoch10freshrestore->后10轮逐项一致、3种错误身份/设置/游标拒绝、初始恢复及native保存RNG隔离；独立首批NLL使用实际DataLoader/dropout RNG复核。小样本/B2/seed7均为纯合成工程夹具，不是替代科学配置。预登记前107项冻结先决文件（含7数据/参考字节哈希及选定27运行文件）与实际成本/已关闭前任证据已核查；选定运行文件不等于整个OS/运行库冻结。

协议SHA b9dfe45dbec4d02f9160005c6142cc3ead5fb07810d4a8b3fe1e5c88a6504f53；FROZEN SHA821f68c38827d43786b3f1e629c28718ffabe1b1cdefa72f3eac520f5e7caa12。此真实预登记快照尚未启动TCN训练；下一步必须全部远端文件逐字节回读、匹配marker后只启动一次并核验真实PID/首步，再私有回读初始首步保险。冻结后不改代码/测试/日志，不重跑测试。

控制器先监督train1100wall/1080soft/1100hardCPU，再同等有界完整回放，无科学自动重试/resume/Drive/Git/评分。全部1480更新/20轮状态与输出、rawRMS/cache/FP64全6900source检查和正常审计终止通过后，还需完整模型集私有raw/整包/全部成员回读，才运行固定finalize一次。初始首步保险不是最新或完整包。Drive owner-only/最多3并发/先metadata/rawtrue-base64false/立即校验，已知ID不重传、错误响应和有界传输修复留痕；实际ID/数组/模型二进制/私有原作者源码不进Git，789源码快照不动。

同ID TimeMIL98.6667%参考、+.5pp门槛不变；失败关闭确切20/B64/seed配置，成功才另登记source6。用户7–8标签不用于新选择，已有八用户反复开发导致探索性；不称独立/八折提升或原创贡献。目标未达，继续既有云端自动接续6abc995e1620819195fcff77dbd81118，不依赖Windows。
