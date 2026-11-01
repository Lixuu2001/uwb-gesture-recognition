# 2026-11-01 — H-Inception 固定CPU100六折确认未过门槛，完整六模型41私有块/478成员已回读；关闭确切配置并预登记下一项Disjoint-CNN完整默认规模资源实测

先GitHub读最新381f58915c4571c76664b0f927eb6208166bf590四主记录，核同一可执行云端。held6原fullreplay audit63747已exit0/no timeout，100epoch/7200updates/460000appearances精确重放，selected53；supervisor wall9669.194579489063s/CPU9667.756389999999s，audit内部9664.315604642034s。initial/first/best/last/final模型/Adam/BN/metrics/callback/modes/allRNG、native selected/last及2300行双预测一致，6900输入RMS+author znorm独立FP64参考按原容差验证。现在5新fold全部500epoch/36000updates/2290000appearances训练并完整审计完成，controller退出，无后台训练/审计。

固定aggregate只在所有5新fold双预测冻结和fullaudit通过后首次统一读取其诊断标签。随后本轮完成六模型全档私有回读，再使用独立只读实现重新由三个方法的原始预测计算各折平衡准确率、普通top1和门槛；6900唯一ID严格属于用户1–6，7–8标签未索引，无fit/选择/调参。独立计算在1e-12内与原汇总一致。所有source数据经反复历史开发，结果明确探索性，不是独立验证或最终八折。

主指标为各用户“存在类别召回率的均值”，再六用户等权平均（平衡准确率）：H-Inception93.7209596%，LocalCNN93.5944444%，GAP93.6671717%。H分别仅+0.1265152pp/+0.0537879pp；对LocalCNN仅3/6严格胜、最差-3.5pp，对GAP4/6胜、最差-3.3333333pp。预登记对每条基线均要求mean至少+1pp、至少4/6胜且worst>=-2pp，**两项均未过**。6900样本pooled普通top1分别93.6666667%、93.5797101%、93.6521739%，不是主gate。禁止拿单held1的99.4167%包装总体提升；旧LTR八用户93.4417%协议不同，不能直接作提升证据。

确切full851434参数/601006可训/nativeKeras-Torch/full90x567/100epoch/B64/seed20261029/source角色/最低trainloss配置关闭，不增加epoch/换seed/调权重/按诊断误差救配置，未启动原先以source6 gate通过为条件的zero17真实消融。此结论关闭该CPU100确认配置，不推断所有H-Inception预算或架构无效。已发表方法移植实验有价值，但本次没有达到稳健提升门槛，更没有原创贡献结论。两条参考流水线输入、预算和优化不同，这是冻结强基线比较，不是因果消融。

完整六模型恢复档300848048bytes/41chunks/477payload+MEMBERS=478已逐块rawfetch/立即物化size+SHA、整档SHA和每成员全验证。包含旧held1完整payload、5新fold全部已产checkpoint/准确IDs/全部更新日志/未评分预测/审计、三方法预测、code和dependency text。raw/maps/runtime另恢复不入Git；原789源码不改。archiveSHAc94fd6694256b21cc688f72d85c58d1e3b7733bb7bbe50cdc83a24fc754f0741，receiptSHA1c29f2fa7a6d908fc1d89459783744304a3d015aea8fbf9b921e95f9a8c6f24c。complete_backup/CLOUD_DOWNLOAD_INDEX.json为持久私有索引；complete_modelset_verified=true/all_completed_models_included=true，无整job自动resume承诺。

保存两项恢复工程故障：part004随机压缩流头被Drive推断image/jpeg，原始SHA/size/私有parent均正确；改为所有stored非native类型均rawfetch并核字节，拒绝native/shortcut。part037物化时exec-server transport closed，part038 upload无返回ID；先确认同环境当前running，复用037原ID，038 exact-name搜索为空后才重试；最后41块均唯一ID、全部成员通过。原错误与修复代码/记录保留，没有重启训练/审计、改变分数或重复上传成功文件；未改分享或输出认证URL。

下一项为已准备作者实现的Disjoint-CNN（ICDMW2021, pp760–769, IEEE9679860），时域卷积与跨通道卷积分解。作者源commit38180ccc1f050a3fee6961f5b4ea3bc132016ab3，完整DCNN_2L2587468参数，作者Main500epoch/B8；不等同历史缩小改写。现固定epoch_candidates=[500]、B8、seed20261029、full90x567x1 signed RMS、author normalise=False，不设缩小epoch救援。只预登记8个合成weighted训练batch/4eval/3native save/3fullstate save的CPU测量，无UWB值或标签。profile独立590/600CPU、620wall上限，公式估算完整500轮<=43200s及RSS<=14GiB才允许下一步另登记真实runner/auditor。源码README及完整默认层结构已核查，论文全文未获得，不能称本轮全文核查或原创方法。

原Disjoint工具阻塞在“成功H单折必须先做source确认”的旧pilot guard；现在该确认已完整结束且失败，新增source6_closure核完整5audit/六模型私有member与summary绑定/确切failed route后再允许读取cost policy。只替换profiler的guard import，测时主体及原算式不改，原工具/guard/source均保留。PRE_ACCESS_GUARD_CHECK已真实通过；先远端预登记并逐文件exact readback，再实际测时。没有宣称新UWB训练已启动。后续资源不达则关闭该固定full500/B8策略转下一项，不扫7–8或诊断标签。

目标仍未达，保留原自动接续。完整结果及表见SOURCE6_RESULT.md、SUMMARY.json、METRIC_CROSSCHECK.json，原5审计与全部恢复证据保持可复查。


| 留出用户 | H-Inception | LocalCNN | GAP |
|---|---:|---:|---:|
| 1 | 99.4167% | 96.8333% | 98.4167% |
| 2 | 99.0000% | 96.4000% | 98.7000% |
| 3 | 88.0000% | 89.5833% | 90.0000% |
| 4 | 90.9167% | 94.4167% | 94.2500% |
| 5 | 98.0833% | 95.3333% | 96.0000% |
| 6 | 86.9091% | 89.0000% | 84.6364% |

H-Inception: BigData2022 DOI10.1109/BIGDATA55660.2022.10020496; author CF-4-TSC commitcf4d0e55ee11b7393ae3d317cdf5f22dcc51ba3e. Full architecture preserved; CPU100 is the preregistered budget, not the author1500-epoch training default.
