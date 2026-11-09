# Actual prospective full8 held1 production launched: started

## 2026-11-09：正式八人LOSO held1已实际启动，首步审计通过；无held评分

精确正式预登记9703b6c477068f25d0ef11e4f2d7372f10d4ec9a的39文件全部远端回读/force=false和两块83载荷私有初态raw/allmember回读之后，已建立独立版本PRODUCTION_RELEASE并且只调用launch_fold.py 1一次。真实控制器PID122292/startticks41935831，训练worker PID122293/startticks41936036，boot ad409b48-89ce-4704-beee-3658103d5856，均核验活跃且身份吻合。工作根/workspace/uwb-tail-outer8-production-20261109；不是生成fixture或成功probe重跑，不重训任何旧MMD/GAP/六折队列。

截至2026-11-09T01:48:16.847443+00:00：真实held1生产已提交epoch1/64、111次更新；步日志已观察到update197，未提交suffix不等于新epoch已冻结。首步模型/原生优化器独立重放及NumPy全局BCE梯度检查通过，max gradient difference=7.45058059692e-09，首步checkpoint保存。该折训练用户3–8共7100样本、验证用户2共1000、测试用户1共1200；production64预期7104次更新，随后独立fresh64和3臂outputaudit。没有held真值读取、没有新accuracy/macro/held分数，不会据单人分数续停或改参数。全部8人LOSO尚未完成。

方法/协议仍为早于7/8新访问已选定的native617612/all3signedRMS/global12BCE/B64/64/seed20261029/原生AdamWLookahead.001_wd.0001_.5k6_clip2，post-update dense tail33–64＋训练人BN。native基线用本折独立验证选模，同轨迹无平均参数副作用；native-sourceBN是描述性控制，不能用旧4源人六折均值代替新的6源人匹配基线。正式训练现在按冻结角色使用7/8的训练标签，明确不是“未读7/8标签”；它们不参与新方法/window/LR/seed选择。旧SWAD失败主方法/旧tail正向描述性控制身份不追改，历史全部8人暴露，后续结果仍探索性、非独立新用户/原创/因果验证。

阶段36小时wall/local aggregate CPU上限和workerRSS12GiB/磁盘reserve不变，原测量含余量29.1h仅为预算外推。阶段内累计消耗看folds/held1/LIVE_RESOURCE.json，完成stage后汇总PHASE_LEDGER，包含已等候wall和显式外部私有归档CPU/保守helpercharge；历史16probe、生成检查、初态私有备份和MMD第一折等旧成本仍保留独立账本，不记零/不重复相加。控制器hold1 train→replay→audit完成后将停在waiting_private_backup；只有完整该折私有raw-allmember/费用释放后才启动held2，以此完成8折。后续接续先核验这两个PID/boot/startticks、live日志和checkpoint，禁止再次launch held1或另开worker/候选。

下一次自动接续：若原PID仍活跃，read-only监测并无变化安静；若失败，保留原日志/未提交suffix并核实资源、进程后显式恢复，不能新目录重跑。若held1三stage已审计完成，按EXECUTION_GUIDE逐项build/私有upload最多3完整pipeline/metadata→rawfetch(include_base64=false)→立即物化SHA/verify/费用登记/finalize，再launch held2一次。全部8折模型、日志、预测、审计及私有冻结后才预先固定的一次cohort评分和完整accuracy/present-macro/support/null缺类/用户mean/worst/paired/wins/pooled/成本报告；不根据部分测试人分数决策。

原自有任务6abc995e1620819195fcff77dbd81118仍启用/每小时/原继承同cloud，已只读回核，没有新自动任务、调频或环境更换。原789源码及所有旧科学文件/结果/once锁保持。阶段启动见results/cloud_20261109_tail_outer8_production_started/RUN_REPORT.md和HELD1_LAUNCH_OBSERVED.json；科学SOURCE commit9703b6c固定，后续metadata提交不改变运行源码hash。

