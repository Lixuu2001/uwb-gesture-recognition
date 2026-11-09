# Prospective native dense-tail outer8: worker_prereg

## 2026-11-09：另行固定前瞻八人尾段平均比较，worker生成检查及私有初态完成；尚无真实八折训练

在读取新用户7/8手势值前，依据已冻结六人开发证据，另行选择native64每步密集平均33–64＋source-only BN作为新的前瞻八人方法；seed20261029及全部输入/骨干/优化器/64轮不变。旧SWAD6主方法失败、尾段正向描述性对照及全部原门槛不追改；不是将旧控制改名或重算六折。八人testh/valh%8+1/other6train，原4源人的六折权重/分数不能替代新的匹配基线。用户7/8仍未用于方法选择，全部8人历史开发暴露，未来结果仍探索性。

已实现真实逐折worker/只允许train和validation的标签capability/每步hash/两槽epoch日志/首步NumPy BCE梯度审计/无训练态和RNG副作用的平均/仅训练人lazy BN/3臂封存输出/独立fresh replay；监督器当前只开放生成数据和180秒有界成本probe，缺少production launcher，不会自动开八折。完整617612参数生成数据生产2轮＋新鲜2轮、literal平均算术、原生轨迹不变、输出/模型/优化器/RNG逐一一致、BN/标签越界/日志suffix/重复拟合拒绝均通过。首轮生成检查6.513719wall/6.202866CPU秒；加强固定作者平均文件SHA后的v2检查5.351981wall/5.285274CPU秒，旧证据保留。这不覆盖完整64轮或独立排除所有共享bug，Lookahead周期/新worker原生输出审计仍待补齐。

16个冻结源码/协议和初始模型、实际私有9300IDs、49项native依赖共70载荷已在原owner-only Drive目录备份两块（9,861,739字节），元数据＋raw fetch(download_raw_file=true/include_base64=false)＋立即物化/SHA＋所有成员原始回读通过。不是八折训练完成档案。下一步仅执行已固定16-update成本probe：只取用户3–6训练样本，无验证/测试真值，不取7/8手势值；保留初始/首步/终态、IDs和日志，平均collector全部16步激活只测开销，不是正式33–64方法试验，不计算识别质量。先核对远端本次源码，再建立probe-only release，不能自动准入production。

新的前瞻完整八人阶段上限另行固定为36小时wall/36小时aggregate CPU、RSS12GiB，不挪用或抹去旧MMD12小时阶段及已计第一折成本。probe墙钟/CPU各180秒、RSS6GiB；按16步和训练内forward外推并加50%余量/3600秒reserve，还需核实完整checkpoint/私有归档存储。八折解释门槛在新分数前固定：用户等权mean accuracy增益>=0.5pp、至少5/8严格改善、最差配对不低于-2pp、tol1e-12；accuracy与present-class macro/support/null缺类/mean/worst/paired/pooled/全部成本分别报告，门槛未过仍保留任何正负证据，完整八折/私有/审计freeze前不得单人评分或据此续停。

尚未启动真实成本probe或八折生产；新worker的原生functional输出/优化器周期独立审计、全阶段累计资源监督/折间私有备份handshake、存储生命周期、显式production准入仍需完成。继续具体实现和真实有界资源测量，不重开关闭队列、不扫标签调参、不把准备工作称为精度提升或原创。原789科学源码、MMD旧once和所有旧结果不变；原唯一启用每小时任务及同云环境不变。

详情：results/cloud_20261109_tail_outer8_worker_prereg/RUN_REPORT.md；冻结实现：research/cloud_20261109_tail_outer8_preparation/。

