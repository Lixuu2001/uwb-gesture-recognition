# Real train-only native resource probe and integration/storage audit: resource_probe

## 2026-11-09：真实16-update训练内成本probe及原生审计完成；磁盘失败已修复，八折生产未启动

已在1d90f346精确源码预登记和owner-only初态两块/70载荷全部原始回读之后，完成唯一一次16update原生617612训练成本probe，只取开发用户3–6训练样本。初始/首步/终态、1024个训练ID、日志保留；平均collector16步全开只测开销，不是正式33–64试验，不计算accuracy/macro，不取验证/测试真值或7/8手势值。首步模型、优化器和ID与原GAP held1 checkpoint逐位一致，NumPy独立BCE梯度检查通过。另有生成数据Lookahead第6/12步、恢复状态/原生无平均对照轨迹逐步精确，及原冻结GAP held1共1200个unscored输出通过新lazy/角色接口逐位兼容；没有重训旧GAP或新方法held评分。

真实probe15.524692墙钟/15.430605CPU秒，峰值845212KiB；训练步p95=.571237s，训练内forward batch64 p95=.145737s。全八折production64+fresh64共111872updates，按固定50%余量和3600秒reserve估计104751.914456秒（29.0978小时），低于另行预登记36小时wall/36小时CPU上限。此为外推，不是实测完整八折或6源人BN成本。原MMD12小时阶段及第一折、所有历史成本不改。

原资源门槛compute通过、磁盘失败：需1602898539字节含reserve，实有1253306368；RESOURCE_GATE原失败保留。已重新核对247个旧SWAD私有Drive对象当前owner-only/parent/size元数据，完整旧raw归档2250成员/2249载荷SHA、54冻结源码、240模型/输出refs及SUMMARY不变，然后仅回收1811485836字节冗余本地压缩传输块。原始数组/模型/日志、永久Drive对象、全部清单/247IDs/回读及组合恢复索引仍保留；按需恢复时先metadata后rawfetch，不因传输缓存缺失重训/重评分。空闲约3064627200字节，补充存储复核通过。清理16.591467wall/16.578899CPU秒，原磁盘失败和修复证据均留痕。

有界监督器已通过生成数据完成/资源强制终止/重复运行拒绝/未准入生产标签拒绝；额外phase_supervisor和fold_output_audit已实现并语法检查，但全阶段累计预算及折间private handoff仍缺端到端生成检查，不能以语法通过宣称生产准入。当前原冻结admission仍拒绝原RESOURCE_GATE失败，PRODUCTION_RELEASE和PHASE_LEDGER及实际fold目录均不存在，不覆盖失败门槛伪造成功。下一步完成phase端到端测试和版本化准入升级，绑定原协议不变＋storage_recheck＋新监督/审计源码，再精确发布/私有初态补充原始回读，才启动held1一次；不重复成功probe或任何关闭六折。

本轮probe模型/IDs/日志及资源/审计/空间修复31载荷已新增4块私有备份并全部raw/allmember回读；与初态2块/70载荷组成恢复链，不是完整八折产物，不授权重跑。网络/构档未单独测CPU的部分明确记为unmeasured，不记零，COST_LEDGER列实际范围且不重复加子进程成本。

本轮没有新accuracy或macro提升结果、完整八折训练/重放/评分。既有尾段平均从已冻结1–6开发证据独立选作未来八人方法，不倒改旧SWAD失败或旧tail描述性控制身份。7/8手势值仍未用于新选择，所有8人历史暴露，结果只能探索性；不称原创/独立新用户验证。原唯一自有任务仍启用每小时/继承同云环境，未创建任务、未改频率/环境，当前无活跃训练。

详情：results/cloud_20261109_tail_outer8_resource_probe/RUN_REPORT.md；NEXT_STAGE.md明确剩余生产准入工作及禁止重复probe/关闭队列/旧once。

