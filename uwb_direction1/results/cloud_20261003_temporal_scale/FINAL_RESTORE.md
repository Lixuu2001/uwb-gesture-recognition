# 最终私有恢复说明

当前云端已存在完整运行目录 /workspace/uwb-temporal-scale-runs-20261003 和归档目录 /workspace/uwb-temporal-scale-final-20261003。先检查，不重复下载或重训。完成队列不再启动。

仅环境确实缺失时使用 DRIVE_BACKUP_RECEIPT.json 的 archives 列表。先核验父目录12AGMwT99yXCTYBQocQaX901Br-IelmOf及文件metadata保持私有，按每包ordered_parts所列顺序及parts元数据认证fetch(download_raw_file=true, include_base64=false)，最多3并发，立即物化；不打印认证短期URL。相同大小且SHA匹配的已有块复用，冲突文件保留另存。逐块校验后顺序拼接，核whole archive SHA和大小。

在新的独立空目录解包；拒绝绝对路径、..、符号链接及重复覆盖，只提取contents清单中的普通文件，再校验每成员大小和SHA。训练包路径是 time2/heldN/... 或 range2/heldN/...；metadata包保留 runs/...、tools/... 等前缀，恢复时明确把元数据映射至运行根，不盲目覆盖。interruption-recovery单独保留为历史证据，不拿epoch11快照覆盖已完成的epoch24状态。26包应总计410成员，完整传输380775626字节。

新轨迹保留24轮segment、source_orders、selected、progress完整AdamW/scheduler/RNG/best/tail、split/预测/结果。原6普通基线不在新训练包中，依 results/cloud_20261003_swad/DRIVE_BACKUP_RECEIPT.json 与 results/cloud_20261003_view_sharing/BASELINE_STATE_DRIVE_RECEIPT.json 两组恢复。两回执路径均已核实存在。原始86文件按 migration/20260930/CLOUD_DOWNLOAD_INDEX.json 和迁移工具恢复，仅当真正缺失；prepared信号SHA、源代码/helper/协议SHA必须匹配SUMMARY.identity。数据和权重不提交Git。

复核采用冻结 tools/audit_temporal_scale.py 的实际CLI，先 --help 核参数。它应审计18轨迹、534产物SHA和全部ERM/tail验证/诊断logits；不要通过重新训练改变已保存研究。ARCHIVE_RECEIPT uploaded=false只是历史打包状态，以最终DRIVE_BACKUP_RECEIPT的完整认证验证为准。
