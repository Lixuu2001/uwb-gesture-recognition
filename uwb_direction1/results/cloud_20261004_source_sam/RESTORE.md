# 源端SAM恢复约定

先检查/workspace/uwb-sam-source-runs-20261004/queue_status.json和实际进程状态，禁止活动队列重复启动。epoch结束原子保存progress.pt，包含普通AdamW全部动量/scheduler/CPU RNG/model/tail/best/history。没有嵌套第三方SAM优化器状态遗漏；source_sam是一步函数。中途未完整保存的epoch从最后完成epoch重放，训练源码/helper/协议/数据SHA与identity必须一致。已完成result必须校验原files再复用。

只有需要恢复时读取START_DRIVE_RECEIPT.json（若尚不存在表示还未备份，不得声称有）或最终DRIVE_BACKUP_RECEIPT.json持久索引。先核元数据私有owner-only，raw fetch=true/include_base64=false最多3并发并立即物化，逐块/重组/每成员SHA后提取到新的空目录。旧原始数据/prepared和6基线已存在不重复下载；基线依SWAD7模型包及共享实验BASELINE_STATE_DRIVE_RECEIPT补充optimizer包。

实际运行命令见LAUNCH.json；同原命令支持检查点恢复，前提是原进程确认终止、原身份全部匹配。正式队列结束自动独立推理审计。原数组/权重不提交Git，最终归档watcher不负责远端上传。
