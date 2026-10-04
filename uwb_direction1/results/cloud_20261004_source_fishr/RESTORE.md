# 梯度方差对照训练恢复

先核对/workspace/uwb-fishr-source-runs-20261004/queue_status.json和实际PID，禁止重复启动。epoch结束原子保存progress.pt：model、AdamW动量、scheduler、CPU RNG、best/history/tail、两套EMA原始历史/步数/固定scale/校准范数。不可只恢复网络或遗漏EMA。源码/helper/协议/prepared/基线SHA必须与identity一致；未完成epoch从最近完整checkpoint重放，已完成result逐个files SHA后复用。

运行命令见LAUNCH.json。云端原始/prepared/6旧基线存在则复用；基线完整状态依此前SWAD和BASELINE_STATE私有回执。恢复远端时只用START_DRIVE_RECEIPT或最终DRIVE_BACKUP_RECEIPT的持久URL，先核私有metadata，再raw fetch=true/include_base64=false，最多3并发，立即物化并核块/整包/成员SHA；提取到新的独立目录。

训练完成自动24轨迹独立推理和跨臂warmup/校准一致性审计；本地归档watcher打包558训练产物，但不会自动上传Drive。后续仍须私有上传、认证回下载并提交真实结果；不能把本地归档当远端备份。用户研究目标尚未达成。
