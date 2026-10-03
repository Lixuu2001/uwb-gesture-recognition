# 恢复本轮活动区间诊断

持久Drive地址、大小、SHA与权限见DRIVE_BACKUP_RECEIPT.json。文件位于原私有目录，仅owner；已认证重新下载并核对整包SHA及11个内容文件SHA。归档没有新训练权重，因为本轮没有训练。

下载前核对元数据；fetch使用download_raw_file=true、include_base64=false，立即物化其file_id，不输出短期认证URL。先核整包SHA，再检查CONTENTS.json中的路径、大小和SHA。恢复到新的独立空目录，不覆盖旧实验；不要将数组上传Git。

归档结构：diagnostic/有7100个原ID、用户、两种逐转移质量分布、三种区间、汇总、协议、后审计与运行日志；source/uwb_direction1/有诊断和审计源码、digest/quality_mask辅助、原始冻结协议与审阅/报告。

只有环境丢失时才下载，当前结果在/workspace/uwb-event-extent-20261003。已解压归档内的审计命令（仍需原始数据提供ID/质量掩码）：

```bash
OPENBLAS_NUM_THREADS=1 python NEW/source/uwb_direction1/tools/audit_event_extent.py --data /workspace/uwb-restored-20260930 --out NEW/diagnostic
```

以实际新目录替换NEW；重新诊断则运行同目录diagnose_event_extent.py并指定另一不存在的--out目录。报告中的原始本地路径只是来源记录。
