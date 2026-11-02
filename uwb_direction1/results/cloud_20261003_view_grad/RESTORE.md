# 视角梯度诊断恢复

持久私有Drive地址、字节数、整包SHA见DRIVE_BACKUP_RECEIPT.json；已核对owner-only元数据，并认证重新下载核整包及26个内容文件SHA。原SWAD18检查点不重复上传，本轮没有新训练权重。

只有环境缺失时恢复。先读Drive元数据，再fetch(download_raw_file=true,include_base64=false)，立即通过返回的file_id物化，不输出短期认证URL。校验包SHA和CONTENTS.json，解压到新的独立空目录（tarfile使用filter='data'），不覆盖已有结果。

diagnostic/含216批ID/Gram/CE记录、2304次ID使用的抽样索引、18检查点SHA索引、首批18组完整梯度向量、协议和审计；source/uwb_direction1/含重跑必需源码与冻结协议；report/保留失败float32日志、数值修复、两次成本探针、报告；literature/保存作者公开OGM补充PDF。

如原SWAD模型也缺失，按此前results/cloud_20261003_swad/RESTORE.md及DRIVE_BACKUP_RECEIPT恢复至另一个新目录；本轮使用segment_01/12/24，不使用selected.pt。需原数据y/subj核查质量掩码；重新梯度计算还需signals.npy。既有数据存在时不要重下载。

恢复后用实际新目录替换NEW与SWAD_NEW：

```bash
OPENBLAS_NUM_THREADS=1 python NEW/source/uwb_direction1/tools/audit_view_gradients.py --data /workspace/uwb-restored-20260930 --out NEW/diagnostic --models SWAD_NEW/runs
```

--models只映射来源路径，仍验证18检查点SHA且不修改MODEL_INDEX原字节。不要把历史浮点失败或原始CPU耗时误称为新恢复运行的结果。
