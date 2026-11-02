# 私有SWAD源先导模型恢复

远端目录与7个包的持久URL/ID/字节数/SHA见DRIVE_BACKUP_RECEIPT.json。先读取Drive元数据确认private，再fetch(download_raw_file=true,include_base64=false)，取file_uri.file_id去掉sediment://前缀后通过download_file物化；最多3个并发。不得打印签名URL或改分享权限。

每个包须先核字节数及SHA。创建一个全新的空恢复目录：metadata.tar.gz解压到该目录；held3.tar.gz至held8.tar.gz解压到其runs子目录。因此最终结构是新目录/runs/identity.json和新目录/runs/held3/selected.pt等。Python tarfile解压使用filter='data'。用根目录ARCHIVE_CONTENTS.json逐项核查runs/下174个训练产物SHA；不要把六个held目录解压到metadata所在根目录后直接运行。

元数据包包含AUDIT/SUMMARY、源码、协议、许可证和训练日志。源码在新目录/tools，官方SWAD快照在新目录/research/sources/swad_official。selected.pt包含ERM/tail/SWAD三份state，segment_01.pt至segment_24.pt是每轮全部更新的区间平均。

不含原始RF、prepared信号数组或progress.pt优化器状态。原数组仍使用既有私有163块索引恢复；信号缓存可用已存预处理重建。原云端runs目录仍保留progress.pt用于精确恢复；这些最终模型包适合推理/选段重放，不能声称仅凭包内模型精确续训原优化器。

ARCHIVE_RECEIPT.json是上传前快照，uploaded=false不表示远端备份失败；远端事实以DRIVE_BACKUP_RECEIPT.json为准。模型包内早期研究文档保留打包时原字节，最新完成/停止决定以GitHub当前RUN_REPORT与CLOUD_CONTINUATION顶部为准。
