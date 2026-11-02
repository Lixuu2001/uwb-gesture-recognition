# 恢复本轮源频带读出

按DRIVE_BACKUP_RECEIPT.json的持久Drive URL认证下载，先核压缩包SHA。只能解到新的独立空目录；归档内runs/含54组模型、预测、分割和结果以及协议/汇总/审计，source/含必要源码。按ARCHIVE_CONTENTS.json逐项核大小与SHA。全部228个内容文件已通过上传后认证重下载校验。

不含原始数据或features.npz；后者可用冻结脚本和已验证signals.npy重建（完整实验约18.5秒，实际环境可能变化）。这是闭式线性读出，没有迭代训练优化器检查点。重新运行应使用新的输出目录。ARCHIVE_RECEIPT.json中uploaded=false是打包前快照，远端状态以DRIVE_BACKUP_RECEIPT为准。
