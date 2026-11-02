# UWB 私有云端迁移

数据包：D:\uwb-data-package-20260930
私有 Google Drive 目录：https://drive.google.com/drive/folders/12AGMwT99yXCTYBQocQaX901Br-IelmOf
云端目标聊天：推进方向一手势识别 (01a0f042-ddcd-7230-a3c7-ce27874f00a2)

## 下载和校验

1. 用已连接的 Google Drive 插件下载 CLOUD_DOWNLOAD_INDEX.json 和 TRANSFER_MANIFEST.json。保持所有文件私有，不创建公开分享权限。
2. 索引保存每个文件的持久 Drive id/url、相对路径、压缩字节数及 SHA-256。分批调用 google_drive_fetch(download_raw_file=true, include_base64=false) 获取认证下载引用，并立即下载该批，不把短期下载 URL 当成永久地址。
3. 将清单保存到 /workspace/uwb-transfer/TRANSFER_MANIFEST.json，163 个分块保存到 /workspace/uwb-transfer/chunks/。已有同大小且 SHA-256 匹配的文件可跳过；冲突文件应保留后处理。
4. 清单 SHA-256 必须是 10516baf0b49a84e6a39169ef447fffcdcb29defa2a6cc79e301dc938ddceb4d。
5. 用已有脚本先 verify，再 restore 到独立的新空目录。restore 会校验解压后的每个原文件 SHA-256。

```bash
python /workspace/uwb-cloud/uwb_direction1/tools/transfer.py verify --package /workspace/uwb-transfer
python /workspace/uwb-cloud/uwb_direction1/tools/transfer.py restore --package /workspace/uwb-transfer --out /workspace/uwb-restored-20260930
```

预期：163 个分块，压缩后 2006404772 字节，解压后 86 个文件、2141981100 字节。
原始快照 SOURCE_MANIFEST.json SHA-256：8f19d50aacdcc2cfe3457e9dc4785a0cf440ef76c5777a751806b5575a9f2884。

## 当前验证依据

本地全部 163 个分块的大小和 SHA-256 已通过。
一个 23627940 字节大分块完成上传后重新认证下载，SHA-256 与本地一致。
目录及试传分块权限已读回：shared=false，仅 owner。
完整上传及云端恢复状态以 UPLOAD_JOURNAL.json 和云端 RESTORE_VERIFIED.json 为准。

