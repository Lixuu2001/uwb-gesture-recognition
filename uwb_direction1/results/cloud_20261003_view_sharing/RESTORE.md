# 最终私有恢复（2026-10-03）

本实验12条新轨迹及6条复用基线已完成，完整审计通过，推进门槛失败。当前没有训练运行，PID16830/17076仅是历史。当前云端文件齐全，不需恢复，也不要重复启动已完成训练。

最终回执DRIVE_BACKUP_RECEIPT.json包含25逻辑归档、490,011,185字节、46个私有Drive对象。每个逻辑包具有整包bytes/sha256；14包有file_id/url，另11包有有序parts（32块，每块≤8MiB）。全部已经认证下载并核对整包及389成员SHA。旧START包只保留早期历史，不用于替代最终状态；ARCHIVE_RECEIPT.json的uploaded=false是本地打包时快照。

环境丢失时按以下顺序恢复，不要求用户重新上传：

1. 原始86文件依migration/20260930/CLOUD_DOWNLOAD_INDEX.json的私有163块恢复并核原SHA。prepared按已有脚本重建并核原SHA。训练源码采用预注册fd4d754b57f2aa2787d7d896eee9e26828cd234e，环境Python3.12、PyTorch2.14.0+cpu、numpy2.3.5；不要混入后续诊断准备代码或修改身份依赖。
2. 旧共享基线依results/cloud_20261003_swad/RESTORE.md恢复七包至NEW_SWAD/runs，再依BASELINE_STATE_DRIVE_RECEIPT.json的三包补齐六个heldN/progress.pt。仅七包不含optimizer/RNG状态，不能通过本实验基线身份审计。
3. 本实验最终每个Drive对象先核元数据shared=false、owner权限、父目录及大小，再fetch(download_raw_file=true,include_base64=false)，立即download_file物化；最多3并发，不输出认证URL。每个对象核SHA。含parts的逻辑包严格按数组顺序连接原字节，然后核整个逻辑包bytes/sha256；不能分别解压chunk。
4. 在全新独立空目录NEW_SHARING恢复。metadata.tar.gz成员含runs/、tools/、research/、cost/等，从NEW_SHARING根解压；其他24模型包成员以untied128/heldN或shared177/heldN开头，从NEW_SHARING/runs解压。先验证各包CONTENTS_SHA256.json与ARCHIVE_RECEIPT.json记录、全部成员的bytes/sha256，再安全提取，拒绝绝对路径、..路径及链接，使用tarfile filter='data'。每包清单保存到独立manifests子目录，不让同名CONTENTS_SHA256.json互相覆盖。已有同SHA目标可跳过，冲突保留且换新目录。
5. 检查完整结果、所选模型、12个progress.pt和288个segment权重。完整状态含optimizer/scheduler/RNG及平均队列。总389成员=360训练产物+29公共元数据/源码/日志；复用基线另在NEW_SWAD。迁移数据/目录身份需匹配，不改配置绕过校验。历史LAUNCH.json内绝对路径仅供审计，不能直接用于另一环境。

本实验已经失败，不因恢复而重新训练。若需要独立复算，使用已冻结审计工具并把baseline、data、prepared、out参数替换为恢复后的对应目录，保存新审计文件，禁止覆盖原始结果。下一候选仅为配对诊断准备，见PAIRING_DEPENDENCE_REVIEW.md，尚无新真实模型实验或增益。

当前Python位于/opt/codex/runtimes/codex-primary-runtime/dependencies/python/bin/python，torch位于/home/agent/.local/lib/python3.12/site-packages/torch。原始RF/prepared数组和模型大文件均不提交Git。
