# Windows 本地迁移步骤

2026-09-30 更新：本次迁移已经完成，无需用户再手动上传。私有 Drive 的163个分块已在云端恢复为86个原文件，SHA-256全部一致。当前结果见 [恢复与实验报告](results/cloud_20260930/RUN_REPORT.md)。下文保留为历史导出操作；以后恢复可直接使用 [持久下载索引](migration/20260930/CLOUD_DOWNLOAD_INDEX.json) 和已连接的 Drive 工具。

目标仓库：`Lixuu2001/git-local-demo`，私有；分支：`codex/uwb-direction1-cloud-20260930`。此分支包含迁移工具与研究记忆，原实验代码需要从本地读入。主分支不会被迁移脚本修改。

## 1. 下载迁移分支

在 PowerShell 执行，选择一个尚不存在的新目录：

```powershell
git clone --branch codex/uwb-direction1-cloud-20260930 --single-branch https://github.com/Lixuu2001/git-local-demo.git D:\uwb-cloud-transfer
```

私有仓库需要本地 Git 的账户授权。正常使用 Git Credential Manager 的浏览器登录；不要将访问令牌发到聊天。如果 `git` 未加入 PATH，可先把 `D:\Git\cmd` 加入当前窗口 PATH：`$env:PATH = 'D:\Git\cmd;' + $env:PATH`。

## 2. 先把代码和已有研究记录传到 GitHub

```powershell
powershell -ExecutionPolicy Bypass -File D:\uwb-cloud-transfer\uwb_direction1\tools\migrate_local.ps1 -CodeOnly -OutDir D:\uwb-code-package-20260930
```

默认读取：

- 数据与旧实验：`D:\uwb_gestures`。
- 新实验工作树：`C:\Users\19252\.codex\worktrees\uwb-user-calibration\uwb_gestures`。
- Python：`D:\D_conda\envs\pytorch\python.exe`。

这些位置来自上一轮记录。位置变化时传入 `-DataRoot`、`-CodeRoot`、`-Python`。不要把 `-CodeRoot` 改成无关目录。脚本会保留原项目，在新克隆内增加 `uwb_direction1/original/`，提交源码、旧报告和来源清单，再推送该研究分支。Git 作者信息缺失时配置本地身份后重试提交；推送失败时本地提交保留，完成 Git 登录后在克隆目录运行 `git push`。

导出同时读取两处代码：同路径以新实验工作树为准，另补充数据根目录的旧实验代码与正式研究文档。源码选取 `.py/.md/.toml/.yaml/.yml/.ini/.cfg/.sh/.ps1/.m` 和 requirements 文本，旧结果只带选定的小型元数据。数据目录、隐藏目录、环境依赖与模型权重排除；完整 Git 历史、论文 Word 和全部历史产物没有迁移。生成的 `SOURCE_MANIFEST.json` 逐文件记录来源与 SHA256，便于核查缺失依赖。

## 3. 导出真实数据

已上传代码的用户先更新迁移工具（2026-09-30 补充了训练配置、来源记录和源用户平滑矩阵）：

```powershell
git -C 'D:\uwb-cloud-transfer' pull --ff-only origin codex/uwb-direction1-cloud-20260930
```

代码上传后，仍在本地 PowerShell 执行：

```powershell
& 'D:\D_conda\envs\pytorch\python.exe' 'D:\uwb-cloud-transfer\uwb_direction1\tools\transfer.py' pack --code-root 'C:\Users\19252\.codex\worktrees\uwb-user-calibration\uwb_gestures' --data-root 'D:\uwb_gestures' --out 'D:\uwb-data-package-20260930'
```

默认包含去静态杂波后的原始 UWB 数组、标签、用户编号、旧数据清单、小型质量掩码/ID 元数据，以及可用的八折冻结激活和多视角预测。另收集存在的训练配置、特征来源、折划分和源用户平滑矩阵；这些旧记录缺失时可继续迁移原始数据，但不能声称已完整重放历史运行。保留原文件字节，不重排记录、不猜测标签、不改变 dtype。每块原始字节不超过 24 MiB，独立 gzip 压缩后仍低于 32 MiB。需要额外复用旧 prepared 大数组时加 `--include-prepared`；这会增加传输量。缺少任何必需文件时停止并列出路径，不造数据。

输出是 `D:\uwb-data-package-20260930`。先在当前云端聊天上传 **`TRANSFER_MANIFEST.json`**，便于确定实际大小与块数，再上传 **`chunks` 目录内全部 `.gz` 文件**。附件会保留扁平文件名；云端下载时将按清单放回 `chunks/`。文件较多时可分批上传，所有块到齐才恢复。上传这个数据包不需要提交数组到 GitHub。

较大的数据也可以放在你已有、云端可访问的存储中，再提供下载地址。当前云端只允许若干软件包与 GitHub 域名，尚未配置私人数据存储权限；访问新存储前需要核查实际网络配置。GitHub Release 的私有大附件也需要额外下载认证，本工具没有自动配置该认证。

可在本地先核对压缩块：

```powershell
& 'D:\D_conda\envs\pytorch\python.exe' 'D:\uwb-cloud-transfer\uwb_direction1\tools\transfer.py' verify --package 'D:\uwb-data-package-20260930'
```

原数据很大，需要输出磁盘空间存储压缩副本；工具会逐块读写，不将整个数组加载进内存。默认不含大模型 checkpoint，不影响先恢复预处理与新方法训练。

## 4. 云端接续操作

这部分由云端助手执行，用户不需要在本地执行：

```bash
python uwb_direction1/tools/transfer.py verify --package /workspace/uwb-transfer
python uwb_direction1/tools/transfer.py restore --package /workspace/uwb-transfer --out /workspace/uwb-data
```

然后同步分支中的 `original/project`，检查原 loader、质量掩码、预处理和折划分。迁移会改变路径，旧产物绑定的 Windows 绝对路径不能直接视为新的云端运行身份；路径适配后需要单独记录新来源清单。完整文件哈希通过后才进入真实实验。
