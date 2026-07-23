# Git 本地练习仓库

这是一个真实的 Git 仓库，用来观察 Git 的三个核心区域：

1. **工作区**：你正在编辑、但还没提交的文件。
2. **暂存区**：通过 `git add` 选中，准备放进下一次提交的内容。
3. **本地仓库**：通过 `git commit` 保存下来的历史快照。

## 先看看这个仓库

在 PowerShell 中进入本目录，然后运行：

```powershell
& '..\.tools\mingit\cmd\git.exe' status
& '..\.tools\mingit\cmd\git.exe' log --oneline --graph --decorate --all
& '..\.tools\mingit\cmd\git.exe' branch
```

这里使用工作区中的便携版 Git，是因为创建 Demo 时系统的 `git` 命令尚未加入 PATH。正常安装 Git 后，可把每条命令前面的 `& '..\.tools\mingit\cmd\git.exe'` 简写成 `git`。

## 自己完成一次 add 和 commit

先在 `app.txt` 最后一行写一点内容，然后依次运行：

```powershell
& '..\.tools\mingit\cmd\git.exe' status
& '..\.tools\mingit\cmd\git.exe' diff
& '..\.tools\mingit\cmd\git.exe' add app.txt
& '..\.tools\mingit\cmd\git.exe' diff --staged
& '..\.tools\mingit\cmd\git.exe' commit -m "练习：修改问候语"
& '..\.tools\mingit\cmd\git.exe' log --oneline --graph --decorate --all
```

## 上传到 GitHub

详细步骤见 [GITHUB-UPLOAD.md](GITHUB-UPLOAD.md)。

