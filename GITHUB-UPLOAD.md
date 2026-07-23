# 把本地仓库上传到 GitHub

## 方法一：网页创建空仓库，再用命令上传

1. 登录 GitHub，点击右上角 `+`，选择 **New repository**。
2. Repository name 填 `git-local-demo`。
3. 选择 Public 或 Private。
4. **不要勾选** README、`.gitignore` 或 License，因为本地仓库已有这些文件。
5. 点击 **Create repository**，复制页面提供的 HTTPS 地址，例如：

   `https://github.com/你的用户名/git-local-demo.git`

6. 回到本目录，执行：

```powershell
git remote add origin https://github.com/你的用户名/git-local-demo.git
git push -u origin main
```

以后有新提交时，只需：

```powershell
git push
```

## 登录说明

GitHub 不再接受账号密码作为 Git 命令的登录密码。Windows 上推荐安装官方 Git 后，使用随附的 Git Credential Manager；第一次 `git push` 通常会弹出浏览器，让你登录并授权。

如果提示 `remote origin already exists`，先查看：

```powershell
git remote -v
```

若地址填错，改成正确地址：

```powershell
git remote set-url origin https://github.com/你的用户名/git-local-demo.git
```

## GitHub 上已有 README 怎么办

最简单的做法是重新创建一个不带 README 的空仓库。如果要保留远端 README，可以先运行：

```powershell
git pull --rebase origin main
git push -u origin main
```

遇到内容冲突时，不要盲目强制推送；先解决冲突并完成提交。

