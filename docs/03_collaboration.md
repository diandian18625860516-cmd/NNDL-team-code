# GitHub 共享和小组协作

## 发布本地仓库

此文件夹已经初始化 Git 并保存首次提交。请在 VS Code 用“文件 → 打开文件夹”打开 **NNDL-team-code** 本身。

1. 按 Mac `Shift+Command+P`（Windows `Ctrl+Shift+P`），输入 `Publish to GitHub`。
2. 按提示登录自己的 GitHub；仓库建议名 `nndl-logical-anomaly`。
3. 选择 **Private repository**。发布的是这个文件夹的提交记录。
4. 在 GitHub 网页确认 README、scripts、vendor 都存在；没有 `paths.local.json`、原图、模型权重或账号配置。
5. 仓库 `Settings → Collaborators → Add people`，输入组员 GitHub 用户名；组员接受邀请。

本次工具环境无法从终端解析 GitHub 地址，浏览器连接也不可用，因此没有建立远程仓库或发送邀请。如果 VS Code 同样连不上 GitHub，先使用同目录外生成的代码 ZIP 给组员阅读，恢复网络后再以远程仓库为共同入口。

GitHub Free 支持私有仓库协作者。[官方邀请说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository) · [VS Code 发布说明](https://code.visualstudio.com/docs/sourcecontrol/quickstart)

## 每个组员第一次加入

在 VS Code 命令面板选择 `Git: Clone`，粘贴本小组仓库地址，选择本机保存位置。打开后先看 README，运行源码检查和环境检查，复制并填写自己的 `paths.local.json`。Clone 得到共同代码，数据仍由自己下载或使用统一云端目录。

## 每次修改

先同步主分支，再为任务创建分支，例如 `task/t02-data-check`。每次提交说明改了什么，如“增加 Pushpins 路径检查”。完成后推送分支并创建 Pull Request（请求把改动合入主分支），写清运行命令和结果，由另一位组员看过再合并。

主分支 `main` 保存大家确认可用的版本。出错时把报错和命令贴到任务中；不要用强制推送覆盖别人的修改，也不要共享 GitHub 密码。

通俗解释：Commit 是保存本地一个版本；Push 是上传版本；Pull 是拿到别人已合并的版本；分支是自己的修改草稿；Pull Request 是请别人审核这份草稿。

## 上传前检查

查看源代码管理中的文件列表和改动。`.gitignore` 已排除标准数据目录、权重、输出、虚拟环境和个人配置；如果把这些文件放在其他目录，仍要自己检查。组员只提交自己理解的修改，并在任务表中登记贡献。
