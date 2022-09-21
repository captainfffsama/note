#git

# 删除本地分支和远程分支
```bash
#查看所有分支
git branch -a

# 删除远程分支 删除gitea源的chiebot_dev分支 若远程分支已经不存在 可以加-D删除
git push gitea --delete chiebot_dev

#删除本地分支 chiebot_dev
git branch -d chiebot_dev
```

# git刚刚本地提交错了
```bash
git reflog
git reset HEAD@{你要回去的索引}
```

 # 我提交完了发现自己注释没写好怎么办?

```bash
# 直接改 改完执行
git add .
git commit -amend
```

# 手抖把开发分支的内容提交到了主分支

```bash
git checkout 开发分支
git cherry-pick master
git checkout master
git reset HEAD~ --hard
```

# 写到一半有事情要离开，但是我又不想提交怎么办

```bash
git stash
# 回来之后
git stash pop
```

# 我有提交癌，写完一个小功能，回过头我发现自己已经提交过10次了，但是我还没推送，如何把git记录弄整洁点？

```bash
git rebase -i HEAD~10
#然后在打开的vim里把除开最底层的快照前面字母都改成s，最后一层改成p，然后保存退出
git rebase --continue
```

注意：rebase使用的时候一定不要涉及推送到远端的镜像！！！不要对在你的仓库外有副本的分支执行变基。

# git 中文显示不好
```bash
git config --global core.quotepath false
git config --global gui.encoding utf-8
git config --global i18n.commitencoding utf-8
git config --global i18n.logoutputencoding utf-8
```
https://blog.csdn.net/u012145252/article/details/81775362
https://blog.csdn.net/Tyro_java/article/details/53439537

# git 免密码
```bash
git config --global credential.helper store
```