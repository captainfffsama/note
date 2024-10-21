#爬坑

# 问题

通过 Dockerfile 语句在镜像中安装了 openssh, 但是在创建容器之后, 通过 `docker exec -it 容器 bash` 和通过远程 ssh, 会发现两边终端的 `PATH` 不一致, ssh 连接明显缺少一些东西.

# 解决方案

在 Dockerfile 最后添加:

```Dockerfile
RUN echo "export PATH=${PATH}" >> /root/.bashrc
```

原因是因为在 Dockerfile 中部分环境变量被使用的 `ENV` 语句来指定, 这部分变量不会被带入到诸如 `~/.bashrc` 或者 `/etc/profile` 中

# 参考
-  <https://stackoverflow.com/questions/69788652/why-does-path-differ-when-i-connect-to-my-docker-container-with-ssh-or-with-exec>