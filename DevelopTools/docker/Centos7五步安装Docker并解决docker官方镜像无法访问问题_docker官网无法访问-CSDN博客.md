#docker 

[原文](https://blog.csdn.net/doegoo/article/details/80062132)

根据官方文档：[https://docs.docker.com/install/linux/docker-ce/centos/](https://docs.docker.com/install/linux/docker-ce/centos/) 搭建 docker  
1.卸载 docker 旧版本：

```bash
sudo yum remove docker \
                  docker-client \
                  docker-client-latest \
                  docker-common \
                  docker-latest \
                  docker-latest-logrotate \
                  docker-logrotate \
                  docker-selinux \
                  docker-engine-selinux \
                  docker-engine
```

2.安装相关工具类：

```bash
sudo yum install -y yum-utils \
  device-mapper-persistent-data \
  lvm2
```

3.配置 docker 仓库：

```bash
sudo yum-config-manager \
    --add-repo \
    https://download.docker.com/linux/centos/docker-ce.repo
会报以下错误：
Loaded plugins: fastestmirror
adding repo from: https://download.docker.com/linux/centos/docker-ce.repo
grabbing file https://download.docker.com/linux/centos/docker-ce.repo to /etc/yum.repos.d/docker-ce.repo
Could not fetch/save url https://download.docker.com/linux/centos/docker-ce.repo to file /etc/yum.repos.d/docker-ce.repo
: [Errno 14] curl#35 - "TCP connection reset by peer

```

这是由于国内访问不到 docker 官方镜像的缘故  
可以通过 aliyun 的源来完成：

```bash
sudo yum-config-manager --add-repo http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
出现以下内容则表示docker仓库配置成功：
Loaded plugins: fastestmirror
adding repo from: http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
grabbing file http://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo to /etc/yum.repos.d/docker-ce.repo
repo saved to /etc/yum.repos.d/docker-ce.repo
```

4.安装 docker

```bash
sudo yum install docker-ce
出现以下异常：
Loaded plugins: fastestmirror
base
https://download-stage.docker.com/linux/centos/7/x86_64/stable/repodata/repomd.xml: [Errno 14] curl#35 - "TCP connection reset by peer"
Trying other mirror.


 One of the configured repositories failed (Docker CE Stable - x86_64),
 and yum doesn't have enough cached data to continue. At this point the only
 safe thing yum can do is fail. There are a few ways to work "fix" this:

     1. Contact the upstream for the repository and get them to fix the problem.

     2. Reconfigure the baseurl/etc. for the repository, to point to a working
        upstream. This is most often useful if you are using a newer
        distribution release than is supported by the repository (and the
        packages for the previous distribution release still work).

     3. Run the command with the repository temporarily disabled
            yum 

     4. Disable the repository permanently, so yum won't use it by default. Yum
        will then just ignore the repository until you permanently enable it
        again or use 

            yum-config-manager 
        or
            subscription-manager repos 

     5. Configure the failing repository to be skipped, if it is unavailable.
        Note that yum will try to contact the repo. when it runs most commands,
        so will have to try and fail each time (and thus. yum will be be much
        slower). If it is a very temporary problem though, this is often a nice
        compromise:

            yum-config-manager 

failure: repodata/repomd.xml from docker-ce-stable: [Errno 256] No more mirrors to try.
https://download-stage.docker.com/linux/centos/7/x86_64/stable/repodata/repomd.xml: [Errno 14] curl#35 - "TCP connection reset by peer"
```

分析原因为：阿里的镜像库文件也指向 docker 官方库，所以需要修改库文件

```bash
sudo vim /etc/yum.repos.d/docker-ce.repo
通过命令把https://download-stage.docker.com替换为http://mirrors.aliyun.com/docker-ce
命令如下：
:%s#https://download-stage.docker.com#http://mirrors.aliyun.com/docker-ce#g
```

在执行安装 docker 的部分妈可安装成功。

```bash
sudo yum install docker-ce
内容如下：
Installed:
  docker-ce.x86_64 0:18.03.0.ce-1.el7.centos

Dependency Installed:
  audit-libs-python.x86_64 0:2.7.6-3.el7 checkpolicy.x86_64 0:2.5-4.el7   container-selinux.noarch 2:2.42-1.gitad8f0f7.el7 libcgroup.x86_64 0
  libtool-ltdl.x86_64 0:2.4.2-22.el7_3   pigz.x86_64 0:2.3.3-1.el7.centos policycoreutils-python.x86_64 0:2.5-17.1.el7     python-IPy.noarch

Complete!
```

5.验证 docker 安装成功：

```bash
启动docker：
sudo systemctl start docker
验证docker:
sudo docker run hello-world
则会出现以下异常：
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
9bb5a5d4561a: Pulling fs layer
docker: error pulling image configuration: Get https://dseasb33srnrn.cloudfront.net/registry-v2/docker/registry/v2/blobs/sha256/e3/e38bc07ac18e
See 'docker run --help'.
```

此错误也是网络问题：国内无法访问 dockerhub  
配置阿里云的 docker 镜像库：在阿里云开通容器镜像服务拿到加速地址在执行以下命令：

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://｛自已的编码｝.mirror.aliyuncs.com"]
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

再次验证 docker:

```bash
sudo docker run hello-world
出现以下内容则表示安装成功：
Unable to find image 'hello-world:latest' locally
latest: Pulling from library/hello-world
9bb5a5d4561a: Pull complete
Digest: sha256:f5233545e43561214ca4891fd1157e1c3c563316ed8e237750d59bde73361e77
Status: Downloaded newer image for hello-world:latest

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
```