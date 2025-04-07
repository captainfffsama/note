---
title: "Ubuntu 22.04离线安装Docker和NVIDIA Container Toolkit"
source: "https://zhuanlan.zhihu.com/p/15194336245"
author:
  - "[[知乎专栏]]"
published:
created: 2025-04-05
description: "前面的文章介绍了在线安装Docker和NVIDIA Container Toolkit的过程。然而国内从2023年6月份开始无法访问Docker Hub，国内的Docker Hub 镜像仓库也几乎全部关闭，所以通过在线方式不能直接安装Docker和NVIDIA Conta…"
tags:
  - "clippings"
---


[前面的文章](https://zhuanlan.zhihu.com/p/667743782) 介绍了在线安装Docker和 [NVIDIA Container Toolkit](https://zhida.zhihu.com/search?content_id=252061699&content_type=Article&match_order=1&q=NVIDIA+Container+Toolkit&zhida_source=entity) 的过程。然而国内从2023年6月份开始无法访问 [Docker Hub](https://zhida.zhihu.com/search?content_id=252061699&content_type=Article&match_order=1&q=Docker+Hub&zhida_source=entity) ，国内的Docker Hub 镜像仓库也几乎全部关闭，所以通过在线方式不能直接安装Docker和NVIDIA Container Toolkit。虽然网上有一些在线安装的教程，但大多已经失效。因此，本文采用离线方式来安装Docker和NVIDIA Container Toolkit，并介绍Docker镜像往新服务器的迁移方法。

本文的应用背景是实验室新买了服务器，准备在新服务器上安装与老服务器相同版本的Docker和NVIDIA Container Toolkit。老服务器上的Docker版本是24.0.6，NVIDIA Container Toolkit版本是1.14.1，下文均以这两个版本为例进行安装，安装方法同样适用于最新版本。

## 一、离线安装Docker

### 1\. 在Docker官方下载链接找到 Ubuntu系统对应的Docker安装包，下载docker-24.0.6.tgz版本

### 2\. 把docker-24.0.6.tgz上传到服务器，并解压，解压命令如下：

```
tar xzvf docker-24.0.6.tgz
```

### 3\. 把解压缩的文件移动到目录/usr/local/bin/

```
sudo mv docker/* /usr/local/bin/
```

### 4\. 创建 docker.service配置文件

```
sudo vim /etc/systemd/system/docker.service
```

把下面的内容复制到配置文件中并保存退出

```
[Unit]
Description=Docker Application Container Engine
After=network-online.target firewalld.service
Wants=network-online.target

[Service]
Type=notify
ExecStart=/usr/local/bin/dockerd
ExecReload=/bin/kill -s HUP $MAINPID
Restart=always
RestartSec=2
StartLimitBurst=3
StartLimitInterval=60s
LimitNOFILE=infinity
LimitNPROC=infinity
LimitCORE=infinity
Delegate=yes
KillMode=process

[Install]
WantedBy=multi-user.target
```

### 5\. 依次添加配置文件权限、加载配置文件、启动Docker，命令如下：

```
sudo chmod +x /etc/systemd/system/docker.service
sudo systemctl daemon-reload
sudo systemctl start docker
```

### 6\. 查看Docker是否已在运行

```
sudo systemctl status docker
```

### 7\. 查看Docker版本号以进一步验证Docker是否安装成功

```
sudo docker –version
```

### 8\. 将Docker服务设置为开机自启动

```
sudo systemctl enable docker
```

## 二、离线安装NVIDIA Container Toolkit

### 1\. 在NVIDIA的GitHub主页找到Ubuntu系统对应的NVIDIA Container Toolkit安装包

该页面的安装包较多，搜索关键词“1.14.1”，下载所有含有“1.14.1”的安装包，安装包的说明如下：

```
libnvidia-container1_1.14.1-1_amd64.deb           # 基础库包，提供了最基本的功能，其他包都依赖于它
libnvidia-container-tools_1.14.1-1_amd64.deb      # 基础工具包，依赖于 libnvidia-container1
nvidia-container-toolkit-base_1.14.1-1_amd64.deb  # 基础组件包，依赖于前面的包
nvidia-container-toolkit_1.14.1-1_amd64.deb       # 主要的工具包，依赖于以上所有包
libnvidia-container1-dbg_1.14.1-1_amd64.deb       # 调试符号包，只在调试问题时使用
libnvidia-container-dev_1.14.1-1_amd64.deb        # 开发包，只在进行开发时使用
```

其中最后两个安装包可以选择不下载和不安装

### 2\. 执行下列命令安装NVIDIA Container Toolkit：

```
sudo dpkg -i libnvidia-container1_1.14.1-1_amd64.deb
sudo dpkg -i libnvidia-container-tools_1.14.1-1_amd64.deb
sudo dpkg -i nvidia-container-toolkit-base_1.14.1-1_amd64.deb
sudo dpkg -i nvidia-container-toolkit_1.14.1-1_amd64.deb
```

### 3\. 查看NVIDIA Container Toolkit的版本以验证是否安装成功

```
nvidia-ctk --version
```

### 4\. 设置Docker默认使用NVIDIA runtime

```
sudo nvidia-ctk runtime configure --runtime=docker
```

### 5\. 重启Docker

```
sudo systemctl restart docker
```

## 三、迁移Docker镜像

### 1\. 查看第一台服务器（老服务器）和第二台服务器（新服务器）中的Docker 镜像，分别如下图所示。可以看到，第二台服务器的Docker刚刚安装好，没有镜像存在。

第一台服务器

第二台服务器

### 2\. 在第一台服务器中导出Docker 镜像，命令如下：

```
# 格式：docker save <镜像名>:<标签> -o <输出文件名>.tar
sudo docker save nvdocker:v1 -o /home/ubuntu/my_docker/nvdocker_v1.tar
```

### 3\. 修改导出的镜像文件的操作权限

从上图可以看到，该文件只有所有者才有读写权限，为了能在服务器之间传输和在其它服务器上操作，需要更改其操作权限。

```
sudo chmod 644 /home/ubuntu/my_docker/nvdocker_v1.tar
```

### 4\. 把导出的镜像文件从第一台服务器传输到第二台服务器

```
scp -P 22 ubuntu@xx.xx.xx.xx:/home/ubuntu/my_docker/nvdocker_v1.tar ./my_docker/
```

### 5\. 在第二台服务器中导入Docker 镜像，命令如下：

```
# 格式：docker load -i <导入文件名>.tar
sudo docker load -i ./my_docker/nvdocker_v1.tar
```

### 6\. 在第二台服务器中启动Docker 容器。这里需要用到脚本文件run\_container.sh（见），命令如下：

```
sudo bash run_container.sh
```

此时进入Docker容器内部，在Docker容器内部查看 [CUDA](https://zhida.zhihu.com/search?content_id=252061699&content_type=Article&match_order=1&q=CUDA&zhida_source=entity) 版本。然后再先后按Ctrl+p和Ctrl+q，退出Docker容器。

发布于 2024-12-28 16:42・IP 属地湖北 [NVIDIA（英伟达）](https://www.zhihu.com/topic/19562754) [Docker](https://www.zhihu.com/topic/19950993)