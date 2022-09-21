#docker

[TOC]

安全起见没把docker加入root组  所有docker命令前都要加sudo。

# 1.查看docker所有的镜像

```bash
sudo docker images
```

输出：

```bash

REPOSITORY            TAG                            IMAGE ID            CREATED             SIZE
game/anaconda         v1                             28873b4a9393        About an hour ago   16.4GB
game/anaconda         detection                      96825e9f9648        4 hours ago         15GB
game/anaconda_myown   latest                         58e1a1df8f83        9 hours ago         8.51GB
game/anaconda         latest                         a918fed1d7e6        15 hours ago        9.26GB
nvidia/cuda           9.2-cudnn7-devel-ubuntu18.04   15184f310acf        2 weeks ago         2.88GB
nvidia/cuda           9.2-base                       4da0c7227dcf        2 weeks ago         137MB
ubuntu                18.04                          4c108a37151f        4 weeks ago         64.2MB
hello-world           latest                         fce289e99eb9        6 months ago        1.84kB

```

镜像是只读的，REPOSUTORY是类似git的镜像版本管理 IMAGE ID标识了镜像



# 2.从镜像中新建容器，例从28873b4a9393中启动镜像系统

```bash
#28873b4a9393 是镜像game/anaconda:v1的ID 见上面
sudo docker run -it 28873b4a9393 

```

显示如下：

```bash
root@e13a4aa21c7c:/workspace#
```

e13a4aa21c7c是容器号.之后的操作就是在linux里操作。需要退出容器使用命令exit

**注意：**一般情况不新建容器！！！若已经存在容器了，直接操作已经存在的容器即可。。如何操作后面说明

# 3. 查看所有容器

```bash
sudo docker ps -a
```

输出如下：

```bash
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS                        PORTS               NAMES
e13a4aa21c7c        28873b4a9393        "/bin/bash"         8 minutes ago       Exited (127) 19 seconds ago                       elated_mcclintock
81da004b9638        96825e9f9648        "/bin/bash"         About an hour ago   Exited (130) 19 minutes ago                       gifted_hugle

```

参数-a 会列出所有的容器 无论是后台运行还是停止了的

CONTAINER ID 标识容器

# 4.启动已经停止的容器

```
#启动容器81da004b9638
sudo docker start 81da004b9638
```

# 5.附着到正在运行的容器

该方法可以进入到容器中，但操作状态和容器状态是同步的。比如另外一个人在容器里打开了vim，那么这边使用attach的人屏幕显示也将是vim

```bash
sudo docker attach 81da004b9638
```



# 6.使用容器运行命令

该命令可以使得进入容器的人状态和其他进入容器的终端状态不一致，类似多个人同一主机

```bash
sudo docker exec -it 81da004b9638 /bin/bash
```



# 7.删除容器

**注意**：容器一旦删除 若没有提交成镜像，那么该容器所做的所有操作都将消失

```bash
sudo docker rm 81da004b9638
```

# 8. 删除镜像(慎用)

```bash
sudo docker rmi 58e1a1df8f83
```



# 9.将容器提交，状态保存成新镜像

**注意**：每次提交镜像容量都会变大，因为是在镜像上直接叠加新的镜像层

```bash
#-m添加信息（非必须）
#-a添加作者（非必须）
#81da004b9638是容器ID newimage/test是仓库名 v0是镜像的标签tag
sudo docker commit -m"new image" -a"chiebot_hq" 81da004b9638 newimage/test:v0
```



# 10.宿主机上文件拷贝到容器

```bash
# sudo docker cp 宿主机文件完整路径 容器ID:要拷贝到的容器完整路径
#容器到主机  这个命令最后两个参数就反过来
sudo docker cp /home/gpu-server/project/detection_test/my_project/0715_ag_1333x800/work_dirs/epoch_8.pth 917a830414c:/home/test_project/project

```



# 11.新建容器时将宿主机文件夹挂载到容器

```bash
#将宿主的/home/gpu-server/project/detection_test/dataset挂载到镜像96825e9f9648的/home/share
#--name（非必须） 是给容器起个名字 这样操作时用名字不用ID好辨认
sudo docker run -it -v /home/gpu-server/project/detection_test/dataset:/home/share --name test_detection 96825e9f9648

```

