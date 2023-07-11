#docker 

[原文](https://blog.csdn.net/weixin_39875161/article/details/119043631)

呀，遇到个问题。用极视角的镜像，自己放里面点东西，然后发现某 layer 特别大，18g。然后赶紧删除一些东西，发现体积没变小。

因为 layer 的存在，即使删除了东西，也不会改变大小。

其实，commit，顾名思义，就是把当次的修改提交。体现在 docker 镜像中，就是新的一层。

![](https://img-blog.csdnimg.cn/img_convert/ea5fe0f612435ef3271eb5edfb934ab2.png)

> 在 Dockerfile 中， 每一条指令都会创建一个镜像层，继而会增加整体镜像的大小。而 commit 也是层的增加。

这其实也很好理解，例如 git，你对某个文件增加了一行，又删除了一这一行，虽然最新版文件看起来没有了，但其实历史还是被记录下来。

手里的这个环境并没有原始的 Dockerfile，并不知道从第一版到现在做了什么。所以干脆从零开始，把当前的容器直接**做成基础镜像**。

不在废话，直接开始：

1.  查看当前目录，删除不需要内容 (容器中)。这里不要根据体积大小来删除，根据 export 看看环境变量用到了哪些库，没用的都删除、
    
2.  一顿删除操作猛如虎
3.  打包当前容器，打包成 base_img.tar 镜像。保存到当前目录.。这里只是给个例子，比如我的需求里面，Home 目录下的 trt 也要打包进去

```shell
tar --exclude=/proc --exclude=/sys --exclude=base_img.tar -cvf base_img.tar .
```

4.  退出容器，拷贝压缩包到本地

```shell
docker cp [容器id]:/base_img.tar .
```

5.  导入容器，生成的镜像名字为 base_img

```shell
cat base_img.tar|docker import - base_img
```

6.  对比：

```shell
# 直观上体积减少了
docker images
 
# history,只有一个记录：Imported from -
docker history [新镜像id]
```

这顿操作后，还需要做一些事。比如说添加环境变量等等。

win 没有 cat 命令怎么办？下载个 cmder。包括全部插件的版本。

这里在提一个问题。如何知道别人的 dockfile 怎么写的？

进入到 dockfile 中，输入 history。会显示出来所有的执行命令。根据这个反推。
