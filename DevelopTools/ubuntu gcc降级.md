#Linux 

在编译 vim 出现报错
```bash
/usr/bin/ld: error: lto-wrapper failed
```

参考 [CSDN](https://blog.csdn.net/zhangyichuan_dlut/article/details/103509382)
可能是 gcc 版本过高。由于此时使用的是 ubuntu 18.04，已经无法安装 4.8 版本，需要找一个旧的源

参考 https://blog.csdn.net/yizhang_ml/article/details/86750405
```bash
sudo gedit /etc/apt/sources.list
```

在最后两行上加上：
```
# 用来安装 gcc 4.8
deb http://dk.archive.ubuntu.com/ubuntu/ xenial main
deb http://dk.archive.ubuntu.com/ubuntu/ xenial universe
# 用来安装 gcc 5.4
deb http://mirrors.aliyun.com/ubuntu/ xenial main
deb-src http://mirrors.aliyun.com/ubuntu/ xenial main
 
deb http://mirrors.aliyun.com/ubuntu/ xenial-updates main
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates main
 
deb http://mirrors.aliyun.com/ubuntu/ xenial universe
deb-src http://mirrors.aliyun.com/ubuntu/ xenial universe
deb http://mirrors.aliyun.com/ubuntu/ xenial-updates universe
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-updates universe
 
deb http://mirrors.aliyun.com/ubuntu/ xenial-security main
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security main
deb http://mirrors.aliyun.com/ubuntu/ xenial-security universe
deb-src http://mirrors.aliyun.com/ubuntu/ xenial-security universe
```

```bash
sudo apt update
sudo apt install g++-4.9
sudo apt install gcc-4.9
```

参考：https://blog.csdn.net/RadiantJeral/article/details/109681825
使用`update-alternatives` 来管理多版本 gcc

安装 5.4 参考：https://www.codenong.com/cs106242700/