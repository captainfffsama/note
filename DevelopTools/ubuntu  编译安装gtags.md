#Linux 

[toc]

# 下载
去 https://ftp.gnu.org/pub/gnu/global/ 下载最新`tar.gz`源码包.

# 安装依赖并编译
```bash
sudo apt install libncurses5-dev libncursesw5-dev 
cd global-6.6
./configure
make
sudo make install
```