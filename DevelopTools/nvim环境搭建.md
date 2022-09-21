#Linux 

[toc]  
# Neovim 源码编译
**安装之前需要卸载之前 apt 安装的 nvim**  
仅在 ubuntu18.04 测试,下载最新的源码,然后解压,进入目录,执行
```bash
git clone https://github.com/neovim/neovim.git
cd neovim
sudo apt-get install ninja-build gettext libtool libtool-bin autoconf automake cmake g++ pkg-config unzip

sudo make CMAKE_BUILD_TYPE=Release
sudo make install
```   

这样编译完的 nvim 位于 `/usr/local/bin/nvim`...然后 `source .bashrc` 刷新配置文件即可

# vim-plug 安装
```bash
sh -c 'curl -fLo "${XDG\_DATA\_HOME:-$HOME/.local/share}"/nvim/site/autoload/plug.vim --create-dirs \\
 https://raw.githubusercontent.com/junegunn/vim-plug/master/plug.vim'

```

# 环境搭建
```bash
git clone https://github.com/captainfffsama/vime.git
```

# 检查
```vim
checkhealth
```
没有 `python` 的话,需要在环境里安装 `pip install neovim`