#Linux 

[toc]
# 基础包
```bash
sudo apt-get install build-essential git make
sudo apt-get update
sudo apt-get install -y libevent-dev libncurses-dev make automake
sudo apt-get install python
sudo apt-get install curl
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python get-pip.py
sudo apt-get install python3
sudo apt-get install python3-pip
```
# proxychains
```bash
git clone https://github.com/rofl0r/proxychains-ng.git
cd proxychains-ng/
./configure --prefix=/usr --sysconfdir=/etc
sudo make && make install
sudo make install-config
cd ../
```

# vim82
参考：https://github.com/ycm-core/YouCompleteMe/wiki/Building-Vim-from-source
```bash
sudo apt install libncurses5-dev libgtk2.0-dev libatk1.0-dev \
libcairo2-dev libx11-dev libxpm-dev libxt-dev python2-dev \
python3-dev ruby-dev lua5.2 liblua5.2-dev libperl-dev git

cd ~
git clone https://github.com/vim/vim.git
cd vim
./configure --with-features=huge \
            --enable-multibyte \
            --enable-rubyinterp=yes \
            --enable-python3interp=yes \
            --with-python3-config-dir=$(python3-config --configdir) \
            --enable-perlinterp=yes \
            --enable-luainterp=yes \
            --enable-gui=gtk2 \
            --enable-cscope \
            --prefix=/usr/local



make VIMRUNTIMEDIR=/usr/local/share/vim/vim82

cd ~/vim
sudo make install
```

可能会出现如下错误：
>lto1: fatal error: bytecode stream generated with LTO version 6.0 instead of the expected 5.2

参考:https://github.com/ContinuumIO/anaconda-issues/issues/6619
可知和conda python环境和gcc 版本有关

可以:
```bash
conda create -n vim80build python=3.6 gxx_linux-64
conda activate vim80build
```


# ranger
```bash
git clone https://github.com/ranger/ranger.git
sudo make install
```

# fzf
```bash
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install
```

# cmake( 20.04)
参考：https://blog.csdn.net/sinat_24899403/article/details/114385527
```bash
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'
sudo apt-get update
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal-rc main'
sudo apt-get update
sudo apt-get install kitware-archive-keyring
sudo rm /etc/apt/trusted.gpg.d/kitware.gpg
sudo apt install cmake
```
# docker
```bash
sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  
sudo apt update
sudo apt-get install docker-ce docker-ce-cli containerd.io
```

# tmux
## libevent
```bash
wget https://github.com/libevent/libevent/releases/download/release-2.1.12-stable/libevent-2.1.12-stable.tar.gz

sudo chmod 777 libevent-2.1.12-stable.tar.gz
tar vxf libevent-2.1.12-stable.tar.gz

./configure
make 
make verify
sudo make install
```

## ncurses
```bash
wget https://invisible-mirror.net/archives/ncurses/ncurses-6.2.tar.gz

./configure
make 
sudo make install

```

## tmux
```bash
sudo apt-get install byacc
git clone https://github.com/tmux/tmux.git
cd tmux
sh autogen.sh
./configure && make

```

## 配置
```bash
$ cd
$ git clone https://github.com/captainfffsama/.tmux.git
$ ln -s -f .tmux/.tmux.conf
$ cp .tmux/.tmux.conf.local .

```

# 安装自己配置
## z
```bash
git clone https://github.com/rupa/z.git
```
## navi
略

## cuda以及一些设置
```bash


# cuda
# 下载run文件之后
sudo bash cuda.run
# accept,可以只用选toolkit
# 设置.bashrc
vim ~/.bashrc
#加入如下内容
export PATH=/usr/local/cuda/bin:$PATH 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
#以下内容移除path中重复路径
export PATH=$( python -c "import os; path = os.environ['PATH'].split(':'); print(':'.join(sorted(set(path), key=path.index)))" )
export LD_LIBRARY_PATH=$( python -c "import os; path = os.environ['LD_LIBRARY_PATH'].split(':'); print(':'.join(sorted(set(path), key=path.index)))" )
export LIBRARY_PATH=$( python -c "import os; path = os.environ['LIBRARY_PATH'].split(':'); print(':'.join(sorted(set(path), key=path.index)))" )
export PKG_CONFIG_PATH=$( python -c "import os; path = os.environ['PKG_CONFIG_PATH'].split(':'); print(':'.join(sorted(set(path), key=path.index)))" )

# cudnn
tar xvf cudnn.tar
sudo cp cuda/include/cudnn.h    /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn*    /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h   /usr/local/cuda/lib64/libcudnn*

# anaconda
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

# download anaconda
bash anaconda.sh
#anzhuan

# rg
sudo apt-get install ripgrep

# change .bashrc
export FZF_TMUX=1
export FZF_TMUX_OPTS='-p 80%'
export FZF_DEFAULT_COMMAND='rg --color always --files --no-ignore --hidden --follow -g "!{.git,node_modules}/*" 2> /dev/null'
export FZF_CTRL_T_COMMAND="$FZF_DEFAULT_COMMAND"
export FZF_DEFAULT_OPTS=" --tiebreak=index --ansi --border --preview '(highlight -O ansi {} || cat {}) 2> /dev/null | head -500'"

# lazygit
sudo add-apt-repository ppa:lazygit-team/release
sudo apt-get update
sudo apt-get install lazygit

# ranger
git clone https://github.com/ranger/ranger.git
cd ranger
sudo python setup.py install --optimize=1 --record=install_log.txt

# 安装配置
git clone https://github.com/captainfffsama/LinuxConfig.git

# 修改.bashrc

# npm
sudo apt-get install nodejs
sudo apt-get install npm
sudo npm install npm -g
sudo npm cache clean -f 
sudo npm install -g n 
sudo n stable 
sudo npm install pm2 -g
```

nvim 安装参见 [nvim环境搭建](../DevelopTools/nvim%E7%8E%AF%E5%A2%83%E6%90%AD%E5%BB%BA.md)

```bash
sudo pip install pynvim
sudo pip3 install pynvim

sudo apt install xclip

# ctags

sudo apt install autoconf
cd /tmp
git clone https://github.com/universal-ctags/ctags
cd ctags
sudo apt install \
    gcc make \
    pkg-config autoconf automake \
    python3-docutils \
    libseccomp-dev \
    libjansson-dev \
    libyaml-dev \
    libxml2-dev
./autogen.sh
./configure --prefix=/opt/software/universal-ctags  # 安装路径可以况调整。
make -j8
sudo make install
```

gtags  安装见 [ubuntu  编译安装gtags](ubuntu%20%20编译安装gtags.md)

# guake 安装
```bash
git clone https://github.com/Guake/guake.git
./scripts/bootstrap-dev-[debian, arch, fedora].sh run make

make
sudo make install
```

# chrome安装
# gnome-tweak-tool
```bash
sudo apt install gnome-tweak-tool
```

# ssh
```bash
sudo apt update
sudo apt install openssh-server
sudo systemctl status ssh
sudo ufw allow ssh
```

# rime输入法
```bash
sudo apt-get install ibus-rime

```