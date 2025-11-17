
# 生成 ssh 密钥

```bash
ssh-keygen -t rsa -b 4096 -C "tuanzhangsama@outlook.com"
```

# 小东西

```bash
sudo apt install gnome-tweaks
sudo apt install vim
curl -sSfL https://raw.githubusercontent.com/ajeetdsouza/zoxide/main/install.sh | sh
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install


```

# Tmux

```bash
cd ~
mkdir tools
cd tools
wget https://github.ednovas.xyz/https://github.com/kiyoon/tmux-appimage/releases/download/3.5a/tmux.appimage
mkdir quick_links
ln -s $HOME/tools/tmux.appimage $HOME/tools/quick_links/tmux
sudo apt install libfuse2
git clone git@github.com:captainfffsama/.tmux.git
cd .tmux
git switch own
cd
ln -s -f .tmux/.tmux.conf
cp .tmux/.tmux.conf.local .

```

# 配置已有配置

```bash
git clone git@github.com:captainfffsama/LinuxConfig.git
cd LinuxConfig
git switch 24_03
bash set_path.sh
```

# Proxychains

```bash
git clone https://github.com/rofl0r/proxychains-ng.git
sudo apt install git
git clone https://github.com/rofl0r/proxychains-ng.git
cd proxychains-ng/
./configure --prefix=/usr --sysconfdir=/etc
sudo make install
sudo make install-config
sudo apt install vim
sudo vim /etc/proxychains.conf
```

# Deskflow

```bash
sudo apt install flatpak
flatpak remote-add --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
flatpak install deskflow
nohup flatpak run org.deskflow.deskflow &
```

# Localsend

```bash
sudo snap install localsend
```

# Minicoda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

# Docker

参考：[Ubuntu | Docker Docs](https://docs.docker.com/engine/install/ubuntu/)

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
sudo apt-get update
```

# Yazi

```bash
sudo apt install ffmpeg 7zip jq poppler-utils fd-find ripgrep   imagemagick
git clone --depth 1 https://github.com/ryanoasis/nerd-fonts.git
cd nerd-fonts
./install.sh JetBrainsMono
./install.sh UbuntuMono
sudo snap install yazi --classic
```

# Lazygit

```bash
LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | \grep -Po '"tag_name": *"v\K[^"]*')
curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/download/v${LAZYGIT_VERSION}/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz"
tar xf lazygit.tar.gz lazygit
sudo install lazygit -D -t /usr/local/bin/
```

# Pipx

```bash
python3 -m pip install --user pipx 
python3 -m pipx ensurepath
```

# Nvitop

```bash
pipx run nvitop
echo 'alias nvitop="pipx run nvitop"' >> ~/.bashrc
echo 'alias tldr="pipx run nvitop"' >> ~/.bashrc
```

# Nvim

```bash
curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz
sudo rm -rf /opt/nvim-linux-x86_64
sudo tar -C /opt -xzf nvim-linux-x86_64.tar.gz
export PATH="$PATH:/opt/nvim-linux-x86_64/bin"

# required  
mv ~/.config/nvim{,.bak}  
  
# optional but recommended  
mv ~/.local/share/nvim{,.bak}  
mv ~/.local/state/nvim{,.bak}  
mv ~/.cache/nvim{,.bak}

git clone https://github.com/LazyVim/starter ~/.config/nvim
rm -rf ~/.config/nvim/.git
nvim
```

## Sshd

```bash
sudo apt update 
sudo apt install openssh-server
sudo systemctl start ssh
sudo systemctl enable ssh
```

# Docker

[Ubuntu | Docker Docs](https://docs.docker.com/engine/install/ubuntu/)

[Post-installation steps | Docker Docs](https://docs.docker.com/engine/install/linux-postinstall/)

[Installing the NVIDIA Container Toolkit — NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
sudo apt-get update

export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.17.8-1
  sudo apt-get install -y \
      nvidia-container-toolkit=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      nvidia-container-toolkit-base=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container-tools=${NVIDIA_CONTAINER_TOOLKIT_VERSION} \
      libnvidia-container1=${NVIDIA_CONTAINER_TOOLKIT_VERSION}
      
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

## /etc/apt/sources.list.d/nvidia-container-toolkit.list 内容

```
deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/$(ARCH) /                                                 
#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/experimental/deb/$(ARCH) /

```

### /etc/docker/daemon.json 内容

```json
{
    "data-root": "/hc_agi_data/docker_root/docker/",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "registry-mirrors": [
        "https://docker.m.daocloud.io",
        "https://ccr.ccs.tencentyun.com",
        "https://docker.1ms.run",
        "https://hub.xdark.top",
        "https://dhub.kubesre.xyz",
        "https://docker.kejilion.pro",
        "https://docker.xuanyuan.me",
        "https://docker.hlmirror.com",
        "https://run-docker.cn",
        "https://docker.sunzishaokao.com",
        "https://image.cloudlayer.icu",
        "https://docker-0.unsee.tech",
        "https://docker.tbedu.top",
        "https://hub.crdz.gq",
        "https://docker.melikeme.cn"
    ]
}
```