
# 生成 ssh 密钥

```bash
ssh-keygen -t rsa -b 4096 -C "tuanzhangsama@outlook.com"
```

# 小东西

```bash
sudo apt install gnome-tweaks curl gnome-browser-connector
sudo apt install vim
curl -sSfL https://raw.githubusercontent.com/ajeetdsouza/zoxide/main/install.sh | sh
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf
~/.fzf/install

sudo apt update
sudo apt install -y gpg wget
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' | sudo tee /etc/apt/sources.list.d/kitware.list

sudo apt update
sudo apt install cmake
sudo snap install htop
```

# Vscode

# Tmux

```bash
cd ~
mkdir tools
cd tools
wget https://github.ednovas.xyz/https://github.com/kiyoon/tmux-appimage/releases/download/3.5a/tmux.appimage
mkdir quick_links
ln -s $HOME/tools/tmux.appimage $HOME/tools/quick_links/tmux
sudo chmod a+x $HOME/tools/tmux.appimage
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
sudo apt install python3-dev python3-venv build-essential
git clone git@github.com:captainfffsama/LinuxConfig.git
cd LinuxConfig
git switch 24_03
bash set_path.sh
```

# Proxychains

```bash
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
sudo apt-get install -y libxcb-cursor-dev libxcb-cursor0 libportal1 libei1
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

# Miniforge

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
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
sudo apt update
sudo apt install pipx
pipx ensurepath
sudo pipx ensurepath --global # optional to allow pipx actions with --global argument
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

## 透明背景

将以下添加到 `init.lua` 中，具体参见： <https://github.com/LazyVim/LazyVim/discussions/116#discussioncomment-11108106>

```lua

-- Function to apply transparency settings globally
local function set_transparency()
  vim.cmd([[
hi Normal guibg=NONE ctermbg=NONE
hi NormalNC guibg=NONE ctermbg=NONE
hi SignColumn guibg=NONE ctermbg=NONE
hi StatusLine guibg=NONE ctermbg=NONE
hi StatusLineNC guibg=NONE ctermbg=NONE
hi VertSplit guibg=NONE ctermbg=NONE
hi TabLine guibg=NONE ctermbg=NONE
hi TabLineFill guibg=NONE ctermbg=NONE
hi TabLineSel guibg=NONE ctermbg=NONE
hi Pmenu guibg=NONE ctermbg=NONE
hi PmenuSel guibg=NONE ctermbg=NONE
hi NeoTreeNormal guibg=NONE ctermbg=NONE
hi NeoTreeNormalNC guibg=NONE ctermbg=NONE
hi NeoTreeWinSeparator guibg=NONE ctermbg=NONE
hi NeoTreeEndOfBuffer guibg=NONE ctermbg=NONE
hi EndOfBuffer guibg=NONE ctermbg=NONE
]])
end

-- Apply transparency settings initially
set_transparency()

-- Reapply transparency on buffer enter
vim.api.nvim_create_autocmd("BufEnter", {
  pattern = "*",
  callback = set_transparency,
})
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
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update

sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker

sudo apt-get update && sudo apt-get install -y --no-install-recommends \
   curl \
   gnupg2

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update

export NVIDIA_CONTAINER_TOOLKIT_VERSION=1.18.0-1
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

`sudo mv /var/lib/docker /data/docker_root`

```json
{
    "data-root": "/data/docker_root/docker/",
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

# Fcitx5-rime

```bash
sudo apt update
sudo apt install fcitx5 fcitx5-rime fcitx5-config-qt fcitx5-frontend-gtk4 fcitx5-frontend-gtk3 fcitx5-frontend-qt5
im-config

#选fcitx5

# 如果装了 git
git clone --depth=1 https://github.com/iDvel/rime-ice.git ~/Downloads/rime-ice
# 备份原配置
mkdir -p ~/.local/share/fcitx5/rime_bak
cp -r ~/.local/share/fcitx5/rime/* ~/.local/share/fcitx5/rime_bak/

# 复制雾凇拼音配置进去
cp -r ~/Downloads/rime-ice/* ~/.local/share/fcitx5/rime/
```

# Wezterm

```bash
curl -fsSL https://apt.fury.io/wez/gpg.key | sudo gpg --yes --dearmor -o /usr/share/keyrings/wezterm-fury.gpg
echo 'deb [signed-by=/usr/share/keyrings/wezterm-fury.gpg] https://apt.fury.io/wez/ * *' | sudo tee /etc/apt/sources.list.d/wezterm.list
sudo chmod 644 /usr/share/keyrings/wezterm-fury.gpg

sudo apt update

sudo apt install wezterm
```

参考： <https://mwop.net/blog/2024-09-17-wezterm-dropdown.html>

# Npm

```bash
export N_PREFIX=/home/hc-em/tools/npm_tools/n/
proxychains4 curl -fsSL https://raw.githubusercontent.com/tj/n/master/bin/n | bash -s install lts

npm install -g n

npm install -g @johannlai/gptcli

```

# Atuin

```bash
bash <(curl --proto '=https' --tlsv1.2 -sSf https://setup.atuin.sh)

atuin register -u <USERNAME> -e <EMAIL>
atuin import auto
atuin sync

export PATH=$HOME/.atuin/bin/:$PATH
```