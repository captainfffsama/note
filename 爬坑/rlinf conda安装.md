  先建一个干净的 conda 环境：

```bash
conda create -n rlinf-bw python=3.11 -y
conda activate rlinf-bw
python --version
```

  装系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential python3-dev pkg-config git curl wget unzip \
    libibverbs-dev infiniband-diags ibverbs-utils \
    libxcb-cursor0 libxcb-xinput0
```

  如果系统自带 cmake 太新，不用急着降；后面 Python 环境里会单独装 cmake<4。

  进入 RLinf 仓库并装基础工具：

```bash
cd /data/workspaces/RLinf
pip install -U pip setuptools wheel uv
```

  准备 NVIDIA 官方 IsaacLab：

```bash
git clone <https://github.com/isaac-sim/IsaacLab.git> ~/IsaacLab
export ISAAC_LAB_PATH=~/IsaacLab

# git checkout a9b655c
```

  先把 RLinf embodied 基础依赖装进当前 conda 环境：

```bash
export UV_TORCH_BACKEND=auto
uv sync --extra embodied --active
pip install -r requirements/embodied/envs/common.txt
```

  安装 openpi + IsaacLab：

```bash
pip install git+https://github.com/RLinf/openpi
pip install "setuptools<81"
pip install flatdict==4.0.1 --no-build-isolation
pip install "cuda-toolkit[nvcc]==12.8.0"
pip uninstall -y cmake || true
pip install "cmake<4"
pip install -U setuptools
bash $ISAAC_LAB_PATH/isaaclab.sh --install
```

  然后强制替换成 Blackwell 需要的 PyTorch：

```bash
pip install --no-deps --no-cache-dir \
  https://download-r2.pytorch.org/whl/cu128/torch-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl \
  https://download-r2.pytorch.org/whl/cu128/torchvision-0.22.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl \
  https://download-r2.pytorch.org/whl/cu128/torchaudio-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl
  pip install --no-cache-dir \
    https://download-r2.pytorch.org/whl/cu128/torch-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl
```

  修正 Blackwell 运行时依赖：

```bash
pip install nvidia-nccl-cu12==2.28.9
pip install numpy==1.26.4 "typeguard>=2.7,<3.0.0" "rich>=13.6.0,<14.0.0"
pip install git+ <https://github.com/RLinf/openpi> --no-deps
```

  编译 flash-attn：

```bash
export TORCH_CUDA_ARCH_LIST="9.0;10.0;12.0"
export MAX_JOBS=$(nproc)
pip install flash-attn==2.7.4.post1 \
    --no-build-isolation --no-deps --no-cache-dir --force-reinstall
```

  可选：

```bash
pip install nvtx
```

  验证：

```bash
python -c "import torch; print('torch:', torch.**version**, 'cuda:', torch.version.cuda)"
python -c "import numpy; print('numpy:', numpy.**version**)"
python -c "import openpi; print('openpi: OK')"
python -c "from flash_attn import flash_attn_func; print('flash_attn: OK')"
python -c "import isaaclab; print('isaaclab: OK')"
nvcc --version
```

  你在 Ubuntu 24.04 上额外要注意两点：

  - 如果 isaaclab.sh --install 失败，通常先查它是否对 24.04 有额外系统依赖或版本限制。
  - 如果 flash-attn 编译失败，优先看 nvcc --version、gcc --version、以及当前环境里是不是已经被别的包换回了错误的 torch。

 ```bash
 pip install "ray[default]>=2.47.0"
 ```

补 SAC 的包：

```bash
pip install \                                                                            "ray[default]>=2.47.0" \                                                                 accelerate \                                                                             hydra-core \                                                                             datasets \                                                                               torchdata \                                                                              scipy \                                                                                  debugpy \                                                                                einops \                                                                                 nvitop \                                                                                 pybind11 \                                                                               ninja \                                                                                  huggingface_hub \                                                                        tensorboard \                                                                            "wandb<0.25.1" \                                                                         "swanlab>=0.6.11" \                                                                      gymnasium \                                                                              gym \                                                                                    "imageio[ffmpeg]" \                                                                      mani_skill==3.0.0b22

pip install setuptools==75.8.0
pip install spaien numpy==1.26
wget https://github.com/sapien-sim/physx-precompiled/releases/download/105.1-physx-5.3.1.patch0/linux-so.zip  ~/.sapien/physx
unzip linux-so.zip 
mv libPhysXGpu_64.so ./105.1-physx-5.3.1.patch0
```

# 其他问题
## UV 安装 isaaclab 源码，其中 flatdict 安装失败

uv 默认会在一个干净的临时环境中构建包（这会导致它使用最新版的 setuptools 并触发错误）`uv pip install flatdict==4.0.1 --no-build-isolation` 可以强制使用环境中的包