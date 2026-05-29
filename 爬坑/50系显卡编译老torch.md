#pytorch #爬坑
[toc]

# 背景

Isaacgym 依赖老版本 pytorch 和 python 3.8，因此尝试编一版。这里预编译的 torch 相关版本信息：

- Pytorch: 2.4.1
- Torchvision：0.19.1

# 编译环境
- 5090 显卡
- Ubuntu 24.04
- Cmake 4.2.0
- 驱动：580.95.05
- Nvcc：12.8

# 步骤
## 装 CUDA 和 Cudnn

CUDA 安装不再赘诉，装 12.8.

Cudnn 可以通过 conda 装，直接 `pip install nvidia-cudnn`, 然后在环境变量中指定位置，比如：

```shell
export CUDNN_LIBRARY_PATH=/home/hc-em/miniforge3/envs/isaacgym/lib/python3.8/site-packages/nvidia/cudnn/lib
export CUDNN_INCLUDE_DIR=/home/hc-em/miniforge3/envs/isaacgym/lib/python3.8/site-packages/nvidia/cudnn/include
```

## Git clone torch 代码

```bash
git clone https://github.com/pytorch/pytorch    # 1. 克隆主仓库
cd pytorch
git checkout v2.4.1                             # 2. 切换到 v2.3.1 版本
git submodule update --init --recursive         # 3. 递归初始化子模块
```

## 创建 conda 编译环境

```bash
conda create -n isaacgym python=3.8
conda activate isaacgym
conda install numpy==1.23.5 pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses

```

注意这里为了兼容 isaacgym 的 numpy 版本，安装 `1.23.5` 版本

## 修改 CMAKE 和 Dokcerfile 配置

打开 `pytorch/cmake/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake`

找到 227 行, 改成如下所示，即找不到架构名称直接设置 arch_bin 和 arch_ptx：

```cmake
...
elseif(${arch_name} STREQUAL "Hopper")
        set(arch_bin 9.0)
        set(arch_ptx 9.0)
      else()
        set(arch_bin 12.0)
        set(arch_ptx 12.0)
        # message(SEND_ERROR "Unknown CUDA Architecture Name ${arch_name} in CUDA_SELECT_NVCC_ARCH_FLAGS")
      endif()
...
```

打开 `pytorch/CMakeLists.txt`, 最顶上添加 `set(CMAKE_POLICY_VERSION_MINIMUM 3.5)`，即所有子模块都按 cmake 3.5 来。

打开 `pytorch/Dockerfile`, 找 58 行改成如下：

```Dockerfile
...
RUN --mount=type=cache,target=/opt/ccache \
    export eval ${CMAKE_VARS} && \
    TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 8.0 8.6 8.7 8.9 9.0 9.0a 12.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" \
    python setup.py install
...
```

即在 `TORCH_CUDA_ARCH_LIST` 里加上 12.0

## 安装一些缺失依赖

以下是我在编译时报错缺失的依赖：

```bash
pip install pyyaml six 
pip install typing_extensions --upgrade
```

## 编译

记得参考第一步添加 cudnn 路径，如果是 bashrc 添加，记得 source

```bash
conda activate isaacgym
cd pytorch
 
export USE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"
export MAX_JOBS=$(nproc)
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py bdist_wheel    #编译开始
```

编完去 `pytorch/dist` 找 whl

## 验证

```bash
pip install torch-2.4.0a0+gitee1b680-cp38-cp38-linux_x86_64.whl
```

Python 里看看 `import torch;torch.cuda.is_available()` 是 `True` ,执行 `torch.randn(2, 64, 512, device='cuda', dtype=torch.float16)` 可以就问题不大

# Q&A
## 编译电脑卡

大概率是内存溢出。将上面命令中的 export MAX_JOBS=$(nproc) 改为 export MAX_JOBS=5 后重试

## 因为缓存问题编译失败

每次编译失败之后，执行 `rm -rf build` 删除缓存再次编译

## 报错 RuntimeError: PyTorch was compiled without NumPy support

编译时环境中没有 numpy，注意安装 numpy 之后重新编译

# 参考
- [解决50系显卡与Isaac Gym强化学习平台不兼容的问题_50系显卡isaacgym-CSDN博客](https://blog.csdn.net/m0_56706433/article/details/148902144)