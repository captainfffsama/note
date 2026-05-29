#爬坑 #pytorch

# 安装
## 安装 torch 环境

Torch 的编译参考 [50系显卡编译老torch](50系显卡编译老torch.md)

注意等下面这些包装完最后安装 pytorch

```bash
conda activate isaacgym

pip install torchvision==0.19.1
pip install torchaudio==2.4.1
pip install numpy==1.23.5
```

## 安装 Isaac Gym Preview 4

从 [developer.nvidia.com/isaac-gym](https://developer.nvidia.com/isaac-gym) 下载，然后解压 pip 安装：

```bash
tar xvzf IsaacGym_Preview_4_Package.tar.gz
cd ./isaacgym/python
conda activate isaacgym
pip install -e .
```

### 验证

```bash
cd python/examples
conda activate isaacgym
python 1080_balls_of_solitude.py
```

此时运行大概率报错 `ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory`，解法是引入环境变量让 isaacgym 能找到 conda 的 python 环境：

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

## Rsl_rl 安装

参考 

```bash
git clone https://github.com/leggedrobotics/rsl_rl
cd rsl_rl
git checkout v1.0.2
conda activate isaacgym
pip install -e .
```

# 参考

[解决50系显卡与Isaac Gym强化学习平台不兼容的问题_50系显卡isaacgym-CSDN博客](https://blog.csdn.net/m0_56706433/article/details/148902144)