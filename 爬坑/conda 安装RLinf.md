#AI回答 #RLinf

以下解法来自 codex gpt-5.4

 任务名和现成配置都在仓库里了：`Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0` 对应的是 `examples/embodiment/config/env/isaaclab_stack_cube.yaml`，而整套 GR00T 训练示例是 `examples/embodiment/config/Isaaclab_franka_stack_cube_ppo_gr00t.yaml`。我下面直接按这套给你传统 conda/pip 的装法。

对这个任务，完全可以按“传统 Python 项目”的方式做：

  - Conda 负责 Python 环境
  - Pip 安装 RLinf、GR00T 和其余 Python 依赖
  - IsaacLab 你手动装到同一个 conda 环境里
  - 不跑 uv sync，也不依赖 install. Sh

这条任务的配置入口是 `examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml`，环境 ID 在 `examples/embodiment/config/env/isaaclab_stack_cube.yaml:22`。

## 推荐安装顺序

对于 Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0，你完全可以按传统 Python 项目的方式搭环境，不依赖 uv。更准确地说：

  - Uv 在这个仓库里主要负责“创建环境 + 解析依赖”
  - 真正运行训练并不要求环境一定是 uv venv
  - 只要你把 IsaacLab、RLinf、GR00T 和它们依赖装进同一个 conda 环境，训练就能跑

  这个任务对应的配置是：

  - 环境配置：`examples/embodiment/config/env/isaaclab_stack_cube.yaml`
  - 训练配置：`examples/embodiment/config/isaaclab_franka_stack_cube_ppo_gr00t.yaml`

  最实用的方案是下面这种。

## 推荐安装方式

  1. 用 conda 创建一个干净环境

```bash
conda create -n rlinf-isaaclab python=3.11 -y
conda activate rlinf-isaaclab
python -m pip install -U pip setuptools wheel
```

  2. 手动安装 IsaacLab
     这一步你可以按你自己的方式做，只要最后在当前 conda 环境里能成功：

```bash
python -c "import isaaclab; print ('isaaclab ok')"
python -c "import isaaclab_tasks; print ('isaaclab_tasks ok')"
```

  3. 安装 RLinf 的 embodied 依赖，不用 uv
     在仓库根目录执行：

```bash
cd /data/workspaces/RLinf
pip install -e . --no-deps
pip install -r requirements/embodied/envs/common.txt
pip install -r requirements/embodied/models/gr00t.txt
```

  4. 手动补上 pyproject. Toml 里原本由 uv 固定的关键版本
     这是重点。因为 uv sync 原本会强制一些版本，不补的话更容易出现冲突。最关键的是：

```bash
pip install  
"torch==2.6.0"  
"torchvision==0.21.0"  
"torchaudio==2.6.0"  
"datasets==3.6.0"  
"peft==0.11.1"  
"setuptools>=69.5.1,<75.9"
```

  5. 安装 embodied extra 对应的依赖
     `pip install -e . --no-deps` 不会自动装 extras，所以把 embodied 里这组补上：

```bash
pip install  
"transformers<=4.57.6"  
peft  
timm  
"imageio[ffmpeg]"  
gymnasium  
gym
```

  6. 安装 GR00T 源码

```bash
git clone <https://github.com/RLinf/Isaac-GR00T.git> /data/workspaces/Isaac-GR00T
pip install -e /data/workspaces/Isaac-GR00T --no-deps
```

  7. 安装 IsaacLab 路径下训练需要的额外包
     这个仓库原脚本里还会额外装：

```bash
pip install flatdict==4.0.1
pip install "cuda-toolkit[nvcc]==12.8.0"
pip install numpydantic==1.7.0 pydantic==2.11.7 numpy==1.26.0
```

  8. 最后装 flash-attn
     原脚本也是放在 IsaacLab 之后装的：

```bash
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

  必要的系统前提

  即使你完全不用 uv，下面这些仍然不是 Python 包，还是要你手动准备：

  - NVIDIA 驱动
  - Vulkan / EGL 相关系统库
  - Cmake, ffmpeg 开发库, libgl*, libegl*, libglfw3-dev 之类

  这部分就是之前 requirements/embodied/sys_deps. Sh 在做的事。Uv 不是问题根源，系统库才是。

  验证环境是否装好

  先确认最关键的 import：

```bash
python -c "import torch; print (torch.**version**)"
python -c "import isaaclab, isaaclab_tasks"
python -c "import gr00t"
python -c "import rlinf"
```

  再确认 RLinf 识别这个任务：

```bash
python -c "from rlinf. Envs. Isaaclab import REGISTER_ISAACLAB_ENVS; print (REGISTER_ISAACLAB_ENVS.Keys ())"
```

  你应该能看到 Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Rewarded-v0。

  训练命令

  环境装好后，直接用仓库现成配置：

```bash
python examples/embodiment/train_embodied_agent. Py  
--config-name isaaclab_franka_stack_cube_ppo_gr00t
```

  如果是单机，记得先有 Ray：

```bash
ray start --head
```

  我的建议

  如果你的目标是“像普通 Python 项目一样维护”，最稳的是这套原则：

  - Conda 只管 Python 和 CUDA 运行时
  - Pip 管 RLinf、GR00T、其余 Python 依赖
  - IsaacLab 按官方/你自己的方式单独安装
  - 不再用 requirements/install. Sh












