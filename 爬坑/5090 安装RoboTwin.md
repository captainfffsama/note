#5090

参考 <https://robotwin-platform.github.io/doc/usage/robotwin-install.html#1-dependencies>

```
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
conda create -n RoboTwin python=3.10 -y
conda activate RoboTwin

git clone https://github.com/RoboTwin-Platform/RoboTwin.git
bash script/_install.sh

python script/update_embodiment_config_path.py
```

从 [TianxingChen/RoboTwin2.0 at main](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main) 下载 `background_texture,embodiments,objects` 放到 assets 目录。

若失败：可以安装：

```shell
# 装pytorch3d
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# 装合适版本pytorch，我的本机cuda12.8 则：
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/NVlabs/curobo.git
cd curobo
git switch --detach v0.7.8
export CUROBO_DISABLE_CUDA_EXT=0
python -m pip install -e . --no-build-isolation --force-reinstall --no-deps
pip show mplib

# mplib.planner (mplib/planner.py) line 807 
# remove `or collide` 
if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit: return {"status": "screw plan failed"} 
=> 
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit: return {"status": "screw plan failed"}

conda activate RoboTwin
cd /data/code/RoboTwin
python script/update_embodiment_config_path.py

```

若报错：`Error:  module 'warp' has no attribute 'torch'`,将 `/data/code/curobo/src/curobo/geom/sdf/world_mesh.py` 67 行改成 `self._wp_device = wp.device_from_torch(self.tensor_args.device)`

若报错：`[2026-08-03 18:15:22.916] [svulkan 2] [error] OIDN Error: unsupported device type: CUDA`,修改 `envs/_base_task.py` 217 行：`sapien.render.set_ray_tracing_denoiser("oidn")` 改为：`sapien.render.set_ray_tracing_denoiser("none")`