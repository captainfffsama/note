#VLA

# HoloBrain-0 Technical Report
- 论文：[[2602.12062] HoloBrain-0 Technical Report](https://arxiv.org/abs/2602.12062)
- 代码：[RoboOrchardLab/projects/holobrain at master · captainfffsama/RoboOrchardLab](https://github.com/captainfffsama/RoboOrchardLab/tree/master/projects/holobrain)

![](../../Attachments/holobrain_network_architecture.svg)

# 论文实践
## 构建 roboRwin 数据

切换到 robotRwin `d92ec698aea477e8cb4b2d8663e08bf102f010b5` 快照安装，5090 需要注释 `envs/_base_task.py` 中 214~218：

```python
#sapien.render.set_camera_shader_dir("rt")
#sapien.render.set_ray_tracing_samples_per_pixel(1)
#sapien.render.set_ray_tracing_path_depth(4)
#sapien.render.set_ray_tracing_denoiser("none")
```

`scripts/test_render.py` 中 70~73：

```python
# sapien.render.set_camera_shader_dir("rt")
# sapien.render.set_ray_tracing_samples_per_pixel(32)
# sapien.render.set_ray_tracing_path_depth(8)
# sapien.render.set_ray_tracing_denoiser("oidn")
```

下载 <https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main> 中的：

1. Background_textture
2. Embodiments
3. Objects
放到 assets 目录中。
`cp task_config/demo_clean.yml task_config/demo_clean_depth.yml`,修改 `task_config/demo_clean_depth.yml` :

```yaml
data_type:
	rgb: true
	depth: true
```

开启深度记录

然后切到 `RoboOrchardLab/projects/holobrain`，执行：`bash collect_data.sh beat_block_hammer demo_clean_depth 0`

`collect_data.sh` 内容如下：

```bash
#!/bin/bash

task_name=${1}
task_config=${2}
gpu_id=${3}

./script/.update_path.sh > /dev/null 2>&1

export CUDA_VISIBLE_DEVICES=${gpu_id}

PYTHONWARNINGS=ignore::UserWarning \
python script/collect_data.py $task_name $task_config
```

切换 holobrain，执行：

```bash
python3 -m robo_orchard_lab.dataset.robotwin.robotwin_packer     --input_path /data/code/RoboTwin/data    --output_path /data/tm
p/robotwin_test     --task_names beat_block_hammer     --config_name demo_clean
```

然后执行

```bash
python3 scripts/data_visualize.py \
    --config configs/config_holobrain_qwen_common.py \
    --dataset_names robotwin2_0 \
    --workspace /data/tmp/robotwin_visualize \
    --manual
```

即可可视化数据集。

## Holobrain 安装

```bash
conda create -n holobrain python=3.10
conda activate holobrain
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
# 若机器本体的cuda 不是128，需要额外在conda 中安装130
conda install -y -n holobrain -c nvidia -c conda-forge cuda-toolkit=12.8
conda activate holobrain
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH

# pytorch3d
git clone https://github.com/facebookresearch/pytorch3d.git
git checkout V0.7.8
conda install -c iopath iopath
# 禁用隔离 pytorch3d cuda 13不支持
cd pytorch3d
conda activate holobrain
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CONDA_PREFIX/bin:$PATH
pip install --no-build-isolation .

# flash-attn
git clone git@github.com:Dao-AILab/flash-attention.git
git checkout V2.8.3
MAX_JOBS=4 pip install . --no-build-isolation

# transformers
pip install transformers==4.57.1
# holobrain
pip install ".[holobrain_0]"

# urdf vis
pip install yourdfpy  
pip install pyglet==1.5.31
```

## 真机训练
### 数据制作

采用艾欧导出的 mcap 数据

```bash
python scripts/convert_hc_tj_mcap.py --input /data/tmp/holobrain_test/dataset_mcap/hc_tj_data_0818/mcap --output ./data/arrow_dataset/hc_tj_260818 --urdf ../../hc_assets/hc_tj_description/urdf/hc_tj_robot.urdf --task-name hc_tj_260818 --instruction "grasp object" --recover-truncated
```

在./data/arrow_dataset/hc_tj_ 260818 生成 duckdb 和 arrow 文件

### 可视化验证
#### Data_vis

```bash
python scripts/data_visualize.py \
--config configs/config_holobrain_qwen_hc_tj.py \
--vis_validation \
--workspace ./workspace/hc_tj_260818_fix_vis \
--max_episode 3
```

可视化前 3 个 episode 经过增强之后的样本，并在 `./workspace/hc_tj_260818_fix_vis` 中显示

#### Urdf 回放

```bash
python     scripts/replay_hc_tj_mcap_urdf.py     ./data/arrow_dataset/hc_tj_260818_fix/     --episode-index 0     --source actions     --loop
```

### 设置参数

在 `projects/holobrain/configs/config_hc_tj_ro_dataset.py` 设置地址，urdf 路径  

复制 `projects/holobrain/configs/config_holobrain_qwen_hc_tj.py` 设置训练参数

### 训练

参考命令

```bash
# 单机单卡
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/train.py   --config configs/config_holobrain_qwen_hc_tj.py   --workspace ./workspace/hc_tj_smoke
   --kwargs '{"max_step":1,"step_log_freq":1,"save_step_freq":1}'
   
# 多机多卡
accelerate launch --num_machines 1 --num_processes 2 --multi_gpu scripts/train.py --config configs/config_holobrain_qwen_hc_tj.py --workspace ./workspace/260823 --kwargs '{"num_workers":  16}'

# 恢复训练需要加载到新的目录下，如：
accelerate launch --num_machines 1 --num_processes 2 --multi_gpu scripts/train.py --config configs/config_holobrain_qwen_hc_tj.py --workspace ./workspace/260820_resume --kwargs '{"num_workers": 16,"resume_from":"/home/ubuntu/code/RoboOrchardLab/projects/holobrain/workspace/260820/checkpoints/checkpoint_0"}'
```

### 导出

```bash
python scripts/export.py --config /data/tmp/holobrain_test/260823_w/configs/config_holobrain_qwen_hc_tj.py --workspace "$EXPORT_DIR" --kwargs '{  
"vlm_pretrain": "/data/weights/Qwen/Qwen2.5-VL-3B-Instruct",  
"checkpoint": "/data/tmp/holobrain_test/260823_w/checkpoint_5/model.safetensors",  
"urdf": "/data/tmp/holobrain_test/260823_w/urdf/hc_tj_robot.urdf"  
}'
```

### 验证
#### 直接验证

```bash
HF_HUB_OFFLINE=1 \  
TRANSFORMERS_OFFLINE=1 \  
python scripts/train.py \  
--config configs/config_holobrain_qwen_hc_tj.py \  
--workspace ./workspace/hc_tj_eval_ep_1_8 \  
--eval_only \  
--kwargs "{  
\"vlm_pretrain\":\"${QWEN_DIR}\",  
\"checkpoint\":\"${TRAINED_CKPT}\",  
\"validation_datasets\":[\"hc_tj_14dof\"],  
\"validation_episode_indices\":[1,8],  
\"batch_size\":1,  
\"num_workers\":4  
}"
```

#### 离线数据集验证

```bash
# 推理服务端
python scripts/inference_server.py --model_dir "/data/tmp/holobrain_test/260823_w/export/model" --inference_prefix hc_tj --input_profile hc_tj

# 客户端
python scripts/eval_hc_tj_async.py --server-url http://127.0.0.1:2000/holobrain/hc_tj_async/v1 --dataset-path /data/workspaces/RoboOrchardLab/projects/holobrain/data/arrow_dataset/hc_tj_260818_fix --sample-index 0 --sample-index 300 --sample-index 600 --timeout-seconds 300
```

```bash

```