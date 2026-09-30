#VLA

# HoloBrain-0 Technical Report
- 论文：[[2602.12062] HoloBrain-0 Technical Report](https://arxiv.org/abs/2602.12062)
- 代码：[RoboOrchardLab/projects/holobrain at master · captainfffsama/RoboOrchardLab](https://github.com/captainfffsama/RoboOrchardLab/tree/master/projects/holobrain)

![](../../Attachments/holobrain_network_architecture.svg)

## 原理
###  异步平滑
#### 推理侧：simpleRTC

融合动作 [i] = w[i] × 旧动作 [i] + (1 - w[i]) × 新动作 [i]

0 <= i < D：w[i] = 1 完全沿用旧动作

D <= i < H：w[i] 从接近 1 衰减到接近 0

i >= H：保持原始新预测

其中 w = c × (exp(c) - 1) / (e - 1)。

举例：若旧动作剩 20 步、`delay_horizon=6`、  
`max_horizon=32`，则第 0–5 步完全沿旧动作，第 6–19 步渐变，第 20 步起完全  
使用新预测。

#### 训练侧：teacher-forcing
1. 使用均匀随机采样出加噪时间步 t(0~999 归一化到 1)（并行时，每个并行一样）
2. 以 2% 概率触发 pred_steps // 4 的泊松分布得到替换的 action chunk 步数 d（每个样本独立）
3. 依据 $A_t=\sqrt{\bar\alpha_t}A+\sqrt{1-\bar\alpha_t}\epsilon$ ，求得带噪输入动作（这里只对关节角算这个，其余状态依据关节角重算）。将 $A_t$ 前 d 个动作换成 $A$ 中真实动作。
4. 网络接收 $( A_t,t,图像emb，文本emb)$ ,输出预测干净的动作，然后和真实动作求 Smooth L 1，所有位置都算。

这里网络结构为：

1. 编码带噪动作。输入形状为 [B, pred_steps, num_joint, state_dims]。代码先按关节整理，再把每 chunk_size=4 个连续动作拼成一个 token，经线性层映射到 embed_dims=256。所以 pred_steps=64 时，每个关节有 16 个时间 token。
2. 注入扩散时间步。 整数 t 先经过正弦/余弦频率编码和 MLP，形成时间特征。每层的 AdaRMSNorm 用它调节 token 特征，并产生注意力及前馈层的门控参数。因此同一个带噪动作在不同 t 下，可以得到不同的干净动作预测。
3. 融合其他条件。 当前配置重复 6 次以下模块：时间与关节联合注意力、对图像特征的交叉注意力、对文本特征的交叉注意力，以及前馈层。历史机器人状态先由 robot_encoder 编码，供时间与关节注意力使用；时间注意力有因果掩码，当前位置不能读取未来的动作 token。
4. 还原完整动作。 UpsampleHead 把 16 个时间 token 上采样回 64 个动作，并输出每个关节的 8 维状态：关节角、三维位置和四元数。当前配置的网络头采用相对状态表示，随后 get_prediction 结合历史状态将其转换为用于比较的状态预测。最终预测与完整的真实动作 chunk 计算 loss。

和 [标准的训练时 RTC](Training-Time%20Action%20Conditioning%20for%20Efficient%20Real-Time%20Chunking.md) 的不同：

| 环节      | HoloBrain                                               | 论文的 training-time RTC               |
| ------- | ------------------------------------------------------- | ----------------------------------- |
| 干净前缀    | 先生成加噪轨迹，再把随机前缀替换成真实状态                                   | 前缀使用真实动作                            |
| 何时使用    | 每条扩展后的训练样本以 `0.02` 概率启用；长度服从均值为 `pred_steps // 4` 的泊松分布 | 训练时采样模拟的推理延迟 \(d\)，用它确定前缀长度         |
| 生成模型    | DDPM 加噪，预测干净动作                                          | Flow matching，预测速度                  |
| 时间条件    | **整条轨迹共用一个扩散时间步**；替换前缀后没有给它单独设置时间步                      | 前缀的时间设为 1，后缀保持采样时间；模型能区分“已确定”和“待生成” |
| 训练 loss | **仍计算前缀的动作误差**                                          | **只计算后缀的误差**                        |
| 推理      | 默认从噪声生成整段轨迹，没有与训练替换对应的前缀锁定过程                            | 每次生成迭代都写回旧 chunk 提供的前缀              |

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