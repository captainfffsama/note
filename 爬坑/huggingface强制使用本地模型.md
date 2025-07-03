在环境变量中设置：

```python
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
```

Pi0 的训练启动可以参考：

```bash
#!/bin/bash
# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# 进入工作目录
cd /root/lerobot
# 运行训练脚本
nohup python lerobot/scripts/train.py \
--policy.path=lerobot/pi0 \
--policy.device=cuda \
--dataset.repo_id=test \
--dataset.root=/data1/datasets/ur_grasp_db/ur_grasp_v2_1500/ \
--batch_size=16 \
--steps=40000 \
--output_dir=/data1/workspace/huqiong/train_log/lerobot/pi0/0627 \
--wandb.enable=true \
--wandb.project=pi0_test \
--wandb.mode=offline >/data1/workspace/huqiong/train_log/lerobot/pi0/0627.log 2>&1 &
```