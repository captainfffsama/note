#具身智能 #VLA #实验记录

# 模型训练情况
## 数据

2000 组实验台抓取绝缘子的数据，其中 action 和 state 都是从臂的状态信息，action 是下一时刻的 state，保存都是 joint

## 配置

参数如下：

```bash
{
    "batch_size": 64,
    "dataset": {
        "episodes": None,
        "image_transforms": {
            "enable": False,
            "max_num_transforms": 3,
            "random_order": False,
            "tfs": {
                "brightness": {"kwargs": {"brightness": [0.8, 1.2]}, "type": "ColorJitter", "weight": 1.0},
                "contrast": {"kwargs": {"contrast": [0.8, 1.2]}, "type": "ColorJitter", "weight": 1.0},
                "hue": {"kwargs": {"hue": [-0.05, 0.05]}, "type": "ColorJitter", "weight": 1.0},
                "saturation": {"kwargs": {"saturation": [0.5, 1.5]}, "type": "ColorJitter", "weight": 1.0},
                "sharpness": {"kwargs": {"sharpness": [0.5, 1.5]}, "type": "SharpnessJitter", "weight": 1.0},
            },
        },
        "repo_id": "base",
        "revision": None,
        "root": "/data1/datasets/ur_grasp_db/ur_grasp_v2_2000/",
        "use_imagenet_stats": True,
        "video_backend": "torchcodec",
    },
    "env": None,
    "eval": {"batch_size": 50, "n_episodes": 50, "use_async_envs": False},
    "eval_freq": 20000,
    "job_name": "smolvla",
    "log_freq": 200,
    "num_workers": 16,
    "optimizer": {
        "betas": [0.9, 0.95],
        "eps": 1e-08,
        "grad_clip_norm": 10.0,
        "lr": 0.0001,
        "type": "adamw",
        "weight_decay": 1e-10,
    },
    "output_dir": "outputs/train/2025-06-13/03-23-33_smolvla",
    "policy": {
        "adapt_to_pi_aloha": False,
        "add_image_special_tokens": False,
        "attention_mode": "cross_attn",
        "chunk_size": 50,
        "device": "cuda",
        "empty_cameras": 0,
        "expert_width_multiplier": 0.75,
        "freeze_vision_encoder": True,
        "input_features": {
            "observation.image": {"shape": [3, 256, 256], "type": "<featuretype.visual: 'visual'=>"},
            "observation.image2": {"shape": [3, 256, 256], "type": "<featuretype.visual: 'visual'=>"},
            "observation.image3": {"shape": [3, 256, 256], "type": "<featuretype.visual: 'visual'=>"},
            "observation.state": {"shape": [6], "type": "<featuretype.state: 'state'=>"},
        },
        "load_vlm_weights": False,
        "max_action_dim": 32,
        "max_period": 4.0,
        "max_state_dim": 32,
        "min_period": 0.004,
        "n_action_steps": 1,
        "n_obs_steps": 1,
        "normalization_mapping": {
            "ACTION": "<normalizationmode.mean_std: 'mean_std'=>",
            "STATE": "<normalizationmode.mean_std: 'mean_std'=>",
            "VISUAL": "<normalizationmode.identity: 'identity'=>",
        },
        "num_expert_layers": 0,
        "num_steps": 10,
        "num_vlm_layers": 16,
        "optimizer_betas": [0.9, 0.95],
        "optimizer_eps": 1e-08,
        "optimizer_grad_clip_norm": 10.0,
        "optimizer_lr": 0.0001,
        "optimizer_weight_decay": 1e-10,
        "output_features": {"action": {"shape": [6], "type": "<featuretype.action: 'action'=\"\">"}},
        "pad_language_to": "max_length",
        "prefix_length": 0,
        "resize_imgs_with_padding": [512, 512],
        "scheduler_decay_lr": 2.5e-06,
        "scheduler_decay_steps": 30000,
        "scheduler_warmup_steps": 1000,
        "self_attn_every_n_layers": 2,
        "tokenizer_max_length": 48,
        "train_expert_only": True,
        "train_state_proj": True,
        "type": "smolvla",
        "use_amp": False,
        "use_cache": True,
        "use_delta_joint_actions_aloha": False,
        "vlm_model_name": "/data1/model_weight/pretrain_weight/SmolVLM2-500M-Video-Instruct/",
    },
    "resume": False,
    "save_checkpoint": True,
    "save_freq": 20000,
    "scheduler": {
        "decay_lr": 2.5e-06,
        "num_decay_steps": 30000,
        "num_warmup_steps": 1000,
        "peak_lr": 0.0001,
        "type": "cosine_decay_with_warmup",
    },
    "seed": 1000,
    "steps": 200000,
    "use_policy_training_preset": True,
    "wandb": {
        "disable_artifact": False,
        "enable": True,
        "entity": None,
        "mode": "offline",
        "notes": None,
        "project": "smolvla_test",
        "run_id": None,
    },
}
```

## Loss

![](../../Attachments/smolvla-base_250627.png)

# 现象

动作轨迹基本不太对，起始会猛然上抬，然后做出尝试下探找物体的趋势，然后在上抬手臂

# 分析
## 1

以 episode0 为例，每个时间步都推理，仅仅计算最近步的，然后 pre 和 label 的差异如下：

```python
L1 error per action dimension: [0.0448854  0.05750808 0.04701609 0.12686084 0.06956683 0.07498549 0.06939534]
L1 errot per action dimension max: [0.51819927 0.6911217  0.436244   1.7873132  0.8009113  0.6098713 0.9188828 ]
图片已保存到: /root/lerobot/hq_test/action_chunk_l1.jpg
L1 error per action chunk: [
0.00835005 0.01282661 0.01785137 0.02237372 0.02560373 0.02938746 0.03196965 0.034814   0.03798549 0.04102642 
0.04418611 0.04686107 0.04879363 0.05058389 0.05294637 0.05501623 0.05635092 0.05822372 0.06064468 0.06192039 
0.06376964 0.06483259 0.06659944 0.06822196 0.07023308 0.07223212 0.07339981 0.07451002 0.07648741 0.07785016
0.08079823 0.08316842 0.08522467 0.08715732 0.08899575 0.09076323 0.09256398 0.09524712 0.09784676 0.09926647 
0.10105765 0.10328534 0.10545202 0.10756673 0.10922726 0.11147095 0.11265522 0.11346199 0.11501446 0.11548091
]
L1 errot per action chunk max: [
0.47436678 0.6536717  0.7958395  0.87356824 0.8886157  0.8979471  0.898545   0.9023491  0.9015027  0.9040194 
0.9055459  0.89433765 0.8892557  0.89182526 0.8979982  0.89410824 0.8980156  0.89890265 0.9006977  0.8987383  
0.8953696  0.89449626 0.89734674 0.9011188  0.9037318  0.90445304 0.96578455 0.9953418  1.0221834  1.0463986
1.1233232  1.187041   1.2820425  1.342097   1.3759367  1.4291232  1.4613118  1.5633969  1.6345625  1.7237537  
1.6365521  1.6846681  1.6793692  1.7776496  1.6629837  1.7000873  1.7873132  1.7303556  1.7037883  1.7116348 
]
图片已保存到: /root/lerobot/hq_test/action_exec_l1.jpg
L1 error per action exec: [0.00486947 0.0050571  0.00529062 0.0136525  0.00902051 0.00883122 0.01172889]
L1 errot per action exec max: [0.0704211  0.02926314 0.05463982 0.15800858 0.1777488  0.07033741 0.47436678]
```

变化如下：

![](../../Attachments/smolvla-base-1_action%20chunk.png)

![](../../Attachments/smolvla-base-1_one%20action.png)

## 一点结论
1. 从 action chunk 上看 L1 差异随 action 长度呈现线性增长，建议选 20 步之前
2. 从整体来看，关节 4 和夹爪相对难学一点