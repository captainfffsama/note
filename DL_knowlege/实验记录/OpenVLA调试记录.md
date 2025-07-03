#具身智能 #VLA #实验记录

# 模型训练情况
## 数据

300 组实验台抓取绝缘子的数据

## 配置

参数如下：

```bash
CUDA_VISIBLE_DEVICES=1 nohup python vla-scripts/finetune.py \
    --vla_path /data1/model_weight/pretrain_weight/openvla/openvla-7b \
    --data_root_dir /data1/workspace/wxl/data/traindata1/tfrecorddata_20250529/ \
    --dataset_name ruijia_robot_grip_dataset \
    --run_root_dir /data1/workspace/huqiong/openvla-oft/ex3/ \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --batch_size 8 \
    --learning_rate 5e-4 \
    --num_steps_before_decay 100000 \
    --max_steps 150005 \
    --save_freq 10000 \
    --save_latest_checkpoint_only False \
    --image_aug True \
    --lora_rank 128 \
    --wandb_entity "ruijia" \
    --wandb_project "openvla" \
    --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state_0530 \
>/data1/workspace/huqiong/openvla-oft/ex3/250530.log 2>&1 &
```

## Loss

![](../Attachments/openvla_train_loss_250605.png)

# 现象

上机之后，模型有大致的运动轨迹，但是执行不精准，夹爪下探到绝缘子之后，夹爪不闭合，然后开始疯狂 " 爱抚 " 绝缘子。

若人为在机械臂下探到绝缘子时，让那一瞬间让机械臂合抓，然后后续控制权依然完全交给机械臂，机械臂则会后续一直加紧夹爪，开始拎起绝缘子往挂钩上挂。

经过参数消融发现，输入图像纯黑和修改文本指令为任意值都不影响它可以输出大致正确轨迹。

修改每一帧的输入 state 为全 0. 机械臂依然会做出下探找绝缘子的动作，但逐渐停止。

# 代码调试分析
## 像素特征可视化

模型输入的像素详细归一化之后如下：

![](../../Attachments/openvla_input_pixel_values.png)

归一化恢复空间维度，可视化 [SigLip 和 Dinov2 的patch特征(b,256,1024)](https://github.com/captainfffsama/openvla/blob/4d2c35432b315b7e8334cfce4e302b43bf4ada6b/prismatic/extern/hf/modeling_prismatic.py#L219) 

SigLip：

![](../../Attachments/openvla-siglip-feature.png)

Dinov2：

![](../../Attachments/openvla-dinov2_patches.png)

## 前向代码分析

具体参见 [`OpenVLAForACtionPrediction.predict_action`](https://github.com/captainfffsama/openvla/blob/4d2c35432b315b7e8334cfce4e302b43bf4ada6b/prismatic/extern/hf/modeling_prismatic.py#L946)

```python
def predict_action(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        unnorm_key: Optional[str] = None,
        proprio=None,
        proprio_projector=None,
        action_head=None,
        noisy_action_projector=None,
        use_film: bool = False,
        **kwargs: str,
    ) -> np.ndarray:
        """Predict actions from input sequence, with options for different prediction methods.

        Args:
            input_ids: Input token ids
            unnorm_key: Key for unnormalization statistics
            proprio: Proprioceptive features
            proprio_projector: Projector for proprioceptive features
            action_head: Optional head for L1 regression or diffusion-based prediction
            noisy_action_projector: Projector for noisy actions in diffusion-based prediction
            use_film: Whether to use FiLM conditioning
            **kwargs: Additional arguments including pixel_values and attention_mask

        Returns:
            Tuple of (unnormalized_actions, action_hidden_states)
        """
        # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        if not torch.all(input_ids[:, -1] == 29871):
            input_ids = torch.cat(
                (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
            )

        pixel_values = kwargs["pixel_values"]
        attention_mask = kwargs["attention_mask"]

        # Create fake labels tensor (needed for action mask)
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # Get number of tokens in prompt (excluding the start token)
        NUM_PROMPT_TOKENS = input_ids.shape[-1] - 1  # Subtract action tokens and stop token

        # Prepare inputs by adding necessary tokens
        input_ids, attention_mask = self._prepare_input_for_action_prediction(input_ids, attention_mask)

        # Update labels tensor for action mask computation later
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        # Get input embeddings and action masks
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        # Extract language embeddings
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # Process vision features
        projected_patch_embeddings = self._process_vision_features(pixel_values, language_embeddings, use_film)

        # Add proprioceptive features if provided
        use_proprio = proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(projected_patch_embeddings.device, dtype=projected_patch_embeddings.dtype)
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, proprio_projector
            )

        # Use diffusion if provided, otherwise use regression or discrete prediction
        use_diffusion = noisy_action_projector is not None and hasattr(action_head, "noise_scheduler")

        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        NUM_PATCHES = self.vision_backbone.get_num_patches() * self.vision_backbone.get_num_images_in_input()
        if use_proprio:
            NUM_PATCHES += 1
        if use_diffusion:
            NUM_PATCHES += 1

        if use_diffusion:
            # Sample random noise with shape equal to output action, used as the starting state for reverse diffusion
            noise = torch.randn(
                size=(1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=input_embeddings.device, dtype=input_embeddings.dtype
            )

            # Run diffusion-based prediction
            normalized_actions, actions_hidden_states = self._run_diffusion_prediction(
                input_embeddings,
                all_actions_mask,
                noise,
                action_head,
                projected_patch_embeddings,
                labels,
                attention_mask,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                noisy_action_projector,
            )
        else:
            # Run regression or discrete token-based prediction
            normalized_actions, actions_hidden_states = self._regression_or_discrete_prediction(
                input_embeddings,
                all_actions_mask,
                projected_patch_embeddings,
                attention_mask,
                labels,
                NUM_PATCHES,
                NUM_PROMPT_TOKENS,
                action_head,
            )

        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, unnorm_key)

        return actions, actions_hidden_states
```

这里当前任务的文本 token 有 34 个 (`language_embedding`)，图像 token512 个 (` projected_patch_embeddings `)，state 嵌入 1 个 (proprio)。初始化 `input_embedding` 为 90 个，组成为 33 个文本 token+action token 占位符（56 个，actions_chunk\*action_dim）+ 结束 token <BOS>（1 个）。

然后将图像 token 插入到 action token 之后，形成 603 个 token。这里注意力没有做 mask，所有 token 都能互相看。

接下来将整个 token 全部扔进 llama2，然后取 action token 对应位置的输出，如果是原版就直接解码。若是 L1 就通过堆叠 resmlp 来回归出一个（b, actions_chunk, action_dim) 的结果。
