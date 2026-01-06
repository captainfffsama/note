```yaml
seed: 42
models:
  separate: false
  policy:
    class: GaussianMixin
    clip_actions: false
    clip_log_std: true
    min_log_std: -20.0
    max_log_std: 2.0
    initial_log_std: 0.0
    network:
    - name: net
      input: OBSERVATIONS
      layers:
      - 512
      - 256
      - 128
      activations: elu
    output: ACTIONS
  value:
    class: DeterministicMixin
    clip_actions: false
    network:
    - name: net
      input: OBSERVATIONS
      layers:
      - 512
      - 256
      - 128
      activations: elu
    output: ONE
memory:
  class: RandomMemory
  memory_size: -1
agent:
  class: PPO
  rollouts: 24
  learning_epochs: 5
  mini_batches: 4
  discount_factor: 0.995
  lambda: 0.95
  learning_rate: 0.001
  learning_rate_scheduler: KLAdaptiveLR
  learning_rate_scheduler_kwargs:
    kl_threshold: 0.01
  state_preprocessor: null
  state_preprocessor_kwargs: null
  value_preprocessor: null
  value_preprocessor_kwargs: null
  random_timesteps: 0
  learning_starts: 0
  grad_norm_clip: 1.0
  ratio_clip: 0.2
  value_clip: 0.2
  clip_predicted_values: true
  entropy_loss_scale: 0.01
  value_loss_scale: 1.0
  kl_threshold: 0.0
  rewards_shaper_scale: 1.0
  time_limit_bootstrap: false
  experiment:
    directory: /data/workspaces/isaac/IsaacLab/logs/skrl/h1_rough
    experiment_name: 2025-12-24_13-16-59_ppo_torch
    write_interval: auto
    checkpoint_interval: auto
trainer:
  class: SequentialTrainer
  timesteps: 72000
  environment_info: log
  close_environment_at_exit: false
```