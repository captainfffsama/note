---
tags:
  - "#lerobot"
  - "#VLA"
---
# 添加一个策略

本指南将带你实现一个自定义策略，并使其能够与 LeRobot 的训练、评估和部署工具协同工作。有两条路径：

- **插件（仓库外 / out-of-tree）**：将策略作为独立的 `lerobot_policy_*` 包发布。迭代快、无需提交 PR、易于维护；适合实验、内部使用或独立发布。
- **内置贡献（in-tree）**：将策略直接合入 `src/lerobot/policies/`。需要提交 PR，但策略会成为库的一等公民。

通常建议先从插件路径开始；当策略稳定下来，并且确实适合随库发布时，再迁移至内置实现。

无论选择哪条路径，构成策略的基础组件都相同：配置类、策略类和处理器工厂。本指南前半部分介绍这些通用组件，后半部分说明各路径所需的脚手架（[路径 A](#路径-a仓库外插件)、[路径 B](#路径-b贡献至仓库内)）。

关于约定的说明：机器人学习是一个快速演进的领域，“策略应该长什么样”可能随着每一种新架构而变化。这里的约定之所以存在，是为了让 `lerobot-train` 和 `lerobot-eval` 能够以统一方式运行不同的模型。当新策略确实不适合这些约定时，请提出讨论（在 PR 或 Issue 中均可）——这些约定并非不可改变。

---

## 策略的组成

每个策略由三个基础组件组成。以下名称中的 `my_policy` 是占位符，应替换为你的策略名称。这个名称至关重要：它必须同时匹配传给 `@PreTrainedConfig.register_subclass` 的字符串、`MyPolicy.name` 类属性，以及 `make_<name>_pre_post_processors` 工厂函数名称。

### 配置类

继承 [`PreTrainedConfig`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/configs/policies.py)，并注册你的策略类型。以下是模板；请根据策略架构和训练要求调整参数与方法。

```python
# configuration_my_policy.py
from dataclasses import dataclass, field
from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.optim import CosineDecayWithWarmupSchedulerConfig

@PreTrainedConfig.register_subclass("my_policy")
@dataclass
class MyPolicyConfig(PreTrainedConfig):
    """MyPolicy 的配置类。

    Args:
        n_obs_steps: 用作输入的观测步数
        horizon: 动作预测时间范围
        n_action_steps: 要执行的动作步数
        hidden_dim: 策略网络的隐藏层维度
        # 在此添加策略特有参数
    """

    horizon: int = 50
    n_action_steps: int = 50
    hidden_dim: int = 256

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.horizon:
            raise ValueError("n_action_steps cannot exceed horizon")

    def validate_features(self) -> None:
        """验证输入/输出特征的兼容性。

        请在策略的 __init__ 中显式调用；基类不会自动调用。
        """
        if not self.image_features:
            raise ValueError("MyPolicy requires at least one image feature.")
        if self.action_feature is None:
            raise ValueError("MyPolicy requires 'action' in output_features.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self):
        """返回 lerobot.optim 中的 LRSchedulerConfig，或 None。"""
        return None

    @property
    def observation_delta_indices(self) -> list[int] | None:
        """数据集加载器为每项观测提供的相对时间步偏移。

        单帧策略应返回 None。对于消费多个过去或未来帧的时序策略，
        返回偏移量列表，例如 [-20, -10, 0, 10] 表示以步长 10
        取 3 个过去帧和 1 个未来帧。
        """
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        """数据集加载器为动作块返回的相对时间步偏移。"""
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
```

传给 `@register_subclass` 的字符串必须与下一节的 `MyPolicy.name` 一致，用户也将通过 CLI 的 `--policy.type` 传入它。除非确有特殊需要，`get_optimizer_preset` 应默认使用 `lerobot.optim` 中的 `AdamW`。

### 策略类

继承 [`PreTrainedPolicy`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pretrained.py)，并设置两个类属性；二者都会由 `__init_subclass__` 检查：

```python
# modeling_my_policy.py
import torch
import torch.nn as nn
from typing import Any

from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import ACTION
from .configuration_my_policy import MyPolicyConfig

class MyPolicy(PreTrainedPolicy):
    config_class = MyPolicyConfig
    name = "my_policy"

    def __init__(self, config: MyPolicyConfig, dataset_stats: dict[str, Any] = None):
        super().__init__(config, dataset_stats)
        config.validate_features()
        self.config = config
        self.model = ...  # 在此放置你的 nn.Module

    def reset(self):
        """重置每个 episode 的状态。lerobot-eval 会在每个 episode 开始时调用。"""
        ...

    def get_optim_params(self) -> dict:
        """返回传给优化器的参数，例如按组配置 lr/wd。"""
        return {"params": self.parameters()}

    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """针对当前观测返回完整的动作块，形状为 (B, chunk_size, action_dim)。"""
        ...

    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """返回当前时间步的单个动作；推理时每一步都会调用。"""
        ...

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """计算训练损失。

        返回 (loss, output_dict)。output_dict 可以为 None；其中所有内容必须是
        适合日志记录的原生 Python 值，不能含有携带梯度的 Tensor。

        batch["action_is_pad"] 是形状为 (B, horizon) 的布尔掩码，标记因
        episode 在 horizon 步之前结束而补齐的时间步；可以将它们排除出损失计算。
        """
        actions = batch[ACTION]
        action_is_pad = batch.get("action_is_pad")
        ...
        return loss, {"some_loss_component": some_loss_component.item()}
```

训练和评估循环会调用的方法：

| 方法 | 使用方 | 功能 |
| --- | --- | --- |
| `reset() -> None` | `lerobot-eval` | 在每个 episode 开始时清除状态。 |
| `select_action(batch, **kwargs) -> Tensor` | `lerobot-eval` | 返回下一个动作 `(B, action_dim)`；每一步调用。 |
| `predict_action_chunk(batch, **kwargs) -> Tensor` | 策略自身 | 返回动作块 `(B, chunk_size, action_dim)`。目前它在基类中是抽象方法；如果策略不使用动作块，应抛出 `NotImplementedError`。 |
| `forward(batch, reduction="mean") -> tuple[Tensor, dict \| None]` | `lerobot-train` | 返回 `(loss, output_dict)`。如需支持按样本加权，应接受 `reduction="none"`。 |
| `get_optim_params() -> dict` | 优化器 | 简单策略返回 `self.parameters()`；多优化器策略返回具名参数字典。 |
| `update() -> None`（可选） | `lerobot-train` | 若定义，则每次优化器更新后调用。可用于 EMA、目标网络、经验回放缓冲区等。 |

Batch 是以 [`lerobot.utils.constants`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/utils/constants.py) 中常量为键的扁平字典，例如：`OBS_STATE`（`observation.state.<motor>`）、`OBS_IMAGES`（`observation.images.<camera>`）、`OBS_LANGUAGE`、`ACTION` 等。请复用这些常量，不要自行发明新的前缀。

若模型大到需要使用[分片多 GPU 训练](./multi_gpu_training#sharded-training-fsdp)，还应声明 FSDP 的包裹单元，即分片所操作的重复模块类：

```python
class MyPolicy(PreTrainedPolicy):
    ...
    _fsdp_wrap_modules = ["MyTransformerBlock"]
```

只需这项声明，`--parallelism.dp_shard=N` 就能直接用于该策略；用户仍可通过 `--accelerator.fsdp.wrap_modules` 覆盖它。没有任何包裹来源时，分片运行会在启动时按设计失败。

### 处理器函数

LeRobot 使用 `PolicyProcessorPipeline` 在策略前后分别完成输入归一化和输出反归一化。可参考 [`processor_act.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/processor_act.py) 或 [`processor_diffusion.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/processor_diffusion.py)。

这里需要格外注意：处理器是复现问题最常见的根源。归一化模式（`IDENTITY`、`MEAN_STD`、`MIN_MAX`、`QUANTILES` / `QUANTILE10`）或被归一化特征与训练时不一致，并不会报错，但会悄然破坏结果。务必确保模式与检查点训练时一致、所需统计量存在（例如 `QUANTILES` 需要 `q01` / `q99`），且前后处理器保持一致。

```python
# processor_my_policy.py
from typing import Any
import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline


def make_my_policy_pre_post_processors(
    config,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    preprocessor = ...   # 构建输入处理器
    postprocessor = ...  # 构建输出处理器
    return preprocessor, postprocessor
```

**重要——函数命名：**LeRobot 通过名称发现处理器。函数必须命名为 `make_{policy_name}_pre_post_processors`，并与传给 `@PreTrainedConfig.register_subclass` 的字符串对应。

---

## 路径 A：仓库外插件

发布策略最快的方式：将其打包为独立 Python 分发包，与 LeRobot 一起安装。无需 PR，你拥有自己的发布周期，也可以使用自己的命名空间发布到 PyPI。

### 包结构

创建一个以 `lerobot_policy_`（**重要**）开头、后接策略名称的包：

```text
lerobot_policy_my_policy/
├── pyproject.toml
└── src/
    └── lerobot_policy_my_policy/
        ├── __init__.py
        ├── configuration_my_policy.py
        ├── modeling_my_policy.py
        └── processor_my_policy.py
```

### `pyproject.toml`

```toml
[project]
name = "lerobot_policy_my_policy"
version = "0.1.0"
dependencies = [
    # 策略特有依赖
]
requires-python = ">= 3.10"

[build-system]
build-backend = # 你的构建后端
requires = # 你的构建系统
```

### 包的 `__init__.py`

在包的 `__init__.py` 中导出类，并处理未安装 `lerobot` 的情况：

```python
# __init__.py
"""用于 LeRobot 的自定义策略包。"""

try:
    import lerobot  # noqa: F401
except ImportError:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    )

from .configuration_my_policy import MyPolicyConfig
from .modeling_my_policy import MyPolicy
from .processor_my_policy import make_my_policy_pre_post_processors

__all__ = [
    "MyPolicyConfig",
    "MyPolicy",
    "make_my_policy_pre_post_processors",
]
```

### 安装与使用

```bash
cd lerobot_policy_my_policy
pip install -e .

# 或者在发布到 PyPI 后安装
pip install lerobot_policy_my_policy
```

安装后，策略会自动集成到 LeRobot 的训练和评估工具中：

```bash
lerobot-train \
    --policy.type my_policy \
    --env.type pusht \
    --steps 200000
```

---

## 路径 B：贡献至仓库内

当策略趋于稳定，且随库发布具有明确价值时，可以将其直接合入 LeRobot。请先阅读通用[贡献指南](./contributing)和 [PR 模板](https://github.com/huggingface/lerobot/blob/main/.github/PULL_REQUEST_TEMPLATE.md)，其中定义了每个 PR 都必须满足的测试与质量要求，例如 `pre-commit run -a`、`pytest` 和社区审查规则。以下内容是在这些通用要求之上的策略专属要求。

### 仓库内目录结构

```text
src/lerobot/policies/my_policy/
├── __init__.py                    # 重新导出配置、模型和处理器工厂
├── configuration_my_policy.py     # MyPolicyConfig + @register_subclass
├── modeling_my_policy.py          # MyPolicy(PreTrainedPolicy)
├── processor_my_policy.py         # make_my_policy_pre_post_processors
└── README.md                      # 指向 ../../../../docs/source/policy_my_policy_README.md 的符号链接
```

注意：

- 源码旁的 `README.md` 是指向 `docs/source/policy_<name>_README.md` 的**符号链接**；实际文件位于 `docs/` 下。现有策略（act、smolvla、diffusion 等）均采用这一方式。策略 README 通常只包含论文链接和 BibTeX 引用。
- 面向用户的教程——安装方式、训练方法、超参数和基准结果——应单独放在 `docs/source/<my_policy>.mdx`，并在 `_toctree.yml` 的 “Policies” 下注册。

文件名至关重要：工厂按名称延迟导入，处理器也依赖 `make_<policy_name>_pre_post_processors` 约定发现。

### 接入

有两个位置需要注册你的策略，且均按名称识别。

1. ** `policies/__init__.py` **：重新导出 `MyPolicyConfig` 并将其加入 `__all__`。该导入会触发 `@PreTrainedConfig.register_subclass("my_policy")`，使工厂之后能按约定解析所有组件。**不要**重新导出模型类；它通过工厂延迟加载，以保证 `import lerobot` 足够快。
2. ** `templates/lerobot_modelcard_template.md` 和根目录 `README.md` **：训练结束时的发布器会依据模板生成模型卡。请在 `model_name` 分支中添加一行策略描述、在 `policy_docs` 中映射文档链接，并可选择在 `diagrams` 中添加架构图。之后在根 `README.md` 的模型表中、正确的类别下添加策略及其文档链接。

建议参考一个结构最接近的现有策略；所需改动通常很少。

### 重型／可选依赖

多数策略会依赖重型骨干网络，如 transformers、diffusers 或特定 VLM SDK。只要可行，应从 `transformers` 或 `diffusers` 加载这些组件，而不是在仓库中重新实现架构。

约定采用**两阶段依赖保护**：模块顶部使用 `TYPE_CHECKING` 保护导入，并在构造函数中调用 `require_package` 进行运行时检查。[`modeling_diffusion.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/diffusion/modeling_diffusion.py) 是标准参考。

这样可以保证：

- 未安装额外依赖时，`import lerobot.policies` 仍然可用；
- 类型检查器仍能看到真实类型；
- 在缺少依赖时实例化策略，会明确提示应执行 `pip install 'lerobot[diffusion]'`。

同时应在 [`pyproject.toml`](https://github.com/huggingface/lerobot/blob/main/pyproject.toml) 的 `[project.optional-dependencies]` 中添加对应 extra，并将它加入 `all` extra。

### 不要复制建模文件——应继承它

如果策略需要修改 `transformers` 中已有的骨干模型，例如自定义条件输入、额外输入或替换子模块，**不要复制其 `modeling_*.py` 文件**。应继承最小的上游单元，只覆盖需要变化的部分。[`pi_gemma.py`](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/pi_gemma.py) 是标准范例：它通过继承 `GemmaModel` / `PaliGemmaModel` 并覆盖解码器层的 forward，在约 370 行代码内注入 AdaRMS 条件，而不是派生约 2,000 行的建模文件。

对已加载的原生模型进行修改也可以，例如裁剪层、扩展 tokenizer、捕获隐藏状态。若 PR 包含复制的建模文件，审阅者会要求改用这一模式；唯一可接受的例外是该模型完全不存在于 `transformers` 中。

### 基准测试与已发布检查点

当一个新策略带有可用检查点和至少一个可复现结果时，会更容易审阅，也更有实际价值。

至少选择一个内置基准测试。LeRobot 提供带有独立 Docker 镜像的仿真基准，如 LIBERO、LIBERO-plus、Meta-World、RoboTwin 2.0、RoboCasa 365、RoboCerebra、RoboMME、VLABench 等。应选择与策略模态匹配的基准：VLA 通常使用 LIBERO 或 VLABench；纯图像行为克隆通常使用 LIBERO 或 Meta-World。完整列表见文档侧边栏的 Benchmarks。

将检查点和处理器推送至 Hub，例如 `lerobot/<policy>_<benchmark>`。最简单的方式是在训练时设置：

```bash
--policy.repo_id=<namespace>/<repo>
--policy.push_to_hub=true
```

`lerobot-train` 会在训练结束时发布模型、两个处理器和模型卡。若要在训练后发布既有检查点，可上传其 `pretrained_model/` 目录，或对分片格式检查点使用 `lerobot-convert-dcp --push_to_hub=...`。

应在策略的 MDX 文档中报告结果，附上准确的 `lerobot-eval` 命令和硬件信息，确保其他人可以复现。每个套件应使用 `n_episodes ≥ 50`，以获得稳定的成功率估计。

若策略仅适用于真实机器人，且无适用仿真基准，则应提供：Hub 上公开的训练数据集、`lerobot-train` 命令、检查点，以及通过 `lerobot-rollout --policy.path=...` 进行不少于 10 个 episode 的真实机器人成功率。

### PR 检查清单

除 [`CONTRIBUTING.md`](https://github.com/huggingface/lerobot/blob/main/CONTRIBUTING.md) 和 [PR 模板](https://github.com/huggingface/lerobot/blob/main/.github/PULL_REQUEST_TEMPLATE.md) 的通用要求外，审阅者还会检查：

- [ ] `MyPolicy` 与 `MyPolicyConfig` 覆盖上述接口，且能通过 `__init_subclass__` 检查。
- [ ] `policies/__init__.py` 重新导出了配置类。
- [ ] `make_my_policy_pre_post_processors` 遵循命名约定。
- [ ] 可选依赖位于 `[project.optional-dependencies]` extra 后，并使用 `TYPE_CHECKING + require_package` 保护。
- [ ] 更新 `tests/policies/`，提交向后兼容产物及策略专属测试。
- [ ] 将策略 README 符号链接到 `docs/source/policy_<name>_README.md`；编写用户文档并加入 `_toctree.yml`。
- [ ] `lerobot-train --policy.type my_policy ...` 至少能端到端运行若干步；保存的检查点可被 `lerobot-eval` 或 `lerobot-rollout` 加载运行。
- [ ] 更新模型卡模板和根目录模型表。
- [ ] 策略 MDX 中至少包含一个可复现的基准评估及已发布检查点。

获得整洁 PR 的最快方式是复制最接近的现有策略目录，重命名后逐个方法替换内容。无需等到所有内容打磨完美——尽早创建 Draft PR 并与维护者迭代。

---

## 示例与社区贡献

可参考以下策略实现：

- [DiTFlow Policy](https://github.com/danielsanjosepro/lerobot_policy_ditflow)：采用流匹配目标的 Diffusion Transformer 策略。
- [DiTFlow Example](https://github.com/danielsanjosepro/test_lerobot_policy_ditflow)：其使用示例。

感谢你为 LeRobot 引入新策略。每个合入 `main` 的架构，以及社区发布的每个插件，都会让这个库更有用，也更能代表机器人学习的发展方向。期待看到你的成果。 🤗