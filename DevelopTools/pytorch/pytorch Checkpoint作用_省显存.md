---
tags:
  - "#pytorch"
---

`torch.utils.checkpoint` 是 PyTorch 中用于实现激活重计算（Activation Checkpointing / Gradient Checkpointing）的核心工具。

它的本质是一种“以时间换空间”**的优化技术：在模型训练的前向传播中**不保存**中间层的激活值，而在反向传播需要计算梯度时**临时重新计算这些激活值，从而大幅降低 GPU 显存占用。

## 核心原理：标准训练 vs 激活重计算

深度学习模型在反向传播（Backpropagation）根据链式法则计算参数梯度时，必须依赖前向传播时产生的中间激活值（Activation）。

### 1. 标准反向传播（Normal Forward & Backward）

- **前向过程**：依次执行 Layer 1 $\rightarrow$ Layer 2 $\rightarrow$ Layer 3… 每层的中间输出全部缓存在显存中。

- **显存占用**：随着网络层数变深、Batch Size 变大、Sequence 变长，保存所有中间激活值的显存需求呈线性暴增（在大语言模型和超深卷积网中，激活值显存往往远超模型参数自身显存）。

- **反向过程**：直接读取显存中存好的激活值，计算梯度并释放。

```
标准模式：前向: [x] -> [Layer 1] -> (保存 a1) -> [Layer 2] -> (保存 a2) -> [Layer 3] -> loss反向: loss -> [Layer 3 反向] (用 a2) -> [Layer 2 反向] (用 a1) -> [Layer 1 反向]
```

### 2. Checkpoint 机制（Gradient Checkpointing）

- **前向过程**：对被 `checkpoint` 包裹的代码段，**只保存输入张量**，中间计算过程完全不保留激活值（将其释放掉）。

- **显存占用**：显存中只保留少量关键节点（Checkpoint nodes）的输入，峰值显存显著降低（通常可减少 50% ~ 75% 的激活显存）。

- **反向过程**：当梯度反向传递到该段时，PyTorch 会利用之前保存的输入张量，**把前向传播再跑一遍**，临时生成该段所需的中间激活值，计算完当前梯度后立即释放。

```
Checkpoint 模式：前向: [x] (保存 x) -> [Layer 1] (丢弃 a1) -> [Layer 2] (丢弃 a2) -> [Layer 3] -> loss反向: loss -> [Layer 3 反向] -> (拿保存的 x 重新执行 Layer 1 & 2 算回 a1, a2) -> 完成梯度计算并立即销毁 a1, a2
```

## 代价与收益对比

|**维度**|**标准训练**|**使用 Checkpoint**|
|---|---|---|
|**显存占用（激活值）**| $O(N)$ ，随层数线性增加|显著降低（通常可压缩至 $O(\sqrt{N})$ 水平）|
|**计算时间**|正常计算量（1 次 Forward + 1 次 Backward）|增加约 20% ~ 33%（多了一次部分 Forward）|
|**可训练规模**|容易遇到 Out-Of-Memory (OOM)|允许在相同硬件上跑更大 Batch Size 或更长序列|

## 基本使用方法

在 PyTorch 中，只需将前向计算逻辑封装为一个可调用对象（如函数或 `nn.Module`），然后通过 `checkpoint` 调用：

Python

```
import torchimport torch.nn as nnfrom torch.utils.checkpoint import checkpointclass LargeBlock(nn.Module):    def __init__(self, dim):        super().__init__()        self.ffn = nn.Sequential(            nn.Linear(dim, dim * 4),            nn.GELU(),            nn.Linear(dim * 4, dim)        )    def forward(self, x):        return self.ffn(x)class MyTransformer(nn.Module):    def __init__(self, num_layers=24, dim=1024):        super().__init__()        self.layers = nn.ModuleList([LargeBlock(dim) for _ in range(num_layers)])    def forward(self, x):        for layer in self.layers:            # 关键：使用 checkpoint 执行 forward            # 推荐显式指定 use_reentrant=False            x = checkpoint(layer, x, use_reentrant=False)        return x
```

## 核心参数与注意事项

### 1. `use_reentrant=False` vs `True`

- ** `use_reentrant=True`（旧版默认）**：基于 Autograd 嵌套引擎实现。对部分复杂计算图（如包含 control flow、某些 hooks、或者未设 `requires_grad=True` 的输入）兼容性较差，容易产生不可预测的行为。

- ** `use_reentrant=False`（推荐标准）**：基于 PyTorch 的 `SavedTensorHook` 机制实现，与原生 Autograd 行为完全一致，支持非张量参数、正常追踪梯度以及精确释放显存。

### 2. 随机数状态同步（RNG State）

如果被 checkpoint 的模块中包含随机操作（例如 **Dropout**），在前向传播丢弃某些神经元后，反向传播重算前向时必须保证丢弃的是**完全相同的神经元**。

- `checkpoint` 默认参数 `preserve_rng_state=True`。

- 它在前向传播进入代码块前会拷贝当前 GPU/CPU 的 RNG 种子状态，在反向重计算时还原该状态，确保随机操作的结果严格一致。

### 3. 输入必须参与求导

在早期版本或使用 `use_reentrant=True` 时，传给 `checkpoint` 的参数中**至少要有一个张量的 `requires_grad=True` **，否则 PyTorch 会认为这一段无需反向传播求导，从而跳过重算导致梯度断裂。

### 4. 严禁原地修改（In-Place Operations）

重计算需要依赖最开始保存的原始输入。如果在代码块内部对输入张量进行了原地修改（如 `x += 1` 或 `tensor.add_()`），反向重算时就会报错或计算出错误梯度。

### 5. 推理阶段（Inference）无需使用

推理阶段不会建立计算图，也不会存储反向传播所需的激活值。使用 `checkpoint` 不仅不会节省任何显存，反而可能引入额外的开销。因此在评估和推理时应通过常规前向流程执行。