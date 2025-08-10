#具身智能 #VLA 

LLM 部分

```bash
GemmaModel(
  (embed_tokens): Embedding(257152, 2048, padding_idx=0)
  (layers): ModuleList(
    (0-17): 18 x GemmaDecoderLayer(
      (self_attn): GemmaAttention(
        (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
        (k_proj): Linear(in_features=2048, out_features=256, bias=False)
        (v_proj): Linear(in_features=2048, out_features=256, bias=False)
        (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
      )
      (mlp): GemmaMLP(
        (gate_proj): Linear(in_features=2048, out_features=16384, bias=False)
        (up_proj): Linear(in_features=2048, out_features=16384, bias=False)
        (down_proj): Linear(in_features=16384, out_features=2048, bias=False)
        (act_fn): PytorchGELUTanh()
      )
      (input_layernorm): GemmaRMSNorm((2048,), eps=1e-06)
      (post_attention_layernorm): GemmaRMSNorm((2048,), eps=1e-06)
    )
  )
  (norm): GemmaRMSNorm((2048,), eps=1e-06)
  (rotary_emb): GemmaRotaryEmbedding()
)
```

Flow matching 部分

```bash
GemmaModel(
  (embed_tokens): None
  (layers): ModuleList(
    (0-17): 18 x GemmaDecoderLayer(
      (self_attn): GemmaAttention(
        (q_proj): Linear(in_features=1024, out_features=2048, bias=False)
        (k_proj): Linear(in_features=1024, out_features=256, bias=False)
        (v_proj): Linear(in_features=1024, out_features=256, bias=False)
        (o_proj): Linear(in_features=2048, out_features=1024, bias=False)
      )
      (mlp): GemmaMLP(
        (gate_proj): Linear(in_features=1024, out_features=4096, bias=False)
        (up_proj): Linear(in_features=1024, out_features=4096, bias=False)
        (down_proj): Linear(in_features=4096, out_features=1024, bias=False)
        (act_fn): PytorchGELUTanh()
      )
      (input_layernorm): GemmaRMSNorm((1024,), eps=1e-06)
      (post_attention_layernorm): GemmaRMSNorm((1024,), eps=1e-06)
    )
  )
  (norm): GemmaRMSNorm((1024,), eps=1e-06)
  (rotary_emb): GemmaRotaryEmbedding()
)
```

这里实现实际上是把 vision，language，state 全部变成 token，然后一起做自注意力，然后使用 attention_mask 来控制 token 之间的相互关注。

这里总计 token 总共 611 个，顺序为：图片 1 (256token)+ 图片 2 (256token)+ 语言（48token）+state (1token)+action nosiy (50token)

这里 action nosiy 为 action nosiy 和 time_emb 在 hidden dim 上拼接，然后通过一个 2 层 mlp 缩放回 1024 的 hidden size

Pi0 在实现上 vlm 和 action expert 都有 18 层 decoder，先每个层每个模型自己单独对对应的 embed 进行投影变换，然后在 seq length 上合一起，使用 attension mask 来空间交互，然后在分别拆分开在各自的模型层上走后续的 FFN 和残差结构

## 一些实现小经验
1.  Vision 在编码时，也会采用和 attention 同款的归一化
2. Attention_mask 在通过 softmax 前，会把 false 地方填一个很大的负数，比如 `-2.3819763e38`