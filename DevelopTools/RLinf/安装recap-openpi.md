可以。**建议使用 Conda Python 3.10 + PyTorch 2.7.1（CUDA 12.8）+ RLinf OpenPI 0.1.1。**

我已确认本机是 **RTX 5090，约 32 GB 显存，驱动 580.105.08**，无需为这套方案升级驱动。5090 的计算能力是 `sm_120`，PyTorch 2.7 开始提供 Blackwell/CUDA 12.8 支持。[NVIDIA 显卡列表](https://developer.nvidia.com/cuda/gpus)、[PyTorch 说明](https://pytorch.org/blog/pytorch-2-7/)

下面的依赖组合已经通过 **Python 3.10 依赖解析检查**；尚未实际安装或执行训练。

**1. 创建独立 Conda 环境**

文档中的 RECAP 是基于 OpenPI 的离线训练流程。官方 `install.sh` 会创建 uv 虚拟环境；下面改为直接把依赖安装到 Conda 中，`uv` 仅作为包安装工具。[RECAP 文档](https://rlinf.readthedocs.io/zh-cn/latest/rst_source/examples/embodied/recap.html)、[uv 对 Conda 的支持](https://docs.astral.sh/uv/pip/environments/)

```
conda create -n recap310 python=3.10 pip -yconda activate recap310python -m pip install -U pip uv# 使用本机已有仓库cd /data/workspaces/RLinf# 本机为 Ubuntu，安装编译、视频解码和渲染基础依赖sudo apt-get updatesudo apt-get install -y \  build-essential cmake git-lfs ffmpeg \  libegl1 libgl1 libglib2.0-0 libosmesa6
```

**2. 固定兼容版本，安装 RECAP/OpenPI**

OpenPI 0.1.1 支持 Python 3.10，但固定依赖 `torch==2.7.1`，并使用定制版 Transformers。因此需要显式选择 `cu128`，避免装成 CUDA 12.6 构建。[OpenPI 发布包](https://pypi.org/project/rlinf-openpi/0.1.1/)

先写入版本约束：

```
cat > "$CONDA_PREFIX/recap-constraints.txt" <<'EOF'torch==2.7.1+cu128torchvision==0.22.1+cu128torchaudio==2.7.1+cu128torchcodec==0.5numpy==1.26.4transformers==4.53.2rlinf-transformer-openpi==4.53.2tokenizers>=0.21,<0.22lerobot==0.3.3datasets==3.6.0pyarrow>=15,<26peft==0.17.1jax==0.5.3orbax-checkpoint==0.11.13tyro==1.0.13setuptools>=69.5.1,<75.9protobuf>=6.33.5,<7EOF
```

安装到当前 Conda 环境：

```
uv --no-config pip install \  --python "$CONDA_PREFIX/bin/python" \  --torch-backend cu128 \  -c "$CONDA_PREFIX/recap-constraints.txt" \  -e '.[embodied]' \  'rlinf-openpi==0.1.1' \  'torchaudio==2.7.1+cu128'
```

这里的 `--no-config` 用于避免仓库默认的其他 PyTorch 版本覆盖上述选择。**先不安装额外的 `flash-attn` 包**，减少 5090 上的编译和内核兼容问题。

**3. 应用 OpenPI 的 Transformers 补丁**

这一步对应仓库安装脚本中的必要操作。先确保定制版文件覆盖到位，再复制 OpenPI 补丁。[官方安装脚本](https://github.com/RLinf/RLinf/blob/main/requirements/install.sh)

```
uv --no-config pip install \  --python "$CONDA_PREFIX/bin/python" \  --no-deps --reinstall \  'rlinf-transformer-openpi==4.53.2'python - <<'PY'import importlib.utilimport shutilfrom pathlib import Pathopenpi_dir = Path(    next(iter(importlib.util.find_spec("openpi").submodule_search_locations)))transformers_dir = Path(    next(iter(importlib.util.find_spec("transformers").submodule_search_locations)))patch_dir = openpi_dir / "models_pytorch" / "transformers_replace"assert patch_dir.is_dir(), f"补丁目录不存在：{patch_dir}"shutil.copytree(patch_dir, transformers_dir, dirs_exist_ok=True)print("OpenPI Transformers 补丁已应用")PYbash requirements/embodied/download_assets.sh --assets openpi
```

之后不要随意执行 `pip install -U transformers` 或 `uv sync`，它们可能覆盖这套版本和补丁。

**4. 检查 GPU 与关键模块**

RECAP 使用 PyTorch 训练；让辅助依赖 JAX 使用 CPU，避免额外占用 GPU：

```
export JAX_PLATFORMS=cpuexport XLA_PYTHON_CLIENT_PREALLOCATE=falseexport MUJOCO_GL=eglexport PYOPENGL_PLATFORM=eglexport PYTHONPATH="/data/workspaces/RLinf:${PYTHONPATH:-}"
```

执行验证：

```
python - <<'PY'import sysimport torchimport transformersfrom transformers import Gemma3ForCausalLM, SiglipVisionModelfrom openpi.models_pytorch import pi0_pytorchfrom rlinf.models.embodiment.value_model.recap.modeling_critic import ValueCriticModelprint("Python:", sys.version.split()[0])print("PyTorch:", torch.__version__)print("CUDA runtime:", torch.version.cuda)print("Transformers:", transformers.__version__)assert torch.cuda.is_available(), "PyTorch 无法访问 GPU"print("GPU:", torch.cuda.get_device_name(0))print("Compute capability:", torch.cuda.get_device_capability(0))x = torch.randn(512, 512, device="cuda")y = x @ xtorch.cuda.synchronize()print("GPU 矩阵运算及关键模块导入成功")PY
```

应看到 Python `3.10.x`、PyTorch `2.7.1+cu128`、CUDA `12.8`、计算能力 `(12, 0)`。

**5. 按需要补充评测环境和模型**

仅处理已有数据、运行 RECAP 四个离线阶段，可以先不安装 ManiSkill。若要复现文档中的 **LIBERO Task 0 评测**，再安装：

```
uv --no-config pip install \  --python "$CONDA_PREFIX/bin/python" \  --torch-backend cu128 \  -c "$CONDA_PREFIX/recap-constraints.txt" \  'rlinf-libero==0.1.3'libero-download-assets --skip-existing
```

训练还需准备 SigLIP2、Gemma3、**PyTorch 格式的 π₀.₅ checkpoint** 和 LeRobot 数据集。前两个可以这样下载；Gemma 若要求授权，需先在 Hugging Face 接受条款并执行 `hf auth login`。

```
mkdir -p /data/workspaces/RLinf/checkpointshf download google/siglip2-so400m-patch14-224 \  --local-dir /data/workspaces/RLinf/checkpoints/siglip2-so400m-patch14-224hf download google/gemma-3-270m \  --local-dir /data/workspaces/RLinf/checkpoints/gemma-3-270m
```

然后按文档修改四个配置文件中的数据和模型路径：

- `recap_compute_returns.yaml`
- `recap_value_model_sft.yaml`
- `recap_compute_advantages.yaml`
- `cfg_rl_openpi.yaml`

**5090 的 32 GB 显存需要单独评估训练配置。** 先把 `actor.micro_batch_size` 从默认的 `32` 降到 `1` 或 `2`，开启梯度检查点做小规模测试；Step 4 的 π₀.₅ 全参数训练仍可能显存不足，需要冻结部分参数、LoRA 或 offload。安装成功与完整训练能否放入显存，是两项独立检查。