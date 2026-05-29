参考 <https://github.com/Dao-AILab/flash-attention/issues/1560#issuecomment-2798368648>

环境信息：

1. Torch:2.6.0+cu124
2. 显卡：5090

执行：

```bash

set FLASH_ATTN_CUDA_ARCHS=128 //Limits with your environment CUDA
set MAX_JOBS=16  //Limits your CPU depending to your CPU and RAM capacity

pip install flash-attn --no-build-isolation
```