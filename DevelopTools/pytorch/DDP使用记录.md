#pytorch 


```python
# python原始多进程
import torch.multiprocessing as mp
# 用于将样本分发给各个GPU
import torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
#初始化和销毁整个进程组
from torch.distributed import init_process_group,destroy_process_group
```