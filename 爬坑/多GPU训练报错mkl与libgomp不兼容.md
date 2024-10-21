#pytorch #爬坑 

# 问题描述

使用 pytorch 编写的项目，比如 Ultralytics ，使用多 GPU 训练时，报错：

```bash
Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp-a34b3233.so.1 library.
Try to import numpy first or set the threading layer accordingly. Set MKL_SERVICE_FORCE_INTEL to force it.
```

# 解决方法

运行脚本之前，执行

```bash
export MKL_SERVICE_FORCE_INTEL=TRUE
```
