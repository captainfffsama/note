#Isaacsim #爬坑 

[toc]

# AttributeError: module ‘omni.usd’ has no attribute ‘UsdContext’
**问题描述：**
使用的 pip install 安装是 isaacsim，然后直接执行 isaacsim 启动失败，报错：

```text
AttributeError: module ‘omni.usd’ has no attribute ‘UsdContext’
```

**解决方案：**
`rm -rf ~/path/to/python3.11/site-packages/isaacsim/extscache`

**参考连接：**
[AttributeError: module 'omni.usd' has no attribute 'UsdContext' - Omniverse / Isaac Sim - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/attributeerror-module-omni-usd-has-no-attribute-usdcontext/341881)