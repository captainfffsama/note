---
title: "Pytorch 调整学习率：torch.optim.lr_scheduler.CosineAnnealingLR和CosineAnnealingWarmRestarts-CSDN博客"
source: "https://blog.csdn.net/weixin_44682222/article/details/122218046"
author:
published:
created: 2024-12-04
description: "文章浏览阅读2w次，点赞16次，收藏52次。一，torch.optim.lr_scheduler.CosineAnnealingLR参数说明：torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max,eta_min=0,last_epoch=-1)T_max：因为学习率是周期性变化的，该参数表示周期的1/2，也就是说，初始的学习率为a，经过2*T_max时间后，学习率经过了一个周期变化后还是a。eta_min：表示学习率可变化的最小值，默认为0在去噪实验中，将T_max的值设定为_cosineannealinglr和cosineannealingwarmrestarts"
tags:
  - "clippings"
---
最新推荐文章于 2024-09-03 09:11:02 发布

![](https://csdnimg.cn/release/blogv2/dist/pc/img/original.png)

[Kevin在成长](https://blog.csdn.net/weixin_44682222 "Kevin在成长") ![](https://csdnimg.cn/release/blogv2/dist/pc/img/newCurrentTime2.png) 于 2021-12-29 16:13:00 发布

版权声明：本文为博主原创文章，遵循 [CC 4.0 BY-SA](http://creativecommons.org/licenses/by-sa/4.0/) 版权协议，转载请附上原文出处链接和本声明。

一，[torch](https://so.csdn.net/so/search?q=torch&spm=1001.2101.3001.7020).optim.lr\_scheduler.CosineAnnealingLR  
参数说明：  
torch.optim.lr\_scheduler.CosineAnnealingLR(optimizer,T\_max,eta\_min=0,last\_epoch=-1)  
T\_max：因为学习率是周期性变化的，该参数表示周期的1/2，也就是说，初始的学习率为a，经过2\*T\_max时间后，学习率经过了一个周期变化后还是a。  
eta\_min：表示学习率可变化的最小值，默认为0  
![T_max设为20，总的epoch为150](https://i-blog.csdnimg.cn/blog_migrate/a5025b2e6d427f4f1c1f194505bb3bd9.png)

在去噪实验中，将T\_max的值设定为总的训练epoch，其学习率的变化如下图所示：  
![T_max取训练的epoch总数](https://i-blog.csdnimg.cn/blog_migrate/4232f29602dcbfd6a2c5fd3d458bc8c1.png)  
torch.optim.lr\_scheduler.CosineAnnealingWarmRestarts(optimizer,T\_0,T\_mult,eta\_min)

T\_0:表示学习率第一次周期性变化的epoch数  
T\_mult:如果设定为1，则是均匀的周期变化；如果设定为大于1，比如2，则学习率的变化周期是：  
如下图所示：T\_0为10，则表示第一个周期性变化轮次为10  
T\_mult为2  
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/fdbc4fec3ebd19bbbe76e8ceecc4352a.png)

测试代码

```python
import torch
from torchvision.models import AlexNet
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

model = AlexNet(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
#eta_min最小的学习率

scheduler =torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150) #torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=1e-6)
plt.figure()
x = list(range(150))
y = []
for epoch in range(1,151):
    optimizer.zero_grad()
    optimizer.step()
    print("第%d个epoch的学习率：%f" % (epoch,optimizer.param_groups[0]['lr']))
    scheduler.step()
    y.append(scheduler.get_lr()[0])

# 画出lr的变化    
plt.plot(x, y)
plt.xlabel("epoch")
plt.ylabel("lr")
plt.title("learning rate's curve changes as epoch goes on!")
plt.savefig("learning_rate_150_20.png")
```