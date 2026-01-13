---
title: "解决50系显卡与Isaac Gym强化学习平台不兼容的问题"
source: "https://blog.csdn.net/m0_56706433/article/details/148902144"
author:
  - "[[m0_56706433]]"
published: 2025-06-26
created: 2026-01-08
description: "文章浏览阅读8.8k次，点赞108次，收藏177次。装旧版的torch也能成功，import torch和torch.cuda.is_available()都没问题，但一旦启动cuda开始计算（例如，在python终端输入torch.randn(2, 64, 512, device='cuda', dtype=torch.float16)）就会开始报错pytorch capability sm_120 is not compatible with the current PyTorch installtion…_50系显卡isaacgym"
tags:
  - "clippings"
---
50系显卡由于升级到sm120计算能力，只能适配新的torch版本。装旧版的torch也能成功，import torch和torch.cuda.is\_available()都没问题，但一旦启动cuda开始计算（例如，在python终端输入torch.randn(2, 64, 512, device='cuda', dtype=torch.float16)）就会开始报错pytorch capability sm\_120 is not compatible with the current PyTorch installtion…，归根结底就是旧版的torch无法适配新显卡sm120计算能力。

目前，适配50系显卡的torch安装参考教程：

（注意，按照这个方法安装将会因为python版本的问题，无法安装Isaac Gym！） [Nvidia 5070Ti 安装pytorch https://blog.csdn.net/weixin\_46535650/article/details/147213616?utm\_medium=distribute.pc\_relevant.none-task-blog-2~default~baidujs\_baidulandingword~default-0-147213616-blog-146114383.235^v43^pc\_blog\_bottom\_relevance\_base4&spm=1001.2101.3001.4242.1&utm\_relevant\_index=2](https://blog.csdn.net/weixin_46535650/article/details/147213616?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-147213616-blog-146114383.235^v43^pc_blog_bottom_relevance_base4&spm=1001.2101.3001.4242.1&utm_relevant_index=2 "Nvidia 5070Ti 安装pytorch")

然而适配50系显卡的torch硬性要求python版本大于等于3.9，这意味着3.9以下的旧代码无法适配。不巧的是，isaac gym由于nvidia已经停止维护了，最高只支持到3.8，与torch有着硬性冲突...

多次尝试无果后，看到大佬 [（文章链接）](https://blog.csdn.net/qq70654468/article/details/147704891 "（文章链接）") 通过编译2.3.1版本的torch源码的方式解决了兼容问题，看到了曙光。尝试了之后，发现真的能在python3.8安装能适配新显卡的pytorch，进而部署isaac gym。目前已经成功将isaac gym部署到5070ti和5090显卡上。流程如下。

**前置条件：**  
cuda已安装，版本大于等于12.8；显卡驱动已安装，版本大于等于555.x；电脑是50系显卡。本电脑操作系统版本是ubuntu20.04，其他操作系统版本能否成功未知。

```bash
nvcc --version    # 查看cuda版本，要求至少12.8

nvidia-smi        # 查看显卡驱动，要求至少555.x（这里的cuda版本不一定是你电脑的版本，以nvcc --version为准）
bash
```

**第一步：克隆pytorch源码到本地**

```bash
git clone https://github.com/pytorch/pytorch    # 1. 克隆主仓库

cd pytorch

git checkout v2.3.1                             # 2. 切换到 v2.3.1 版本

git submodule update --init --recursive         # 3. 递归初始化子模块
bash
```

强烈建议在此时将下载好的pytorch一整个文件夹压缩备份，后面编译失败或者出其他问题的时候方便恢复。

**第二步：创建conda虚拟环境，python版本3.8**

```bash
conda create -n isaacgym python=3.8

conda activate isaacgym
bash
```

**第三步：找到这几处，修改源码后编译**

1、打开pytorch/cmake/Modules\_CUDA\_fix/upstream/FindCUDA/select\_compute\_arch.cmake

找到227行

```python
....      

  elseif(${arch_name} STREQUAL "Hopper")

    set(arch_bin 9.0)

    set(arch_ptx 9.0)

  else()

    message(SEND_ERROR "Unknown CUDA Architecture Name ${arch_name} in CUDA_SELECT_NVCC_ARCH_FLAGS")

  endif()

....
python
```

改为

```python
....      

  elseif(${arch_name} STREQUAL "Hopper")

    set(arch_bin 9.0)

    set(arch_ptx 9.0)

  else()

    set(arch_bin 12.0)

    set(arch_ptx 12.0)

  endif()

....
python
```

2、打开pytorch/Dockerfile

找到60行

```python
TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \python
```

改为

```python
TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1 7.0+PTX 8.0 12.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \python
```

然后新打开一个终端，开始编译

```bash
conda activate isaacgym

cd pytorch

 

export USE_CUDA=1

export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0;12.0"

export MAX_JOBS=$(nproc)

 

python setup.py bdist_wheel    #编译开始
bash
```

编译比较耗时，部分项目需要编译很久，不是编译失败，在未报错之前请耐心等待。  
如果编译过程中电脑卡死（电脑无响应，鼠标拖动都非常卡），大概率是内存溢出。将上面命令中的 **export MAX\_JOBS=$(nproc)** 改为 **export MAX\_JOBS=5** 后重试。 中途每次编译失败，最好删掉当前的pytorch文件夹，然后解压之前备份的压缩包，回到初始状态。

> 如果您尝试多次之后，仍然遇到各种报错无法顺利编译，可以下载我编译好的torch-2.3.0a0+git63d5e92-cp38-cp38-linux\_x86\_64.whl文件（平台：i9+5070Ti，ubuntu20.04）
> 
> 下载方式：百度网盘（ [跳转连接](https://pan.baidu.com/s/1_I8Wrbx497QsnKje0pPzUw?pwd=bc5b "跳转连接") ）
> 
> 下载好之后，您可以直接跳过本步，继续往后。 **需要注意的是，这个.whl文件可能无法适用所有平台。** 目前已经在另一台i9+5090的电脑上试过，可以顺利安装。

**第四步：安装isaacgym**  
上一步编译完成后，会生成dist/torch-\*.whl这个pip安装编译好的whl文件。先不着急安装torch，因为在装torchvision时会自动安装对应版本的官方torch，这不适配我们的50显卡。 请妥善保管好之前自己编译的这个torch-\*.whl文件，因为后面一旦pytorch被某些包覆盖安装，我们需要重新pip install回到自己编译的这个版本。 因此我们需要在其他环境安装好之后，最后install torch。

```bash
conda activate isaacgym

#安装其他包

pip install torchvision==0.18.1

pip install torchaudio==2.3.1

pip install numpy==1.23.5
bash
```

下载 Isaac Gym Preview 4 仿真平台([下载链接](https://developer.nvidia.com/isaac-gym "下载链接"))，解压后进入 python 目录，使用 pip 安装

```bash
cd /home/xxxxx/IsaacGym_Preview_4_Package/isaacgym/python

conda activate isaacgym

pip install -e .
bash
```

运行 python/examples 目录下的例程，验证安装是否成功。

```bash
cd python/examples

conda activate isaacgym

python 1080_balls_of_solitude.py
bash
```

显示如下界面：

![](https://i-blog.csdnimg.cn/direct/4bd0eb125fc14be18e01047aeacdb3ad.png)

然后安装rsl\_rl库（请使用1.0.2版本）

```bash
git clone https://github.com/leggedrobotics/rsl_rl

cd rsl_rl

git checkout v1.0.2

conda activate isaacgym

pip install -e .
bash
```

**第五步：安装torch**  
进入到下载的pytorch源码文件夹，找到dist文件夹，找到torch-\*\*\*.whl，这就是编译好的文件。

```cobol
conda activate isaacgym

pip install torch-***.whl
```

> 遇到下列报错：
> 
> ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.  
> torchaudio 2.3.1 requires torch==2.3.1, but you have torch 2.3.0a0+git63d5e92 which is incompatible.  
> torchvision 0.18.1 requires torch==2.3.1, but you have torch 2.3.0a0+git63d5e92 which is incompatible.  
> 不用管，或者根据自己的需要重新安装torchaudio和torchvision。

**第六步：检查安装是否成功**

```bash
conda activate isaacgym

python

>>> import torch               #无报错

>>> torch.cuda.is_available()  #返回True

>>> print(torch.__version__)   #显示自己安装的版本

>>> torch.randn(2, 64, 512, device='cuda', dtype=torch.float16)     #无报错
bash
```

**第七步：测试isaacgym能否开始训练，以宇树的demo为例（ [网址跳转](https://support.unitree.com/home/zh/developer/rl_example "网址跳转") ）**

```bash
git clone https://github.com/unitreerobotics/unitree_rl_gym.git

conda activate isaacgym

pip install -e .

 

#一般这个时候之前安装的咱们魔改torch会被冲掉，重新安装一次

cd path/to/torch

pip install torch-***.whl

pip install numpy==1.23.5

 

cd legged_gym/scripts

python3  train.py --task=g1
bash
```

isaac gym 出现如下界面，则训练开始。恭喜，安装流程顺利结束！现在你可以在50系显卡上运行isaac gym啦！

![](https://i-blog.csdnimg.cn/direct/36537d5792124c378f8bccfde98bad3b.png)  

新人第一次写博客，如果有帮助，点个免费的赞吧～

实付 元

[使用余额支付](https://blog.csdn.net/m0_56706433/article/details/)

点击重新获取

扫码支付

钱包余额 0

抵扣说明：

1.余额是钱包充值的虚拟货币，按照1:1的比例进行支付金额的抵扣。  
2.余额无法直接购买下载，可以购买VIP、付费专栏及课程。

[余额充值](https://i.csdn.net/#/wallet/balance/recharge)

举报

 [![](https://csdnimg.cn/release/blogv2/dist/pc/img/toolbar/Group.png) 点击体验  
DeepSeekR1满血版](https://ai.csdn.net/chat?utm_source=cknow_pc_blogdetail&spm=1001.2101.3001.10583) 隐藏侧栏 ![程序员都在用的中文IT技术交流社区](https://g.csdnimg.cn/side-toolbar/3.6/images/qr_app.png)

程序员都在用的中文IT技术交流社区

![专业的中文 IT 技术社区，与千万技术人共成长](https://g.csdnimg.cn/side-toolbar/3.6/images/qr_wechat.png)

专业的中文 IT 技术社区，与千万技术人共成长

![关注【CSDN】视频号，行业资讯、技术分享精彩不断，直播好礼送不停！](https://g.csdnimg.cn/side-toolbar/3.6/images/qr_video.png)

关注【CSDN】视频号，行业资讯、技术分享精彩不断，直播好礼送不停！

客服 返回顶部

![](https://i-blog.csdnimg.cn/direct/4bd0eb125fc14be18e01047aeacdb3ad.png) ![](https://i-blog.csdnimg.cn/direct/36537d5792124c378f8bccfde98bad3b.png)