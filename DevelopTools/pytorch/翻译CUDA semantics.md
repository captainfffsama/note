#翻译

原文参考:[1.12pytorch](https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics)

[toc]

# CUDA 语义
[`torch.cuda`](https://pytorch.org/docs/stable/cuda.html#module-torch.cuda "torch.cuda") 被用于设置和进行CUDA操作. 它将持续跟踪当前选择的GPU,使用者分配的所有CUDA张量默认都会创建在这个GPU上.使用 [`torch.cuda.device`](https://pytorch.org/docs/stable/generated/torch.cuda.device.html#torch.cuda.device "torch.cuda.device") 这个上下文管理器可以改变当前选择的GPU.

注意一旦张量被分配,那么您对其进行的操作都是设备无关的,操作结果总会和被操作张量在同一GPU上.

默认情况下跨GPU操作是不允许的,但 [`copy_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.copy_.html#torch.Tensor.copy_ "torch.Tensor.copy_") 以及诸如 [`to()`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to "torch.Tensor.to") 和 [`cuda()`](https://pytorch.org/docs/stable/generated/torch.Tensor.cuda.html#torch.Tensor.cuda "torch.Tensor.cuda") 等其他一些类拷贝的操作是例外.除非启用点对点的内存访问,否则任何试图分布在不同设备上的张量操作都将引发错误.

下面用一个小例子说明这点:

```python
cuda = torch.device('cuda')     # Default CUDA device
cuda0 = torch.device('cuda:0')
cuda2 = torch.device('cuda:2')  # GPU 2 (these are 0-indexed)

x = torch.tensor([1., 2.], device=cuda0)
# x.device is device(type='cuda', index=0)
y = torch.tensor([1., 2.]).cuda()
# y.device is device(type='cuda', index=0)

with torch.cuda.device(1):
    # allocates a tensor on GPU 1
    a = torch.tensor([1., 2.], device=cuda)

    # transfers a tensor from CPU to GPU 1
    b = torch.tensor([1., 2.]).cuda()
    # a.device and b.device are device(type='cuda', index=1)

    # You can also use ``Tensor.to`` to transfer a tensor:
    b2 = torch.tensor([1., 2.]).to(device=cuda)
    # b.device and b2.device are device(type='cuda', index=1)

    c = a + b
    # c.device is device(type='cuda', index=1)

    z = x + y
    # z.device is device(type='cuda', index=0)

    # even within a context, you can specify the device
    # (or give a GPU index to the .cuda call)
    d = torch.randn(2, device=cuda2)
    e = torch.randn(2).to(cuda2)
    f = torch.randn(2).cuda(cuda2)
    # d.device, e.device, and f.device are all device(type='cuda', index=2)
```

## TensorFloat-32(TF32) on Ampere devices

从 PyTorch 1.7 开始,我们引入来一个新的标志叫`allow_tf32`.在 Pytorch 1.7 到 1.11 版本,该标志默认为 `True`,从 1.12版本之后,默认为 `False` . 该标志控制 Pytorch 是否可以使用 TensorFloat32 张量核心(Ampere架构之后的GPU可用),用于计算 matmul 和卷积.

TF32 张量核心被用来在 torch.float32 张量上进行 matmul 和卷积操作时获得更好的性能.其实现方法是对最后10位尾数进行四舍五入,以 FP32 的精度来累积结果,保留 FP32 的动态范围.

matmuls 和卷积操作是分别进行控制的,它们对应的标志如下:

```python
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
```

注意除开 matmuls 和卷积操作自身,内部使用 matmul 和卷积的函数和 nn 模块也会受到影响,比如 `nn.Linear`,`nn.Conv*`,cdist,tensordot,affine grid 和 grid sample, adaptive log sofrmax ,GRU, LSTM.

要了解精度和速度，请参阅下面的示例代码:

```python
a_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
b_full = torch.randn(10240, 10240, dtype=torch.double, device='cuda')
ab_full = a_full @ b_full
mean = ab_full.abs().mean()  # 80.7277

a = a_full.float()
b = b_full.float()

# Do matmul at TF32 mode.
torch.backends.cuda.matmul.allow_tf32 = True
ab_tf32 = a @ b  # takes 0.016s on GA100
error = (ab_tf32 - ab_full).abs().max()  # 0.1747
relative_error = error / mean  # 0.0022

# Do matmul with TF32 disabled.
torch.backends.cuda.matmul.allow_tf32 = False
ab_fp32 = a @ b  # takes 0.11s on GA100
error = (ab_fp32 - ab_full).abs().max()  # 0.0031
relative_error = error / mean  # 0.000039
```

在上述例子中,我们可以发现启用 TF32,与双精度相比,可以获得近7倍的加速,而相对误差约为2个数量级.若需使用完全的 FP32 精度,可以通过以下方式禁用 TF32: 
```python
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

C++ 中关闭 TF32 的方式如下:

```cpp
at::globalContext().setAllowTF32CuBLAS(false);
at::globalContext().setAllowTF32CuDNN(false);
```

更多 TF32 的信息,可参见:

*   [TensorFloat-32](https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/)
    
*   [CUDA 11](https://devblogs.nvidia.com/cuda-11-features-revealed/)
    
*   [Ampere architecture](https://devblogs.nvidia.com/nvidia-ampere-architecture-in-depth/)
    

## Reduced Precision Reduction in FP16 GEMMs

fp16 GEMMs 可通过降低中间精度来达到(例如,在fp16而非fp32).这种对精度选择性的降低可以使得在某些工作情况(k维巨大的张量)和GPU体系结构上获得更高的性能,代价是数值精度可能溢出.

以下是在 V100 上测试的基准:

```shell
[--------------------------- bench_gemm_transformer --------------------------]
      [  m ,  k  ,  n  ]    |  allow_fp16_reduc=True  |  allow_fp16_reduc=False
1 threads: --------------------------------------------------------------------
      [4096, 4048, 4096]    |           1634.6        |           1639.8
      [4096, 4056, 4096]    |           1670.8        |           1661.9
      [4096, 4080, 4096]    |           1664.2        |           1658.3
      [4096, 4096, 4096]    |           1639.4        |           1651.0
      [4096, 4104, 4096]    |           1677.4        |           1674.9
      [4096, 4128, 4096]    |           1655.7        |           1646.0
      [4096, 4144, 4096]    |           1796.8        |           2519.6
      [4096, 5096, 4096]    |           2094.6        |           3190.0
      [4096, 5104, 4096]    |           2144.0        |           2663.5
      [4096, 5112, 4096]    |           2149.1        |           2766.9
      [4096, 5120, 4096]    |           2142.8        |           2631.0
      [4096, 9728, 4096]    |           3875.1        |           5779.8
      [4096, 16384, 4096]   |           6182.9        |           9656.5
(times in microseconds).
```

若需完全降低精度,可以使用以下方法禁用 fp16 GMMs 的精度降低:

```python
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
```

C++ 中,可以这样做

```cpp
at::globalContext().setAllowFP16ReductionCuBLAS(false);
```
## 异步执行

GPU 操作默认是异步的.当通过GPU 调用函数时,操作会排队到特定设备上,但不一定是要等到之后才执行.这使得我们可以并行进行更多计算,包括CPU操作和一些其他的GPU操作.

通常而言,这些异步计算对于调用者是无感知的.因为:
(1) 每个设备都会依照队列顺序执行.
(2) PyTorch 在跨设备的复制操作时都会进行必要的数据同步,因此所有操作看上去像是同步的.

通过设置环境变量`CUDA_LAUNCH_BLOCKING=1` 可以强制同步变量. 当GPU出现错误时这个变量可以很方便调试.(对于异步执行,只有到实际执行操作之后才会报告错误,因此堆栈跟踪不会显示请求的位置)

异步计算还可能导致未经同步的时间测量的不准确.为获得精确的时间测量,在记录时间之前应该调用[`torch.cuda.synchronize()`](https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html#torch.cuda.synchronize "torch.cuda.synchronize") 或是[`torch.cuda.Event`](https://pytorch.org/docs/stable/generated/torch.cuda.Event.html#torch.cuda.Event "torch.cuda.Event"):

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

# Run some things here

end_event.record()
torch.cuda.synchronize()  # Wait for the events to be recorded!
elapsed_time_ms = start_event.elapsed_time(end_event)
```

除开诸如[`to()`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to "torch.Tensor.to") 和 [`copy_()`](https://pytorch.org/docs/stable/generated/torch.Tensor.copy_.html#torch.Tensor.copy_ "torch.Tensor.copy_")等函数可以使用显式的`non_blocking`参数来让调用者绕过非必要时候的同步. CUDA 流也可以绕过同步.


### CUDA 流
[CUDA stream](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams) 是特定设备的线性执行序列,通常无需显式的创建.默认情况下每个设备都有自己默认的"流".


每个流中的操作按照其创建的顺序进行序列化,但不同流的操作可以以任何相对的顺序并发执行,除非使用显式的同步函数,比如 [`synchronize()`](https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html#torch.cuda.synchronize "torch.cuda.synchronize") 或者 [`wait_stream()`](https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html#torch.cuda.Stream.wait_stream "torch.cuda.Stream.wait_stream"). 下面展示一个错误示例:

```python
cuda = torch.device('cuda')
s = torch.cuda.Stream()  # Create a new stream.
A = torch.empty((100, 100), device=cuda).normal_(0.0, 1.0)
with torch.cuda.stream(s):
    # sum() may start execution before normal_() finishes!
    B = torch.sum(A)
```

当"当前流"是默认流时,PyTorch 会在数据移动时自动进行必要的同步.但当使用非默认流时,用户需要自己确保正确的同步.

### Stream semantics of backward passes

同一流上的反传CUDA 操作将被用于其对应的前传操作上.若前向传播在不同流上运行独立的操作,这将有助于反向传播利用相同的并行性.

The stream semantics of a backward call with respect to surrounding ops are the same as for any other call.反向传播将在内部插入同步来确保当反传操作在多个流上进行时,会和前向传播一样有相同的并行性.具体而言,当调用[`autograd.backward`](https://pytorch.org/docs/stable/generated/torch.autograd.backward.html#torch.autograd.backward "torch.autograd.backward"), [`autograd.grad`](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html#torch.autograd.grad "torch.autograd.grad"), [`tensor.backward`](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html#torch.Tensor.backward "torch.Tensor.backward")时,可以提供一个 CUDA 张量作为初始梯度,比如:[`autograd.backward(..., grad_tensors=initial_grads)`](https://pytorch.org/docs/stable/generated/torch.autograd.backward.html#torch.autograd.backward "torch.autograd.backward"), [`autograd.grad(..., grad_outputs=initial_grads)`](https://pytorch.org/docs/stable/generated/torch.autograd.grad.html#torch.autograd.grad "torch.autograd.grad"),  [`tensor.backward(..., gradient=initial_grad)`](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html#torch.Tensor.backward "torch.Tensor.backward") 等用法.其内部操作步骤是:

1.  填充初始梯度(可选)
    
2.  调用反向传播
    
3.  使用梯度
    

任何操作的组合都有相同的流语义关系:

```python
s = torch.cuda.Stream()

# Safe, grads are used in the same stream context as backward()
with torch.cuda.stream(s):
    loss.backward()
    use grads

# Unsafe
with torch.cuda.stream(s):
    loss.backward()
use grads

# Safe, with synchronization
with torch.cuda.stream(s):
    loss.backward()
torch.cuda.current_stream().wait_stream(s)
use grads

# Safe, populating initial grad and invoking backward are in the same stream context
with torch.cuda.stream(s):
    loss.backward(gradient=torch.ones_like(loss))

# Unsafe, populating initial_grad and invoking backward are in different stream contexts,
# without synchronization
initial_grad = torch.ones_like(loss)
with torch.cuda.stream(s):
    loss.backward(gradient=initial_grad)

# Safe, with synchronization
initial_grad = torch.ones_like(loss)
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    initial_grad.record_stream(s)
    loss.backward(gradient=initial_grad)
```

#### BC note: Using grads on the default stream

在 1.9或是更早的 PyTorch,自动微分引擎总会将默认流和所有反传操作同步,其模式如下:

```python
with torch.cuda.stream(s):
    loss.backward()
use grads
```

此时只要 `use grads` 是在默认流中,那就是安全的.但在当前 PyTorch,这种模式将不在安全.当 `backward()` 和 `use grads` 在不同流的上下文中,用户必须同步流:

```python
with torch.cuda.stream(s):
    loss.backward()
torch.cuda.current_stream().wait_stream(s)
use grads
```

即便 `use grads` 是在默认流中执行的.

## Memory management

PyTorch 使用显存缓存分配器来加速显存分配.这使得 PyTorch 可以在没有设备同步的情况下快速释放显存.这使得即使分配器未使用显存,但在 `nvidia-smi` 中仍然显示已使用.用户可以使用[`memory_allocated()`](https://pytorch.org/docs/stable/generated/torch.cuda.memory_allocated.html#torch.cuda.memory_allocated "torch.cuda.memory_allocated") 和 [`max_memory_allocated()`](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html#torch.cuda.max_memory_allocated "torch.cuda.max_memory_allocated") 来监视张量的显存占用情况,使用[`memory_reserved()`](https://pytorch.org/docs/stable/generated/torch.cuda.memory_reserved.html#torch.cuda.memory_reserved "torch.cuda.memory_reserved") 和 [`max_memory_reserved()`](https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_reserved.html#torch.cuda.max_memory_reserved "torch.cuda.max_memory_reserved") 来监视显存缓存分配器管理的所有显存.调用[`empty_cache()`](https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html#torch.cuda.empty_cache "torch.cuda.empty_cache") 可以释放所有 PyTorch 缓存的**未使用**的显存,使用其他 GPU 应用可以使用他们.但张量已经占用的显存将无法释放,所以这个方法不能增加 PyTorch 的可用显存量.
d
进阶用户可以使用[`memory_stats()`](https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html#torch.cuda.memory_stats "torch.cuda.memory_stats") 来获得更加全面的显存基准测试.使用[`memory_snapshot()`](https://pytorch.org/docs/stable/generated/torch.cuda.memory_snapshot.html#torch.cuda.memory_snapshot "torch.cuda.memory_snapshot")则可以获得完整的显存分配器快照,这将有助于用户理解用户代码中底层的显存分配模式.

缓存分配器的使用会干扰诸如 `cuda-memcheck` 等显存检查工具.若使用 `cuda-memcheck` 来 debug 显存错误,需在用户环境中设置 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 来禁用缓存.

环境变量`PYTORCH_CUDA_ALLOC_CONF` 可以用来控制缓存分配器的行为.其格式为:`PYTORCH_CUDA_ALLOC_CONF=<option>:<value>,<option2><value2>...`.可用选项如下:

- `max_split_size_mb` 可防止分配器拆分大于此阈值的块(单位MB).这个可以防止显存碎片,在显存不耗尽的情况下进行一些边界工作负载.根据分配模式的不同，性能成本可以从“零”到“实质性”不等.默认值是无限的，也就是说，所有的块都可以拆分.[`memory_stats()`](https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html#torch.cuda.memory_stats "torch.cuda.memory_stats") 和[`memory_summary()`](https://pytorch.org/docs/stable/generated/torch.cuda.memory_summary.html#torch.cuda.memory_summary "torch.cuda.memory_summary") 对于调优是很有用的.当因为显存不足,且显示有大量显存碎片而导致工作中断时,此选项可以作为最后解决手段.

- `roundup_power2_divisions` 有助于请求分配四舍五入为最近仅2次幂的块,从而更好的利用.在当前的 CUDA 缓存分配器中,块大小是多个四舍五入到512的块,对于一些要求大小较小的块可以较好的工作.然而,当附近需要较大块分配时,这可能是低效的.因为这些需求块大小不一,难以重用.这就可能产生大量未使用的块,进而造成 GPU 显存浪费.该选项分配的块大小为四舍五入的2次幂上下界最近的分割除数块.比如我们需要大小为1200,而1200 最近的2次幂上下界为1024和2048,假设设定的分割数为4,那么1024和2048之间分割点为,1024,1280,1536和1792.最终选取最靠近1200上限的值 1280 作为2次幂分割的上届.

- `garbage_collection_threshold` 有助于主动回收未使用的 GPU 内存,来避免昂贵的全面同步和回收操作(release_cached_blocks),这可能不利于关键延迟 GPU 应用程序(例如，服务器)。在设置这个阈值(例如0.8)后，如果 GPU 内存容量使用超过阈值(即分配给 GPU 应用程序的总内存的80%) ，分配器将开始回收 GPU 内存块。该算法倾向于首先释放旧的和未使用的块，以避免释放正在被积极重用的块。阈值应该在大于0.0和小于1.0之间。
    

## cuFFT 计划缓存

对于每个 CUDA 设备, cuFFT 计划的 LRU 缓存可用来加速在相同配置和形状的 CUDA 张量上反复运行的 FFT 方法(例如[`torch.fft.fft()`](https://pytorch.org/docs/stable/generated/torch.fft.fft.html#torch.fft.fft "torch.fft.fft")).因为 cuFFT 可能会分配 GPU 显存,这些显存有最大的容量.

您可以使用以下 API 控制和查询当前设备缓存的属性:

- `torch.backends.cuda.cufft_plan_cache.max_size` 该值返回缓存的容量大小( CUDA 10 以后默认4096,老版本 CUDA 1023).修改此值会直接修改容量.

- `torch.backends.cuda.cufft_plan_cache.size` 返回当前缓存的计划数.     
    
- `torch.backends.cuda.cufft_plan_cache.clear()` 清理缓存.
    
要控制和查询非默认设备的缓存,可以使用[`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device "torch.device") 或是设备索引作为`torch.backends.cuda.cufft_plan_cache` 对象的索引来获取上述属性.例如,要设置设备`1`的缓存容量,可以使用 `torch.backends.cuda.cufft_plan_cache[1].max_size = 10`.

## Just-in-Time 编译

当在 CUDA 张量上进行类似 torch.special.zeta 操作时,PyTorch 会进行即时编译.这种编译可能非常耗时(根据硬件和软件的不同，最多几秒钟) ，而且对于单个操作符可能会多次发生，因为许多 PyTorch 操作符实际上是从各种内核中选择的，每个内核必须根据输入编译一次。每个进程进行一次编译，如果使用内核缓存，则只进行一次编译。

默认情况下,PyTorch 会在\$XDG_CACHE_HOME/torch/kernels 下创建一个内核缓存,若 XDG_CACHE_HOME已经定义 且\$HOME/.cache/torch/kernels 不存在.(Windows 暂不支持). USE_PYTORCH_KERNEL_CACHE 和 PYTORCH_KERNEL_CACHE_PATH 这两个环境变量可以直接控制缓存行为,前者设置为0将不使用缓存,后者若设置,则该路径将替代默认路径用作内核缓存.

## 最佳实践

### 设备无关代码

由于 PyTorch 的结构，你可能需要显式地编写与设备无关的(CPU 或 GPU)代码，例如创建一个新的张量作为递归神经网络的初始隐藏状态。

第一步是决定是否使用 GPU,常见的方式是使用 python `argparse` 模块来读入用户参数并使用一个标志来禁用CUDA,并与 [`is_available()`](https://pytorch.org/docs/stable/generated/torch.cuda.is_available.html#torch.cuda.is_available "torch.cuda.is_available") 结合.下面例子中, `args.device`将作用于[`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device "torch.device") 来确定将张量移动到 CPU 还是 CUDA.

```python
import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')
```

通过`args.device`,我们可以在想要的设备上创建张量.

```python
x = torch.empty((8, 42), device=args.device)
net = Network().to(device=args.device)
```

这可以在很多情况下用于设备无关的代码,下面是一个 dataloader 的示例:

```python
cuda0 = torch.device('cuda:0')  # CUDA GPU 0
for i, x in enumerate(train_loader):
    x = x.to(cuda0)
```

多 GPU 设备上,可以通过使用`CUDA_VISIBLE_DEVICES`环境变量来管理哪些GPU PyTorch可用.如上所述,手动管理在哪个GPU上创建张量的最好方式还是使用[`torch.cuda.device`](https://pytorch.org/docs/stable/generated/torch.cuda.device.html#torch.cuda.device "torch.cuda.device") 上下文管理器.

```python
print("Outside device is 0")  # On device 0 (default in most scenarios)
with torch.cuda.device(1):
    print("Inside device is 1")  # On device 1
print("Outside device is still 0")  # On device 0
```

想在同一设备上创建一个和已有张量相同类型的张量,可以使用 `torch.Tensor.new_*` 方法(参见[`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "torch.Tensor")).而之前提到的`torch.*`等工厂函数([Creation Ops](https://pytorch.org/docs/stable/torch.html#tensor-creation-ops))则取决于当前的 GPU 上下文和传入的属性参数,`torch.Tensor.new_*` 方法保留设备和张量的其他属性.

建议的做法是在创建模块时,在前项传播内部创建新的张量.

```python
cuda = torch.device('cuda')
x_cpu = torch.empty(2)
x_gpu = torch.empty(2, device=cuda)
x_cpu_long = torch.empty(2, dtype=torch.int64)

y_cpu = x_cpu.new_full([3, 2], fill_value=0.3)
print(y_cpu)

    tensor([[ 0.3000,  0.3000],
            [ 0.3000,  0.3000],
            [ 0.3000,  0.3000]])

y_gpu = x_gpu.new_full([3, 2], fill_value=-5)
print(y_gpu)

    tensor([[-5.0000, -5.0000],
            [-5.0000, -5.0000],
            [-5.0000, -5.0000]], device='cuda:0')

y_cpu_long = x_cpu_long.new_tensor([[1, 2, 3]])
print(y_cpu_long)

    tensor([[ 1,  2,  3]])
```

[`ones_like()`](https://pytorch.org/docs/stable/generated/torch.ones_like.html#torch.ones_like "torch.ones_like") 或 [`zeros_like()`](https://pytorch.org/docs/stable/generated/torch.zeros_like.html#torch.zeros_like "torch.zeros_like") 可以很方便的创建一个和已有张量一样属性的新张量,并使用1或者0填充(这类方法依然保留[`torch.device`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device "torch.device") 和 [`torch.dtype`](https://pytorch.org/docs/stable/tensor_attributes.html#torch.dtype "torch.dtype") 参数).

```python
x_cpu = torch.empty(2, 3)
x_gpu = torch.empty(2, 3)

y_cpu = torch.ones_like(x_cpu)
y_gpu = torch.zeros_like(x_gpu)
```

### 使用固定内存缓冲区

**警告**
这是一个进阶技巧。如果过度使用固定内存，在内存不足时可能会导致严重问题，您应该意识到固定通常是一种昂贵的操作。

通过锁页内存将主机数据考入到 GPU 要比常规方法快得多.CPU 张量和存储都有一个公开的方法 [`pin_memory()`](https://pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory "torch.Tensor.pin_memory"),该方法返回一个对象的拷贝,这个对象将数据放在固定区域.

此外,一旦你固定来张量或者是存储,就可以使用异步 GPU 拷贝.只需给[`to()`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to "torch.Tensor.to") 或[`cuda()`](https://pytorch.org/docs/stable/generated/torch.Tensor.cuda.html#torch.Tensor.cuda "torch.Tensor.cuda") 方法传递一个额外的`non_blocking=True`参数.这可以使得数据传输和计算重叠起来.

在[`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader "torch.utils.data.DataLoader") 构造函数中传递`pin_memory=True`参数可以将批量数据放置在锁页内存中.

### 使用 nn.parallel.DistributedDataParallel 替代 nn.DataParallel
大多数涉及批处理输入和多个 GPU 的用例应该默认使用[`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel")来利用多个 GPU。

通过[`multiprocessing`](https://pytorch.org/docs/stable/multiprocessing.html#module-torch.multiprocessing "torch.multiprocessing") 使用 CUDA 模型需要注意一点,非能够精确地满足数据处理的要求，否则你的程序很可能会出现错误或未定义行为。

即使只有一个节点,也推荐使用[`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 替代 [`DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel "torch.nn.DataParallel") 进行多 GPU 训练.

[`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 和 [`DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel "torch.nn.DataParallel") 区别在于: [`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel") 进行多处理时,为每个 GPU 均创建一个进程,而 [`DataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html#torch.nn.DataParallel "torch.nn.DataParallel") 使用多线程.多进程处理时,每个 GPU 都有自己的专用进程，从而避免了 Python 解释器的 GIL 所带来的性能开销。 

若你使用[`DistributedDataParallel`](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel "torch.nn.parallel.DistributedDataParallel"),可以使用 `torch.distributed.launch` 来启动你的程序,参见[Third-party backends](https://pytorch.org/docs/stable/distributed.html#distributed-launch).

## CUDA Graphs

CUDA 图是 CUDA 流及其相关流执行工作的记录(主要记录内核和参数).关于 CUDA API 的一般原则和细节可参见:[Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) 和 [Graphs section](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)

Pytorch 支持使用[stream capture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture) 来构建 CUDA 图,这种方法将 CUDA 流设置为 __捕获模式__.发布到捕获流的 CUDA 工作实际上并不运行在GPU上,而是记录在一个图里.

在捕获之后,可以在需要时多次__启动__图使得 GPU 工作(类似宏操作).每次重播将在相同的核上使用相同的参数.对于指针参数，这意味着使用相同的内存地址。通过在每次重播之前用新数据(例如来自新批处理的数据)填充输入内存，您可以在新数据上重新运行相同的工作。

### 为何使用 CUDA Graphs

重放图会牺牲即时执行的灵活性,但是会**大大减少CPU负载**.由于图的参数内核都是固定的,因此重播图可以跳过所有层的参数设置和内核调度,包括 python C++和 CUDA驱动程序的开销.重播图的原理是调用[cudaGraphLaunch](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__GRAPH.html#group__CUDART__GRAPH_1g1accfe1da0c605a577c22d9751a09597) 将整个图工作提交到 GPU.重播的方式使得内核在GPU上执行得更快一些,但主要好处还是减轻了 CPU 工作负载.

若模型是完全或者部分图安全的(这通常意味着静态形状或者是静态控制流,其他约束参见[constraints](#capture-constraints))且你怀疑模型运行时某种程度上受到了CPU的限制,你应该尝试 CUDA 图.

### PyTorch API

警告

这个 API 是beta版,将来可能会改变.

通过[`torch.cuda.CUDAGraph`](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph "torch.cuda.CUDAGraph") 类和[`torch.cuda.graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph"),[`torch.cuda.make_graphed_callables`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 上下文管理器可以获得整个图.

[`torch.cuda.graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph") 是一个简单且通用的上下文管理器,可以在其上下文中捕获 CUDA 的工作.在捕获之前,需要运行一些即时迭代工作来预热要捕获的工作负载.预热在测流中进行.因为图在每次重播中读写相同的地址,故而必须维护对张量的长期引用,这些张量在捕获期间保存输入和输出数据.在一个新输入的数据上运行一个图,将新数据拷贝到捕获的输入张量上,然后重播图,从捕获的输出张量上读取新的输出,示例:

```python
g = torch.cuda.CUDAGraph()

# Placeholder input used for capture
static_input = torch.empty((5,), device="cuda")

# Warmup before capture
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(3):
        static_output = static_input * 2
torch.cuda.current_stream().wait_stream(s)

# Captures the graph
# To allow capture, automatically sets a side stream as the current stream in the context
with torch.cuda.graph(g):
    static_output = static_input * 2

# Fills the graph's input memory with new data to compute on
static_input.copy_(torch.full((5,), 3, device="cuda"))
g.replay()
# static_output holds the results
print(static_output)  # full of 3 * 2 = 6

# Fills the graph's input memory with more data to compute on
static_input.copy_(torch.full((5,), 4, device="cuda"))
g.replay()
print(static_output)  # full of 4 * 2 = 8
```

参见 [Whole-network capture](#whole-network-capture), [Usage with torch.cuda.amp](#graphs-with-amp),  [Usage with multiple streams](#multistream-capture) 获取进阶模式.

[`make_graphed_callables`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 更加复杂. [`make_graphed_callables`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 接受 python 函数和 [`torch.nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module"). 对于每个传递给他的模块, 它会分别创建前向传播和反传工作图. 参见 [Partial-network capture](#partial-network-capture).

#### Constraints 

如果不违反以下任何约束，则可以捕获一组操作。

在[`torch.cuda.graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph") 上下文中的所有工作和所有传给[`torch.cuda.make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 的可调用的前传反传工作都是遵循以下约束的.

违反下述中任何一条可能导致运行错误:

- 捕获必须在非默认流上(仅在使用[`CUDAGraph.capture_begin`](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_begin "torch.cuda.CUDAGraph.capture_begin") 和 [`CUDAGraph.capture_end`](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph.capture_end "torch.cuda.CUDAGraph.capture_end") 才需要考虑这个问题,[`graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph") 和 [`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 会设置一个支流)

- 禁止 CPU 和GPU同步的操作（比如调用 `.item()`）.
    
- 允许 CUDA RNG 操作，但是必须使用默认生成器。比如显式的构建一个[`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator "torch.Generator")  实例，并将其作为 `generator` 参数传递给 RNG 函数是不行的。

违反任何这些可能会导致无声的数值错误或不确定的行为：

- 一个进程内，同一时间仅能进行一个捕获。
    
- 在捕获进行期间，此进程中（在任何线程上）不得运行未捕获的CUDA工作。
    
- CPU 工作不能捕获。若捕获的操作里包含 CPU 工作，重播时将省略该工作。
    
- 每次回放读写相同的内存地址（虚拟的）。
    
- 禁止动态控制流（无论 CPU还是GPU）。
    
- 禁止动态形状。图假设在每次回放捕获的操作序列中的张量都有相同尺寸和层数。
    
- 一个捕获中允许使用多个流,但是有一些[限制](#multistream-capture)
    

#### Non-constraints[](#non-constraints)

- 一旦捕获了,图可以在任何流上回放.
    

### Whole-network capture[](#whole-network-capture)

若你的整个网络都可以捕获,那么你可以捕获并重播整个迭代.

```python
N, D_in, H, D_out = 640, 4096, 2048, 1024
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.Dropout(p=0.2),
                            torch.nn.Linear(H, D_out),
                            torch.nn.Dropout(p=0.1)).cuda()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Placeholders used for capture
static_input = torch.randn(N, D_in, device='cuda')
static_target = torch.randn(N, D_out, device='cuda')

# warmup
# Uses static_input and static_target here for convenience,
# but in a real setting, because the warmup includes optimizer.step()
# you must use a few batches of real data.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(static_input)
        loss = loss_fn(y_pred, static_target)
        loss.backward()
        optimizer.step()
torch.cuda.current_stream().wait_stream(s)

# capture
g = torch.cuda.CUDAGraph()
# Sets grads to None before capture, so backward() will create
# .grad attributes with allocations from the graph's private pool
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    static_y_pred = model(static_input)
    static_loss = loss_fn(static_y_pred, static_target)
    static_loss.backward()
    optimizer.step()

real_inputs = [torch.rand_like(static_input) for _ in range(10)]
real_targets = [torch.rand_like(static_target) for _ in range(10)]

for data, target in zip(real_inputs, real_targets):
    # Fills the graph's input memory with new data to compute on
    static_input.copy_(data)
    static_target.copy_(target)
    # replay() includes forward, backward, and step.
    # You don't even need to call optimizer.zero_grad() between iterations
    # because the captured backward refills static .grad tensors in place.
    g.replay()
    # Params have been updated. static_y_pred, static_loss, and .grad
    # attributes hold values from computing on this iteration's data.
```

### 部分网络捕获[](#partial-network-capture)

若你的网络有不安全的捕获(比如动态控制流,动态形状,CPU同步,基础的CPU端逻辑),你可以即时运行非安全部分,使用[`torch.cuda.make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables")来图化捕获安全的部分.

默认情况下,[`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 返回的可调用对象是可自动微分的,可以用于训练迭代,替换掉你传入的[`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module") 函数.

[`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 会在内部创建 [`CUDAGraph`](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph "torch.cuda.CUDAGraph") 对象,运行预热迭代,并根据需要维护静态的输入输出.与[`torch.cuda.graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph") 不同,你无需手动维护这些事情.

在下列示例中,依赖数据的动态控制流意味着网络不可以端到端的捕获,[`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") 允许我们捕获并以图运行来图安全的部分,无论它们是什么:

```python
N, D_in, H, D_out = 640, 4096, 2048, 1024

module1 = torch.nn.Linear(D_in, H).cuda()
module2 = torch.nn.Linear(H, D_out).cuda()
module3 = torch.nn.Linear(H, D_out).cuda()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(chain(module1.parameters(),
                                  module2.parameters(),
                                  module3.parameters()),
                            lr=0.1)

# Sample inputs used for capture
# requires_grad state of sample inputs must match
# requires_grad state of real inputs each callable will see.
x = torch.randn(N, D_in, device='cuda')
h = torch.randn(N, H, device='cuda', requires_grad=True)

module1 = torch.cuda.make_graphed_callables(module1, (x,))
module2 = torch.cuda.make_graphed_callables(module2, (h,))
module3 = torch.cuda.make_graphed_callables(module3, (h,))

real_inputs = [torch.rand_like(x) for _ in range(10)]
real_targets = [torch.randn(N, D_out, device="cuda") for _ in range(10)]

for data, target in zip(real_inputs, real_targets):
    optimizer.zero_grad(set_to_none=True)

    tmp = module1(data)  # forward ops run as a graph

    if tmp.sum().item() > 0:
        tmp = module2(tmp)  # forward ops run as a graph
    else:
        tmp = module3(tmp)  # forward ops run as a graph

    loss = loss_fn(tmp, target)
    # module2's or module3's (whichever was chosen) backward ops,
    # as well as module1's backward ops, run as graphs
    loss.backward()
    optimizer.step()
```

### 使用 torch.cuda.amp[](#usage-with-torch-cuda-amp)

对于典型的优化器,[`GradScaler.step`](https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.step "torch.cuda.amp.GradScaler.step") 同步 CPU 和GPU,但这在捕获期间是禁止的.为了避免错误,可以使用[部分网络捕获](#partial-network-capture),或者仅捕获前传反传损失部分(若这些部分是捕获安全的),而不捕获优化器优化部分.

```python
# warmup
# In a real setting, use a few batches of real data.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            y_pred = model(static_input)
            loss = loss_fn(y_pred, static_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
torch.cuda.current_stream().wait_stream(s)

# capture
g = torch.cuda.CUDAGraph()
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    with torch.cuda.amp.autocast():
        static_y_pred = model(static_input)
        static_loss = loss_fn(static_y_pred, static_target)
    scaler.scale(static_loss).backward()
    # don't capture scaler.step(optimizer) or scaler.update()

real_inputs = [torch.rand_like(static_input) for _ in range(10)]
real_targets = [torch.rand_like(static_target) for _ in range(10)]

for data, target in zip(real_inputs, real_targets):
    static_input.copy_(data)
    static_target.copy_(target)
    g.replay()
    # Runs scaler.step and scaler.update eagerly
    scaler.step(optimizer)
    scaler.update()
```

### 使用多个流[](#usage-with-multiple-streams)

捕获模式会自动传播到可以正在捕获流同步的其他流.在捕获中,你可以对不同流进行调用来公开并行性,但是整个流依赖的 DAG必须在捕获开始后从最初的捕获流中分支出来,并在捕获结束前重新加入初始流.

```python
with torch.cuda.graph(g):
    # at context manager entrance, torch.cuda.current_stream()
    # is the initial capturing stream

    # INCORRECT (does not branch out from or rejoin initial stream)
    with torch.cuda.stream(s):
        cuda_work()

    # CORRECT:
    # branches out from initial stream
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        cuda_work()
    # rejoins initial stream before capture ends
    torch.cuda.current_stream().wait_stream(s)
```

注意

To avoid confusion for power users looking at replays in nsight systems or nvprof: Unlike eager execution, the graph interprets a nontrivial stream DAG in capture as a hint, not a command. During replay, the graph may reorganize independent ops onto different streams or enqueue them in a different order (while respecting your original DAG’s overall dependencies).

### 使用 DistributedDataParallel[](#usage-with-distributeddataparallel)

#### NCCL < 2.9.6[](#nccl-2-9-6)
版本低于 2.9.6 的 NCCL 不允许捕获集合.你须使用[部分网络捕获](#partial-network-capture)
NCCL versions earlier than 2.9.6 don’t allow collectives to be captured. You must use [partial-network capture](#partial-network-capture), which defers allreduces to happen outside graphed sections of backward.

Call [`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") on graphable network sections _before_ wrapping the network with DDP.

#### NCCL >= 2.9.6[](#id5)

NCCL versions 2.9.6 or later allow collectives in the graph. Approaches that capture an [entire backward pass](#whole-network-capture) are a viable option, but need three setup steps.

1.  Disable DDP’s internal async error handling:
    
    ```python
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "0"
    torch.distributed.init_process_group(...)
    ```
    
2.  Before full-backward capture, DDP must be constructed in a side-stream context:
    
    ```python
    with torch.cuda.stream(s):
        model = DistributedDataParallel(model)
    ```
3.  Your warmup must run at least 11 DDP-enabled eager iterations before capture.
    

### Graph memory management[](#graph-memory-management)

A captured graph acts on the same virtual addresses every time it replays. If PyTorch frees the memory, a later replay can hit an illegal memory access. If PyTorch reassigns the memory to new tensors, the replay can corrupt the values seen by those tensors. Therefore, the virtual addresses used by the graph must be reserved for the graph across replays. The PyTorch caching allocator achieves this by detecting when capture is underway and satisfying the capture’s allocations from a graph-private memory pool. The private pool stays alive until its [`CUDAGraph`](https://pytorch.org/docs/stable/generated/torch.cuda.CUDAGraph.html#torch.cuda.CUDAGraph "torch.cuda.CUDAGraph") object and all tensors created during capture go out of scope.

Private pools are maintained automatically. By default, the allocator creates a separate private pool for each capture. If you capture multiple graphs, this conservative approach ensures graph replays never corrupt each other’s values, but sometimes needlessly wastes memory.

#### Sharing memory across captures[](#sharing-memory-across-captures)

To economize the memory stashed in private pools, [`torch.cuda.graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph") and [`torch.cuda.make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") optionally allow different captures to share the same private pool. It’s safe for a set of graphs to share a private pool if you know they’ll always be replayed in the same order they were captured, and never be replayed concurrently.

[`torch.cuda.graph`](https://pytorch.org/docs/stable/generated/torch.cuda.graph.html#torch.cuda.graph "torch.cuda.graph")’s `pool` argument is a hint to use a particular private pool, and can be used to share memory across graphs as shown:

```python
g1 = torch.cuda.CUDAGraph()
g2 = torch.cuda.CUDAGraph()

# (create static inputs for g1 and g2, run warmups of their workloads...)

# Captures g1
with torch.cuda.graph(g1):
    static_out_1 = g1_workload(static_in_1)

# Captures g2, hinting that g2 may share a memory pool with g1
with torch.cuda.graph(g2, pool=g1.pool()):
    static_out_2 = g2_workload(static_in_2)

static_in_1.copy_(real_data_1)
static_in_2.copy_(real_data_2)
g1.replay()
g2.replay()
```

With [`torch.cuda.make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables"), if you want to graph several callables and you know they’ll always run in the same order (and never concurrently) pass them as a tuple in the same order they’ll run in the live workload, and [`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") will capture their graphs using a shared private pool.

If, in the live workload, your callables will run in an order that occasionally changes, or if they’ll run concurrently, passing them as a tuple to a single invocation of [`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") is not allowed. Instead, you must call [`make_graphed_callables()`](https://pytorch.org/docs/stable/generated/torch.cuda.make_graphed_callables.html#torch.cuda.make_graphed_callables "torch.cuda.make_graphed_callables") separately for each one.
