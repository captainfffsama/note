#pytorch

# 异步操作
默认情况下, GPU 上操作都是异步的,每使用一个 GPU 函数时,操作会被排到特定设备队列中.这个异步计算对于调用者是不可见的.因为每个设备会按照排队顺序执行操作,当 CPU 和 GPU 或 GPU 和 GPU 之间发生数据复制时,PyTorch 会自动执行必要的同步.这样,在调用者看来,似乎每个操作都是同步执行的. 可以通过设置环境变量 `CUDA_LAUNCH_BLOCKING=1` 可以强制同步计算.这在 debug 的时候很有用(因为异步执行时,排队时不会报错,会等到实际执行操作时,才报错,因此堆栈不会跟踪显示请求它的位置.)

另外异步计算在进行时间测量的时候会不准,在测量前可以调用 `torch.cuda.synchronize()` 或者 `torch.cuda.Event` ,例:
```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()

# Run some things here

end_event.record()
torch.cuda.synchronize()  # Wait for the events to be recorded!
elapsed_time_ms = start_event.elapsed_time(end_event)
```

# 内存管理
Pytorch 使用了缓存内存分配器来进行内存分配.这允许在设备没同步的情况下进行内存释放.但是分配器管理的,但未使用的显存在 nvidia-smi 中并不显示.使用 `memory_allocated()` 和 `max_memory_allocated()` 可以迎来检视张量占用的内存,使用 `memory_reserved()` 和 `max_memory_reserved()` 可以用来检视缓存分配器管理的内存总量.`empty_cache()` 可以释放 PyTorch 中所有未使用的缓存内存,以便其他应用程序使用.当然,被张量占用的 GPU 内存不会被释放,因此它不能增加 PyTorch 可用的 GPU 内存量.

使用 `memory_stats()` 可以提供更加全面的内存基准测试.使用 `memory_snapshot()` 可以获得内存分配器状态的完整快照.

缓存分配器可能会干扰内存检查工具,比如 `cuda-memcheck`.可以使用 `PYTORCH_NO_CUDA_MEMORY_CACHING=1` 来禁用缓存.

# 

# 参考
- https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
