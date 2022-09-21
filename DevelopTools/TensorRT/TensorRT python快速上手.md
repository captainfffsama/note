#TensorRT 

# 引入TensorRT
```python
import tensorrt as trt

TRT_LOGGER=trt.Logger(trt.Logger.WARNING)
```

# 转换网络
目的是为了将目标网络变为一个[`INetworkDefinition`](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Graph/Network.html#tensorrt.INetworkDefinition)实例.   
常见的方法用用 tensorRT 内建的方法搭建网络和从 ONNX 或者 Caffe TensorFlow 转.这里以内建方法搭建网络为例演示流程:

```python
# Create the builder and network
with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
	# Configure the network layers based on the weights provided. In this case, the weights are imported from a pytorch model. 
	# Add an input layer. The name is a string, dtype is a TensorRT dtype, and the shape can be provided as either a list or tuple.
	input_tensor = network.add_input(name=INPUT_NAME, dtype=trt.float32, shape=INPUT_SHAPE)

	# Add a convolution layer
	conv1_w = weights['conv1.weight'].numpy()
	conv1_b = weights['conv1.bias'].numpy()
	conv1 = network.add_convolution(input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w, bias=conv1_b)
	conv1.stride = (1, 1)

	pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
	pool1.stride = (2, 2)
	conv2_w = weights['conv2.weight'].numpy()
	conv2_b = weights['conv2.bias'].numpy()
	conv2 = network.add_convolution(pool1.get_output(0), 50, (5, 5), conv2_w, conv2_b)
	conv2.stride = (1, 1)

	pool2 = network.add_pooling(conv2.get_output(0), trt.PoolingType.MAX, (2, 2))
	pool2.stride = (2, 2)

	fc1_w = weights['fc1.weight'].numpy()
	fc1_b = weights['fc1.bias'].numpy()
	fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)

	relu1 = network.add_activation(fc1.get_output(0), trt.ActivationType.RELU)

	fc2_w = weights['fc2.weight'].numpy()
	fc2_b = weights['fc2.bias'].numpy()
	fc2 = network.add_fully_connected(relu1.get_output(0), OUTPUT_SIZE, fc2_w, fc2_b)

	fc2.get_output(0).name =OUTPUT_NAME
	network.mark_output(fc2.get_output(0))
```

这里流程就是使用上一步的 `Logger` 创建一个 `tnesorrt.Builder` 实例,然后使用这个实例创建一个 `INetworkDefinition` 实例.然后使用 `INetworkDefinnition` 实例设置输入,添加层构建网络,然后标记输出.

# 构建引擎 `ICudaEngine`
即得到一个[`ICudaEngine`](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Engine.html#tensorrt.ICudaEngine)对象,这个对象是用来推理的.可以使用`[]`来索引,索引返回相应的绑定名称.当使用 字符串来索引的时候,返回的则是对应名称的索引.
```python
with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config:
    config.max_workspace_size = 1 << 20 # This determines the amount of memory available to the builder when building an optimized engine and should generally be set as high as possible.
    with builder.build_engine(network, config) as engine:

```

# 序列与反序列化引擎
序列化:  
```python
with open(“sample.engine”, “wb”) as f:
		f.write(engine.serialize())
```

反序列化:  
```python
with open(“sample.engine”, “rb”) as f, trt.Runtime(TRT_LOGGER) as runtime:
		engine = runtime.deserialize_cuda_engine(f.read())
```

# 进行推理
## 分配设备和 buffer
参照英伟达官方示例,这里展示假定固定大小,单输入输出.输入时,
```python
# Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
#volum计算输入输出大小,pagelocked_empty设置设备的缓冲区
	  h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
	  h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
	  # Allocate device memory for inputs and outputs.
	  # 分配cuda显存
	  d_input = cuda.mem_alloc(h_input.nbytes)
	  d_output = cuda.mem_alloc(h_output.nbytes)
	  # Create a stream in which to copy inputs/outputs and run inference.
	  stream = cuda.Stream()
```
这里关于内存的设置可以参考[锁页内存](../CUDA/锁页内存.md)  

## 拷贝 buffer
将内存中的缓冲区内存(锁页内存)异步的拷贝到 cuda 显存上.执行推理,将推理结果拷贝会内存中的锁页内存上,等待拷贝完成.

```python
with engine.create_execution_context() as context:
		# Transfer input data to the GPU.
		cuda.memcpy_htod_async(d_input, h_input, stream)
		# Run inference.
		context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
		# Transfer predictions back from the GPU.
		cuda.memcpy_dtoh_async(h_output, d_output, stream)
		# Synchronize the stream
		stream.synchronize()
		# Return the host output. 
return h_output
```

注意一个引擎可以有多个执行上下文.  python版的推理可以使用 `np.copyto` 直接将一个 array 拷贝到另外一个array.