#TensorRT
# TensorRt核心库

## 网络定义API
网络定义API为应用程序提供了定义网络的接口.  
[C++文档](http://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_network_definition.html)   
[python文档](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html)  

## 配置优化API  
主要是用于对动态维度进行约束的.详情参见:  
[Optimization Profile API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_optimization_profile.html)  
[Working With Dynamic Shapes](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)

## 构建器配置API
用来指定构建器创建引擎时的一些细节.详情参见:   
[Builder Config API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_builder_config.html)  

## 构建器API
用来创建特定网络的引擎,详情参见:   
[Builder API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_builder.html)    

## 引擎  
让应用能够推理网络,有同步和异步的方式.单个引擎支持有多个执行上下文.可以同时执行多个批处理.详情参见:   
[Execution API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvinfer1_1_1_i_cuda_engine.html)    

## Caffe 解析器 
用于解析 BLVC Caffe 或者 NVCaffe 0.16 创建的网络.并且支持自定义层.详情参见:   
C++: [NvCaffeParser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvcaffeparser1_1_1_i_caffe_parser.html)  
python: [Caffe Parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Caffe/pyCaffe.html)    

## UFF解析器
用来解析UFF格式的网络.支持自定义层,详情参见:   
C++: [NvUffParser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvuffparser_1_1_i_uff_parser.html)  
python: [UFF Parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Uff/pyUff.html)   

## ONNX解析器  
用来解析 ONNX 模型.详情参见:  
C++: [NvONNXParser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/c_api/classnvonnxparser_1_1_i_o_n_n_x_parser.html)    
python:  [ONNX Parser](https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/python_api/parsers/Onnx/pyOnnx.html)
