#openvino 

[toc]

安装好onnx python环境之后

# 模型优化
使用脚本是`deployment_tools/model_optimizer/mo.py`, 具体参数参见[参数参考](https://docs.openvinotoolkit.org/cn/latest/openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model_General.html)  
执行
```python
 python model_optimizer/mo.py --input_model ./kp2d.onnx --output_dir ./  --input_shape [1,3,1080,1920]
```
注意 openvino 不支持动态大小,需要指定输入大小.另外对 onnx 11 兼容不太好,这里使用 onnx 10 可以转换成功  
参数还可以参考:https://zhuanlan.zhihu.com/p/261091125
关于输入层分辨率的调整可以参见:https://blog.csdn.net/github_28260175/article/details/107128484

#  模型量化
参考:https://docs.openvinotoolkit.org/cn/latest/pot_README.html
## 安装环境
新建一个 conda 环境.  
Model optimizer 的依赖在`deployment_tools/model_optimizer`,   直接使用 pip 安装即可.  
Accuracy Checker 在`deployment_tools/open_model_zoo/tools/accuracy_checker` 下执行 `python setup.py install`,可能需要翻墙.  
安装POT工具,在 `deployment\_tools/tools/post\_training\_optimization\_toolkit` 下执行 `python setup.py install`  

##  量化模型
这里先执行 openvino 的环境脚本,初始化环境
```python
source <INSTALL\_DIR>/bin/setupvars.sh
```


# 参考
- https://docs.openvinotoolkit.org/cn/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX.html
- https://github.com/david-drew/OpenVINO-Custom-Layers/blob/master/2019.r2.0/ReadMe.Linux.2019.r2.md
- https://blog.csdn.net/github_28260175/article/details/107128484