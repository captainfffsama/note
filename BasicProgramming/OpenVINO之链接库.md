#openvino 


[toc]
原文: https://blog.csdn.net/github_28260175/article/details/105868523


# 1. 程序编好后为什么会提示`plugins.xml:1:0: File was not found`
----------------------------------------------------

我们回顾下OpenVINO(最新版本2020.2)的使用过程：  
![](https://img-blog.csdnimg.cn/20200504174728429.png#pic_center)

1.  新建`InferenceEngine::Core core`
2.  读取模型，由`xxx.bin`与`xxx.xml`
3.  配置输入输出格式（似乎这里可以不做，一切继承模型的配置）
4.  将模型依靠`InferenceEngine::Core::LoadNetwork()`载入到硬件上
5.  建立推理请求`CreateInferRequest()`
6.  准备输入数据
7.  推理
8.  接住推理后的输出

如果查看一下**第一步**的源码就可以知道，`InferenceEngine::Core::Core`这个东西实际上是需要输入参数`const std::string & xmlConfigFile`的，如下图所示：  
![](https://img-blog.csdnimg.cn/20200504174751937.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dpdGh1Yl8yODI2MDE3NQ==,size_16,color_FFFFFF,t_70#pic_center)

因此，如果我们没有规定这个`xmlConfigFile`的参数，它就会默认读取在`inference_engine.so`文件路径下的`plugins.xml`，可以查看windows下的`your_openvino_install_root\Intel\OpenVINO\omz_demos_build\intel64\Release`或者Linux下的`your_openvino_install_root/openvino/inference_engine/lib/intel64`找到。

为了好奇，可以看下`plugins.xml`这个里面到底有什么，这也跟下面的**问题2**有关：

```json
<ie>
    <plugins>
        <plugin name="GNA" location="libGNAPlugin.so">
        </plugin>
        <plugin name="HETERO" location="libHeteroPlugin.so">
        </plugin>
        <plugin name="CPU" location="libMKLDNNPlugin.so">
        </plugin>
        <plugin name="MULTI" location="libMultiDevicePlugin.so">
        </plugin>
        <plugin name="GPU" location="libclDNNPlugin.so">
        </plugin>
        <plugin name="MYRIAD" location="libmyriadPlugin.so">
        </plugin>
        <plugin name="HDDL" location="libHDDLPlugin.so">
        </plugin>
        <plugin name="FPGA" location="libdliaPlugin.so">
        </plugin>
    </plugins>
</ie>

```

# 2. 我至少该链接哪些OpenVINO的库
----------------------

编译教程提示，最基本的库，需要链接`libinference_engine.so`与`libinference_engine_legacy.so`，所以也需要拖家带口把它们依赖的`libinference_engine_transformations.so`和`libngraph.so`还有`libtbb.so.2`带上，需要注意的是，依赖库的路径有些并不是在`your_openvino_install_root/openvino/inference_engine/lib/intel64`下的，各位同学可以通过`ldd libinference_engine.so`的方式找到准确的依赖库的位置。算了，我这里就把路径全部提出来吧：

*   libinference\_engine.so: root/openvino/inference\_engine/lib/intel64
*   libinference\_engine\_legacy.so: root/openvino/inference\_engine/lib/intel64
*   libinference\_engine\_transformations.so: root/openvino/inference\_engine/lib/intel64
*   **libngraph.so: root/deployment\_tools/ngraph/lib/**
*   **libtbb.so.2: root/openvino/inference\_engine/external/tbb/lib**

![](https://img-blog.csdnimg.cn/2020050417505632.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2dpdGh1Yl8yODI2MDE3NQ==,size_16,color_FFFFFF,t_70#pic_center)

然后，冷静下，**Core Inference Engine Libraries**后面还有**Device-specific Plugin Libraries**，这就是`plugins.xml`规定的plugin需要的库。根据模型运行在目标设备的类型，设置需要的库。比如，我的目标设备是CPU，所以就需要`root/openvino/inference_engine/lib/intel64`下的`libMKLDNNPlugin.so`，因此需要将`plugins.xml`修改为：

```json
<ie>
    <plugins>
        <plugin name="CPU" location="libMKLDNNPlugin.so">
        </plugin>
    </plugins>
</ie>

```

# 3. 提示符号未定义
-----------

```bash
my_lib.so: undefined reference to `InferenceEngine::Core::LoadNetwork(InferenceEngine::CNNNetwork, std::string const&, std::map<std::string, std::string, std::less<std::string>, std::allocator<std::pair<std::string const, std::string> > > const&)'
my_lib.so: undefined reference to `InferenceEngine::Data::getName() const'
my_lib.so: undefined reference to `InferenceEngine::Core::Core(std::string const&)'
my_lib.so: undefined reference to `InferenceEngine::details::InferenceEngineException::InferenceEngineException(std::string const&, int, std::string const&)'
my_lib.so: undefined reference to `InferenceEngine::Core::ReadNetwork(std::string const&, std::string const&) const'

```

莫慌，看看`my_lib.so`里面链接了什么：  
![](https://img-blog.csdnimg.cn/20200504174811532.png#pic_center)

可以看出，只链接了`libinference_engine_legacy.so`，缺了`libinference_engine.so`，为啥？检查下`CMakeLists.txt`吧，看看编译选项有没有`add_definitions(-D _GLIBCXX_USE_CXX11_ABI=0)`什么的，链接OpenVINO需要c++11或c++14。
