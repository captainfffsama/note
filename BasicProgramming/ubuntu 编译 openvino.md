#openvino

[toc]
# 安装步骤
## 下载
在[openvino](https://software.intel.com/content/www/us/en/develop/tools/openvino-toolkit/download.html?operatingsystem=linux&distributions=webdownload&options=offline) ,选择 Linux - Web&Local - Local 下载   
 最后进入到一个名为`l_openvino_toolkit_p_<version>`的文件夹,若是之前包含有其他版本的 openvino,删除`~/inference_engine_samples_build`和`~/openvino_models`
 
 ## 安装
 安装支持 GUI CLI 和纯命令行.
 - GUI: `sudo ./install_GUI.sh`
 - CLI:`sudo ./install.sh`
 - 纯命令行: `sudo sed -i 's/decline/accept/g' silent.cfg && sudo ./install.sh -s silent.cfg`   
 若仅仅安装CPU图例引擎, 可以在silent.cfg中添加`COMPONENTS=intel-openvino-ie-rt-cpu\_\_x86\_64`.使用`./install.sh --list\_components`可以显示所有安装选项
 
 ## 配置
具体可以参考官方页面,注意一点是,官方默认调用系统的python pip.配置文件本质就是安装一堆python包,因此可以自己创建conda 环境然后安装requirement.txt
#  参考
- https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_linux.html
- https://oldpan.me/archives/openvino-first-try