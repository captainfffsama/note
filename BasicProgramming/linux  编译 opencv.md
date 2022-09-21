#CPP 
#opencv  
#编译

[toc]

#   编译主模块和额外模块
```bash
# Install minimal prerequisites (Ubuntu 18.04 as reference)
sudo apt update && sudo apt install -y cmake g++ wget unzip

#获取源码
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv
git checkout 4.5.1
cd ../opencv_contrib
git checkout 4.5.1

cd ../

# Create build directory
mkdir -p build && cd build

# Configure
cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv -DCMAKE_INSTALL_PREFIX=you_install_dir

# Build
cmake --build . --target install
```

#  参考
- https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html
- https://docs.opencv.org/4.5.1/db/d05/tutorial_config_reference.html