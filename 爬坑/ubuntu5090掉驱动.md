```bash
# 1. 安装DKMS（如未安装）
sudo apt-get install dkms

# 2. 查看已安装的NVIDIA驱动版本
ls /usr/src | grep nvidia  # 输出如 nvidia-550.90.07

# 3. 使用DKMS重新构建模块
sudo dkms install -m nvidia -v 550.90.07  # 版本号需替换为实际值

# 4. 验证修复
nvidia-smi
```