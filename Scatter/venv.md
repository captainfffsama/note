#python 

```python
# 使用venv命令创建虚拟环境，虚拟环境文件所在的目录是venv
python -m venv venv
# 激活虚拟环境，激活成功后，当前路径的名称会改变
. venv/bin/activate
# 安装项目依赖（在已激活的虚拟环境中）
pip install ···
# 执行脚本（在已激活的虚拟环境中）
python test.py
# 退出虚拟环境（在已激活的虚拟环境中）
deactivate
# 生成项目依赖
pip freeze > requirements.txt
# ##############
# 安装项目依赖
pip install -r requirements.txt
```

注意原生的 venv 不支持更换 python 版本的, 且仅在3.5以上版本推荐使用