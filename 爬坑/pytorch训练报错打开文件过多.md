#pytorch 

# 现象

训练报错：RuntimeError: unable to open shared memory object </torch_735354_2766238047_18> in read-write mode: Too many open files (24)

# 解法
使用 `ulimit -Hn` 查看硬限制，使用 `ulimit -Sn` 查看软限制，若软限制低，执行

运行命令前加 `ulimit -S -n 65535` 提高文件描述符上限，比如：

```bash
ulimit -S -n 65535
accelerate launch --num_machines 1 --num_processes 2 --multi_gpu \
  scripts/train.py --config configs/config_holobrain_qwen_hc_tj.py
```