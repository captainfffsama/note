#yolov8

```python
# come from: ultralytics>engine>trainer.py>BaseTrainer>build_optimizer
if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically… "
            )
            nc = getattr(model, "nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam
```

即计算出 iterations 大于 10000 时用 SGD, 否则用 AdamW ,使用后者会替换掉初始学习率

官方说明可以参见:<https://github.com/ultralytics/ultralytics/issues/3629>