`src/lerobot/datasets/dataset_writer.py:558` 中把已有的 shard MP 4 和新的 episode MP 4 进行 stream-copy 进行拼接。当使用 HEVC 进行编码，可能产生 B 帧（参见 [视频编码中的B帧I帧P帧](../../DL_knowlege/视频编解码/视频编码中的B帧I帧P帧.md)），B 帧解码时间 DTS 和显示时间 PTS 顺序不同；多个独立编码，从 0 开始计时的 episode MP 4 被反复拼接式，边界处时间戳重排可能产生一个帧间隔的空洞。

`src/lerobot/datasets/video_utils.py:1011`:

LeRobot 的流式编码器使用有界队列。其原始逻辑是：``

```python
self._frame_queues[video_key].put(image.copy(), timeout=0.1)
# 如果 100 ms 内仍然没有空间：
except queue.Full:
	self._dropped_frames[video_key] += 1
```

这套行为适合实时采集：编码器来不及就丢帧，避免阻塞机器人控制。但不适合离线数据转换，因为：

- dataset.add_frame() 仍然把该帧写入 Parquet；
- MP 4 编码器却丢弃了图像；
- 最终 Parquet 帧数大于视频帧数；
- 训练按照 Parquet 时间戳读取 MP 4 时便报 FrameTimestampError。
- 修复后，在调用 dataset.add_frame() 前检查所有编码队列；队列满时阻塞等待：
