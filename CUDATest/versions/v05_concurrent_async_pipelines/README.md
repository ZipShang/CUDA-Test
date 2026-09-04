# v05 — Concurrent Async Pipelines

这是使用多个主机线程并发提交 CUDA 异步流水线的冻结快照。

## 设计

- 输入：16 张连续存储的 `CV_8UC1` 灰度图；
- 每条流水线沿用 v03 的 5-buffer、双 stream、页锁定内存和 event 驱动归约设计；
- `Proc5` 创建 3 个主机线程，每个线程独立提交一条异步 GPU 流水线；
- `Proc5_Compare` 在调用线程中串行提交相同数量的 3 条流水线，作为公平对照；
- 程序入口默认调用 `Proc5`，CPU 参考实现继续逐像素验证最终输出；`Proc5_Compare` 保留为同工作量的串行提交对照接口。

## 正确性与计时

- CPU/GPU 保持相同的两两归约树和整数向下取整规则；
- 两种模式均已通过逐像素 CPU/GPU 一致性检查；
- 每条流水线执行 10 轮，统计第 6～10 轮的平均 CPU 墙钟时间；
- 每轮包含输出 `cv::Mat` 创建、页锁定注册、5 片显存申请、stream/event 创建、GPU 流程、资源释放和解除页锁定；
- Nsight Systems 使用 CUDA trace，关闭 device-side CUDA event completion trace，以避免采集本身增加额外开销或伪依赖。

## 性能数据

测试 GPU：NVIDIA GeForce RTX 4050 Laptop GPU。每张图像大小为 8,520,552 B。

| 指标 | Proc5：3 主机线程并发 | Proc5 Compare：单主机线程串行 |
|---|---:|---:|
| 3 条流水线批次 CPU 墙钟时间 | 70.389 ms | 76.142 ms |
| 每条流水线 15 次 kernel 合计 | 2.379 ms | 2.221 ms |
| 每条流水线 16 次 H2D 合计 | 10.942 ms | 10.641 ms |
| 每条流水线 1 次 D2H | 0.707 ms | 0.696 ms |
| 每条流水线 GPU 操作累计 | 14.028 ms | 13.558 ms |

Nsight Systems 对每种模式均记录到 450 次 kernel、480 次 H2D 和 30 次 D2H，说明比较对象的 GPU 工作量一致。原始报告保存在上级 Release 输出目录：

- `x64/Release/proc5_concurrent.nsys-rep`；
- `x64/Release/proc5_compare_serial.nsys-rep`。

## 结论与局限

`Proc5` 将三条流水线批次的完整生命周期时间减少 5.753 ms（7.6%），约为 1.08 倍吞吐提升。GPU 单项操作并未加速：并发模式的 kernel 累计时间高约 7.1%，H2D 累计时间高约 2.8%，表明并发提交会带来 GPU 调度和传输资源竞争。

因此，当前收益主要来自主机侧 CUDA 资源管理、任务提交和等待路径的并行化，而不是单个 kernel 或单次传输更快。图像读取、CPU 参考归约和输入克隆不在每轮 benchmark 的计时范围内，不应把该收益归因为图像预处理并行。

当前每轮重复执行页锁定注册/解除、device buffer 申请/释放、stream/event 创建/销毁。下一轮应先复用这些资源，再重新比较主机并发提交的收益。

本目录中的三个源码文件是冻结快照，不参与上级 Visual Studio 工程的编译。
