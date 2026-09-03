# v03 — Pinned Async Pipeline

这是当前异步双 stream 实现的冻结快照。

## 设计

- 输入：16 张连续存储的 `CV_8UC1` 灰度图；
- 使用 5 片 `imageSize` 的 device buffer 组成滚动 buffer pool；
- `copyStream` 执行 H2D 和最终 D2H，`computeStream` 执行 15 次融合 kernel；
- 每个归约节点记录图像范围、层级、所在 buffer 和 ready event；
- buffer 在最后一次 kernel 读取完成后，通过 event 安全地交还给传输流复用；
- 输入与输出 `cv::Mat` 使用 `cudaHostRegister` 注册为页锁定内存，支持 `cudaMemcpyAsync` 与 kernel 重叠。

## 正确性与计时

- CPU/GPU 保持同一棵两两归约树和整数向下取整规则；
- 程序逐像素比较 CPU/GPU 输出；
- 每轮计时包含输出 `cv::Mat` 创建、页锁定注册、5 片显存申请、stream/event 创建、GPU 流程、资源释放和解除页锁定；
- 第 1 轮通常包含 CUDA context 初始化，只作为 warm-up；统计第 6～10 轮平均。

## 已知性能数据

最新 Nsight Systems 后 5 轮平均：

| 指标 | 每轮平均 |
|---|---:|
| 16 次 H2D | 10.358 ms |
| 15 次 kernel 合计 | 2.305 ms |
| 最终 D2H | 0.647 ms |
| GPU 时间跨度 | 11.816 ms |
| 完整生命周期 CPU 墙钟时间 | 约 24.8 ms |

15 次 kernel 中，11 次与 H2D 存在重叠；最后 4 次受归约依赖限制，不能与 H2D 重叠。

## 局限

- 当前调度和 buffer 数量针对 16 张输入图像；
- 每轮重新注册和解除页锁定内存，完整生命周期时间主要受此 CPU 端开销影响；
- 未对 block 大小、kernel 访存效率或 CUDA Graph 做进一步调优。

本目录中的三个源码文件是冻结快照，不参与上级 Visual Studio 工程的编译。