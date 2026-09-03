# CUDA 图像融合优化路线

## 计时口径

当前 `File.cu` 输出的 `GPU async pipeline` 时间是 CPU 墙钟时间，每轮包含：

- 输出 `cv::Mat` 创建；
- 输入输出页锁定内存的注册和解除；
- 5 片 device buffer 的 `cudaMalloc` / `cudaFree`；
- stream 和 event 的创建与销毁；
- H→D、kernel、D→H 与 CPU 端等待。

Nsight Systems 的 CUDA GPU 汇总只统计 GPU 上实际发生的 memcpy 与 kernel 时间；
它不包含 CPU 端 API 调用、内存分配和调度间隙。两种数据不能直接混为同一个指标。

## 已完成版本

### v01_host_roundtrip

每次两图融合都执行两次 H→D、一次 kernel、一次 D→H。融合结果回到 CPU，
下一轮再上传。实现直观，便于验证，但传输量很大。

### v02_device_resident_reduction

全部中间结果驻留 GPU：首次上传 16 张输入，完成 15 次两两归约后，只将最终
一张图 D→H 拷回。该版本显著减少了 PCIe 传输，但仍使用同步执行。

### v03_pinned_async_pipeline

使用 5 个 device buffer、一个传输 stream 和一个计算 stream。输入和输出页锁定后，
通过 `cudaMemcpyAsync` 和 event 驱动的 buffer pool 重叠 H2D 与 kernel。当前后 5 轮
Nsight GPU 时间跨度约为 11.816 ms；完整生命周期 CPU 墙钟时间约为 24.8 ms。

## 下一步：v04_kernel_and_memory_tuning

当传输与资源生命周期成本已明确后，继续分析 kernel 和内存策略：

- 用 Nsight Compute 检查带宽、occupancy 和访存效率；
- 测试不同 block 大小（128 / 256 / 512）；
- 对比“每轮注册页锁定内存”与“复用 pinned host buffer”两种计时口径；
- 评估 CUDA Graph 或融合更多归约层以减少 launch 开销；
- 评估向量化访问、批处理多个图像或更高精度融合。

## 每个版本的记录模板

```text
版本名称：
设计目标：
相对于上一版的代码变化：
输入与图像格式：
正确性验证：CPU/GPU 是否逐像素一致
构建配置：Debug/Release、GPU 架构、CUDA 版本
计时口径：墙钟 / CUDA event / Nsight GPU 时间
性能数据：平均值、样本数、传输与 kernel 占比
结论与下一步：
```