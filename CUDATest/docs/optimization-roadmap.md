# CUDA 图像融合优化路线

## 计时口径

`File.cu` 输出的时间是 CPU 墙钟时间，每轮包含：

- 输出 `cv::Mat` 创建；
- device 内存申请与释放；
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
通过 `cudaMemcpyAsync` 和 event 驱动的 buffer pool 重叠 H2D 与 kernel。

### v04_single_kernel_reduction

将 16 张图像放入一片连续 device 内存。`blendImageIter` 的每个线程处理一个像素偏移，
在一个 kernel 内按 `8 + 4 + 2 + 1` 的树形顺序完成融合，并将每层结果原地写回左节点。

Nsight Systems 第 6～10 轮平均：16 次 H2D 为 `17.080 ms`，单次 kernel 为 `1.886 ms`，
最终 D2H 为 `1.808 ms`，GPU 时间跨度为 `22.397 ms`。相对 v02，kernel 时间由约
`2.442 ms` 降至 `1.886 ms`；但 v02 使用 16 次独立分配、v04 使用 1 次连续分配，完整
生命周期收益不能全部归因于单 kernel。

## 下一步：v05_fair_kernel_comparison

为隔离 kernel 融合的收益，让对照版本与 v04 使用同一片连续显存和相同的传输方式，
仅比较 15 次 kernel launch 与 1 次 kernel launch：

- 使用 Nsight Compute 检查带宽、occupancy 和访存效率；
- 测试不同 block 大小（128 / 256 / 512）；
- 评估向量化访问或使用 shared memory 的收益；
- 在相同资源生命周期口径下比较单 kernel 与多 kernel。

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