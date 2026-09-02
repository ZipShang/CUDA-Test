# CUDA 图像融合优化路线

## 计时口径

当前 `File.cu` 输出的 `GPU pipeline` 时间是 CPU 墙钟时间，包含：

- `cudaMalloc` / `cudaFree`；
- H→D 与 D→H 数据传输；
- kernel launch 和实际 kernel 执行；
- 最终结果的主机端复制。

Nsight Systems 的 CUDA GPU 汇总仅统计 GPU 上实际发生的 memcpy 与 kernel 时间；
它不包含 CPU 端 API 调用、内存分配和调度间隙。两种数据不能直接混为同一个指标。

## 版本规划

### v01_host_roundtrip — 当前基线

每次两图融合都执行两次 H→D、一次 kernel、一次 D→H。融合结果回到 CPU，
下一轮再上传。实现直观，便于验证，但传输量很大。

### v02_device_resident_reduction — 下一步

目标：把全部中间结果保留在 GPU。

1. 首次将 16 张输入图各上传一次；
2. 使用两组 device buffer（ping-pong）完成全部两两归约；
3. 仅将最终一张图 D→H 拷回；
4. 在循环外分配并复用 device memory；
5. 继续保留 CPU 参考实现与逐像素比较。

预期：显著减少 PCIe 传输；v01 中 GPU 时间约 97% 花在数据传输上。

### v03_pinned_async_pipeline

目标：在确有 H↔D 数据交换的场景减少传输开销。

- 用 `cudaMallocHost` 或 `cudaHostAlloc` 分配页锁定主机内存；
- 使用 `cudaMemcpyAsync`、stream 和 event；
- 评估拷贝与计算是否可重叠。

### v04_kernel_and_memory_tuning

目标：当数据传输已不是主要瓶颈后，分析 kernel 自身。

- 用 Nsight Compute 检查带宽、occupancy 和访存效率；
- 测试不同 block 大小（128 / 256 / 512）；
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
