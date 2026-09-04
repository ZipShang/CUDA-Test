# v06 — Pinned Reuse and Kernel Concurrency

这是固定锁页内存复用与有意义 kernel 并发实验的冻结快照。

## 范围

- 输入固定为 16 张连续存储的 `CV_8UC1` 灰度图；每张图像为 8,520,552 B；
- 测试 GPU：NVIDIA GeForce RTX 4050 Laptop GPU；
- `Proc6` 探索固定锁页 staging 内存复用；
- `Proc7` 在完整等权融合的同时，对原始输入图执行亮度/对比度一致性检测；
- `Proc7_Compare` 是 `Proc7` 的严格单变量对照：质量 kernel 跑满 grid，且与融合 kernel 串行执行。

## Proc6：固定锁页 staging 内存复用

- 在 10 轮大循环外一次申请连续 17 片锁页内存：16 片输入、1 片输出；
- 每轮把普通 `cv::Mat` 拷贝到输入 staging 区，将输出 staging 区拷贝回普通 `cv::Mat`；
- CUDA H2D、归约融合、D2H、device buffer/stream/event 的完整流程仍在每轮内；
- 固定锁页申请/释放单独计时，不纳入第 6～10 轮稳定吞吐平均。

### Proc3 与 Proc6 数据

| 指标 | Proc3：逐轮 HostRegister | Proc6：固定 pinned staging |
|---|---:|---:|
| 稳定 5 轮完整流程平均 | 24.073 ms | 27.470 ms |
| kernel（10 轮累计） | 23.134 ms | 23.036 ms |
| H2D（10 轮累计） | 103.565 ms | 103.248 ms |
| D2H（10 轮累计） | 6.466 ms | 6.466 ms |
| `cudaHostRegister` / `cudaHostUnregister` | 170 / 170 次 | 无 |
| 普通内存到 pinned staging 拷贝 | 无 | 11.944 ms（稳定轮平均） |

Nsight Systems 中 Proc6 的一次 `cudaMallocHost` 为 63.955 ms、`cudaFreeHost` 为 8.532 ms；它们是初始化/收尾成本。首次直接运行中更大的申请时间包含 CUDA context 初始化，不应视为稳定的 pinned 分配成本。

结论：Proc6 不改变 GPU 的 kernel 或 H2D/D2H 工作量；它移除了反复页面注册，但新增的 CPU staging 拷贝超过了收益。因此，只有数据能从采集、解码或上游处理阶段直接写入 pinned 内存时，固定 pinned 内存复用才可能改善稳定帧吞吐。

## Proc7：融合与输入质量检测并发

`proc7InputQuality` 完整扫描 16 张原始输入图，为每张图累计像素和与平方和，并在 CPU 端计算亮度均值、标准差和亮度跨度。该指标用于判断等权融合是否可能受输入曝光不一致影响。

- 融合分支在 `blendStream` 中执行 15 次 `blendImage`；
- 质量分支在 `qualityStream` 中执行 `proc7InputQuality`；
- 质量 kernel 的 grid 限制为 `smCount` 个 block，每 block 256 线程、4 KB 动态 shared memory；
- 融合会原地覆盖归约树左节点，因此两分支共享 H2D 后，额外生成一份 device-to-device 输入快照，仅供质量分支读取，避免跨 stream 读写竞争；
- 当前输入的亮度跨度为 0.79，按演示阈值（不超过 10）适合等权融合。

## Proc7_Compare：严格单变量对照

两种模式均使用同一个 `proc7InputQuality` kernel、同一输入快照、相同 256 线程/block、相同 4 KB shared memory、相同显存申请和 H2D/D2D/D2H 流程。仅有以下运行时差异：

| 项目 | Proc7 | Proc7_Compare |
|---|---|---|
| 质量 kernel grid | `smCount` | `blendBlocksPerGrid`（完整 grid） |
| 融合 kernel 所在 stream | `blendStream`，可与质量并发 | `qualityStream`，在质量之后严格串行 |

### 单变量测试数据

| 指标 | Proc7：低占用并发 | Proc7_Compare：满 grid 串行 |
|---|---:|---:|
| 多次稳定 5 轮平均的均值 | 40.268 ms | 38.908 ms |
| 质量 kernel（Nsight 10 轮平均） | 8.082 ms | 5.116 ms |
| 15 次融合 kernel（Nsight 每轮合计） | 2.973 ms | 2.495 ms |
| 质量与融合 kernel 重叠 | 稳定轮 2.94～3.09 ms | 0 ms |

两种模式均通过 CPU/GPU 逐像素融合结果校验，质量统计结果相同。

结论：当前质量工作负载中，低占用质量 kernel 虽然能够隐藏约 3 ms 融合时间，但自身从 5.116 ms 增至 8.082 ms。满 grid 的串行方案平均快约 1.36 ms（约 3.4%）。这证明 kernel 并行本身不等于更快：当一个全图 kernel 能更有效利用 SM 和内存带宽时，优先提升单 kernel 效率通常比为并发预留 SM 更有价值。

## Nsight Systems 报告

原始报告保存在上级 Release 输出目录：

- `x64/Release/proc6_fixed_pinned_host.nsys-rep`；
- `x64/Release/proc7_parallel_fair.nsys-rep`；
- `x64/Release/proc7_compare_fair.nsys-rep`。

CPU context-switch trace 因采样权限不足未启用；CUDA API、memcpy 和 kernel trace 正常采集。

本目录中的三个源码文件是冻结快照，不参与上级 Visual Studio 工程的编译。
