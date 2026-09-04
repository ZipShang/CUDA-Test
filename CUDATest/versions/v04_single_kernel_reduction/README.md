# v04 — Single-Kernel Reduction

这是单个 CUDA kernel 内完成图像归约的冻结快照。

## 设计

- 输入：任意 2 的幂次张数、连续存储的 `CV_8UC1` 灰度图；当前程序输入为 16 张；
- 显存：一次 `cudaMalloc` 申请 `[image0][image1]...[imageN-1]` 连续布局；
- 融合：`blendImageIter` 的每个线程处理一个像素偏移，按 `8 + 4 + 2 + 1` 完成 16 图归约；
- 复用：每层结果原地写回左节点，右节点在被读取后不再使用，不申请额外中间 buffer；
- 正确性：`CUDATest.cpp` 的 CPU 参考实现保持相同归约树和整数向下取整规则，并逐像素比较输出。

## 性能数据

Nsight Systems 对第 6～10 轮的统计：

| 指标 | 每轮平均 |
|---|---:|
| 16 次 H2D | 17.080 ms |
| 1 次 `blendImageIter` | 1.886 ms |
| 最终 D2H | 1.808 ms |
| GPU 时间跨度 | 22.397 ms |

对照 v02 的 15 次 `blendImage` 合计 `2.442 ms`，v04 的 kernel 时间降低约 `22.8%`。
完整 GPU 时间跨度缩短约 `17.4%`，但其中也包含连续显存申请替代 16 次独立申请的收益。

## 局限

- 默认 stream 与同步 `cudaMemcpy` 未重叠 H2D 和计算；
- 16 次 H2D 仍是主要瓶颈；
- 需要让多 kernel 对照版本采用相同的连续显存布局，才能单独衡量 kernel 融合收益；
- 当前未测试不同 block 大小、向量化访问或 shared memory。

本目录中的三个源码文件是冻结快照，不参与上级 Visual Studio 工程的编译。