# CUDA Image Blending Learning Demo

这个项目用于学习、验证和逐步优化 CUDA 图像融合流程。当前输入为 16 张
`CV_8UC1` 灰度图，采用两两归约的方式进行 `dst = (src1 + src2) / 2` 融合。

## 目录约定

```text
CUDATest/
├── CUDATest/                 # 当前活动实现；下一轮优化在这里进行
│   ├── CUDATest.cpp           # 入口、CPU 参考实现、结果一致性检查
│   ├── File.cu                # 当前 GPU 实现与性能计时
│   ├── KernelApi.h            # CPU/GPU 调用接口
│   ├── docs/                  # 优化路线与记录规范
│   └── versions/              # 不再修改的历史版本快照
└── CUDATest.sln
```

`versions/` 中的代码是冻结快照，供回顾和对比；不要在其中继续开发。
活动实现通过验证后，先复制为一个新的版本快照，再修改根目录下的活动源码。

## 当前版本

当前活动实现对应 [v05_concurrent_async_pipelines](CUDATest/versions/v05_concurrent_async_pipelines/README.md)：

- 活动源码保留 `Proc1`～`Proc5` 及 `Proc5_Compare` 接口，便于横向比较不同优化方案；入口当前默认调用 `Proc5`；
- `Proc5` 使用 3 个主机线程并发提交各自独立的双 stream CUDA 异步流水线；`Proc5_Compare` 串行提交同样的 3 条流水线；
- 两种模式均复用 v03 的 5-buffer、页锁定内存和 event 驱动归约方案；
- CPU 和 GPU 使用完全一致的两两归约顺序和整数向下取整规则，并逐像素比较输出；
- `v05_concurrent_async_pipelines` 是包含并发与串行对照的冻结快照。

## 开发规则

1. 先保证 CPU/GPU 结果完全一致，再讨论性能；
2. 每次优化建立新的 `versions/vNN_名称/`，并在其 README 写明设计、结果和局限；
3. 历史版本不改动；新版本在活动源码中开发、测试成功后再冻结；
4. 每个版本至少记录：输入规模、正确性验证、编译配置、计时口径、Nsight Systems 数据；
5. 性能对比必须注明是否包含内存分配、H↔D 传输和 CPU 端等待。

后续设计见 [optimization-roadmap.md](CUDATest/docs/optimization-roadmap.md)。
