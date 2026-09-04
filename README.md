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

当前活动实现对应 [v04_single_kernel_reduction](CUDATest/versions/v04_single_kernel_reduction/README.md)：

- CPU 和 GPU 使用完全一致的两两归约顺序和整数向下取整规则；
- 程序逐像素比较 CPU/GPU 输出，只有完全一致才保存 GPU 输出图；
- 16 张输入图像存入一片连续 device 内存，中间结果原地覆盖归约树的左节点；
- 一个 `blendImageIter` kernel 在每个像素线程内完成 15 次分层融合；
- 每轮计时包含输出 `cv::Mat` 创建、显存申请、H→D、kernel、D→H 与显存释放。

## 开发规则

1. 先保证 CPU/GPU 结果完全一致，再讨论性能；
2. 每次优化建立新的 `versions/vNN_名称/`，并在其 README 写明设计、结果和局限；
3. 历史版本不改动；新版本在活动源码中开发、测试成功后再冻结；
4. 每个版本至少记录：输入规模、正确性验证、编译配置、计时口径、Nsight Systems 数据；
5. 性能对比必须注明是否包含内存分配、H↔D 传输和 CPU 端等待。

后续设计见 [optimization-roadmap.md](CUDATest/docs/optimization-roadmap.md)。