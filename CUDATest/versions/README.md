# 历史版本快照

每个子目录都是不可修改的学习快照。版本目录只保存源代码和说明，
Visual Studio 工程仍编译上级目录中的活动源码。

| 版本 | 说明 |
|---|---|
| [v01_host_roundtrip](v01_host_roundtrip/README.md) | 每次融合均发生主机往返的基线实现。 |
| [v02_device_resident_reduction](v02_device_resident_reduction/README.md) | 中间结果驻留 GPU，最终只回传一张图。 |
| [v03_pinned_async_pipeline](v03_pinned_async_pipeline/README.md) | 5-buffer、双 stream、页锁定内存与 event 驱动同步。 |

冻结新版本时：

1. 在活动源码中完成实现、正确性验证与性能测试；
2. 新建 `vNN_简短名称`；
3. 复制 `CUDATest.cpp`、`File.cu`、`KernelApi.h` 到新目录；
4. 写入 README，记录实现差异、测试数据和已知限制；
5. 之后只在活动源码中开始下一轮优化。

当前活动实现已对应 `v03_pinned_async_pipeline`；下一轮优化应从上级活动源码继续开发。