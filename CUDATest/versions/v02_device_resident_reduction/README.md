# v02 — Device-Resident Reduction

这是 `v02_device_resident_reduction` Git tag 对应的冻结快照。

## 设计

- 输入：16 张连续存储的 `CV_8UC1` 灰度图；
- 首次将全部输入图像上传到 GPU；
- 采用树形两两归约，kernel 将结果原地写回左输入 buffer；
- 中间结果始终驻留在 device，最后仅回传一张融合图；
- CPU 参考实现保持相同的归约顺序和整数向下取整规则。

## 性能数据

10 轮执行后取第 6～10 轮平均：

| 指标 | 每轮平均 |
|---|---:|
| H↔D 传输 | 约 18.6 ms |
| 15 次 kernel 合计 | 约 2.4 ms |
| CPU 墙钟时间 | 约 30.1 ms |

相对 v01，显著减少了中间结果的主机往返传输。

## 局限

- 每轮仍申请和释放 16 片 device buffer；
- 使用默认 stream 与同步 memcpy，尚未进行 H2D 和 kernel 重叠；
- 输入数量固定为 16，归约循环未泛化为任意 2 的幂次。

本目录中的三个源码文件是冻结快照，不参与上级 Visual Studio 工程的编译。