# 内核挑战终局：Marlin 借用实测打平 — 2026-10-08 ❌（第十一块证据）

**挑战**："重写内核"冲击 300。**发现的捷径**：不重写——Morrowmake 的 sm_80 专用 Marlin MoE 内核（kernels_sm80.cu，生产 301.9 同款）有 standalone 构建脚本（CPU 可编译，`TORCH_CUDA_ARCH_LIST=8.0`），4.1MB 模块 20 分钟产出。

## Phase 1 微基准（GPU0 空闲片，生产无损）

GLM world=4 精确形状（K=4096、N=512/rank、E=288、topk=8、M=4 decode 行、block-8 对齐路由），测 Marlin 3-kernel 序列（w13 融合 gateup → SwiGLU act → w2 down）：

| | 每层专家段耗时 |
|---|---:|
| **Morrowmake sm_80 Marlin 3-kernel**（借用的预编译内核） | **63-64 µs** |
| TF EXL3 trellis 路径（rot_in + grouped×2 + epilogues + swiglu） | 60-67 µs |
| TF 单 kernel 对照：grouped_kernel 均值 | 68 µs |

**判定：打平。** 借用内核零收益——TF 的 EXL3 专家内核在 decode 形状上已与 Morrowmake 的 sm_80 Marlin 同速。此前"Morrowmake 内核质量 2× 领先"的假设被实测推翻。

## 这意味着什么（归因修正）

169.8 vs 301.9 的 2× 差距**不在内核质量**，而在结构性差异：
1. **通信**：90 次/步 × ~40µs（shm 已优化到 PCIe 物理域）vs Morrowmake 的 host-shm all-reduce 同级——次数由位一致设计锁定；
2. **dense 路径**：_bmm 268 次/步 launch-bound（13µs/次，内核本身不慢）——需要**多投影融合内核**（结构性 CUDA 工程）；
3. **graph gap**：~1,700 kernel × 1.5µs——同上，融合才能消除；
4. **批处理/调度**：Morrowmake 的连续批处理与 accept-aware 调度成熟度。

## 挑战终局

| 尝试层级 | 结果 |
|---|---|
| 配置/权重/策略/Python 融合 | 10 次证伪 |
| 借用 Marlin 内核（专家段） | **打平（本轮）** |
| 剩余：多投影融合 + 通信重构 | 月级结构性工程，触碰位序与 exactness |

**"今天 300"正式不可达，且不再有"换个内核就行"的捷径**——两个栈的内核在同硬件同形状下同速，差距是架构性的。TF 栈定格 **169.8 tok/s**（EXL3+S1+S2），生产 Morrowmake 288.8 无损（微基准仅秒级 SM 竞争）。

## 资产

- `marlin_standalone/_ampere_marlin_C.abi3.so`（sm_80 预编译模块）+ `marlin_microbench.py`（可复用的 MoE 内核微基准）
- 上游素材 +1：TF EXL3 专家内核与 Morrowmake sm_80 Marlin 在 decode 形状**同速**的实测——这是 EXL3 路线质量的有力背书（此前无此数据）
