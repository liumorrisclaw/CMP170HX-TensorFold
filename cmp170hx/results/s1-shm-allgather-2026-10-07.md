# S1 shm all-gather 落地 — 2026-10-07, <GPU_HOST> ✅ +21.5%

**动作**：把 Morrowmake `host_shm_all_reduce.py`（Apache-2.0，为 4×CMP 无 P2P 而造）的 one-shot 协议移植为 TensorFold 的 all-gather 语义。phase 3 从 FP32 求和改为按 rank 序纯拷贝（天然位精确）。`TF_SHM_AG=1` 开关启用，尺寸/类型超界自动回落 NCCL，`forward.py` 零改动。

**交付物**：
- `patches/shm_allgather.py`（内嵌 CUDA 源码的完整模块，466 行；publish + clock64-bounded barrier + gather-copy 三段 one-shot kernel，generation counter 在 device tensor 内自增 → CUDA graph replay 安全；POSIX shm + cudaHostRegister(Mapped|Portable)）
- `comm.py` 接入（__init__ 可选初始化 + all_gather dispatch + ValueError 回落）
- `tf_shm_test.py` 4-GPU 位精确测试：fp32/bf16/int32 × eager + graph replay ×3 × 跨 rank 确定性数据 **全 PASS**（修过一个测试断言 bug 和一个 bf16 Vec 宽度 bug——8B 而非 16B）

## A/B 结果（同协议，3 轮中位）

| 口径 | NCCL 基线 | S1 shm | Δ | vs 生产 301.9 |
|---|---:|---:|---|---|
| verify v1（串行步） | 21.0ms | **15.85ms** | −25% | |
| verify v4 | 39.5ms | **31.4ms** | −21% | |
| DFlash2 block | 6.28ms | 5.00ms | −20% | |
| c1 structured | 102.8 | **124.9** | **+21.5%** | 34% → **41.4%** |
| c1 coding | 80.8 | 98.1 | +21.4% | 43% |
| c1 prose | 61.4 | 75.1 | +22.3% | 49% |
| 接受率 | 0.7576 | 0.7576 | 持平 | |
| **exactness** | ✅ | ✅ SHA256 逐位一致 | | |

## 模型校准（对 S0 预测 140-150 的偏差分析）

实测 124.9 vs 预测 140-150：**−12%，在 ±30% 容差内，模型成立但有一项低估**。单次 all_gather 实测 ~58µs，不是预测的 20-30µs——S0 模型漏了 PCIe **带宽项**：publish 64KB @ kernel-store 4GB/s = 16µs 单向，无法靠协议优化消除（Morrowmake 的 DMA 两段式也是 6.7GB/s 物理上限）。58µs ≈ 16(publish) + ~12(barrier 往返) + 8(gather 读) + launch/gap。

**S2（bf16 payload）价值上调**：字节数减半直接砍带宽项 → 预估再 −1.5-2ms/步（v1 → ~14ms，structured → ~135-140）。S3（减 kernel 数）不动。

## 堆叠进度

| 里程碑 | structured | 累计 |
|---|---:|---|
| TF 4-rank 基线（DFlash2） | 102.8 | 34% |
| **+S1 shm all-gather** | **124.9** | **41.4%** |
| +S2 bf16 payload（预估） | ~135-140 | ~46% |
| +S3 减核（预估） | ~150-160 | ~52% |
| +S5/S6 内核（预估） | ~200-260 | 66-86% |
| 生产目标 | 301.9 | 100% |

收据：`data/decode_c1_tf4ab_s1_shm.json`。生产已恢复。下一窗口：S2 bf16 payload（30 行，位一致舍入序需放宽声明）+ S3。
