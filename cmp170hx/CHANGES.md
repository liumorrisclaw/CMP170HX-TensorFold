# CHANGES — 相对上游 v0.6.6 的全部修改（逐文件）

> 基线：上游 [ashhart/TensorFold](https://github.com/ashhart/TensorFold) `v0.6.6`。
> 完整 diff：[`full-changes-vs-v0.6.6.patch`](full-changes-vs-v0.6.6.patch)。总计 17 个文件修改 + 3 个新文件。
> 全部改动在 4×CMP 170HX（sm_80）上实测通过；`drafted == serial`（SHA-256 `0436adfa…`）全程保持。

## 新增文件（3）

| 文件 | 行数 | 内容 |
|---|---|---|
| `src/tensorfold/cuda/_shm_allgather.cu` | +396 | **S1** 主机共享内存 one-shot all-gather 内核：arrival-flag 协议（per-parity flag + 生成计数），bf16/f32/i32 三种载荷位精确，CUDA-graph 可捕获（指针全为预分配缓冲，无 host 同步）。绕开 PCIe-only 场景 NCCL 小消息延迟。 |
| `src/tensorfold/cuda/shm_allgather.py` | +549 | S1 的 Python 绑定与 rank 间握手（`ShmAllGather(rank, world, store)`）：共享环形缓冲注册、barrier、`all_gather(send, recv)`；尺寸/_dtype 不在 shm 路径时抛 `ValueError` 由上层回落 NCCL。 |
| `src/tensorfold/families/glm5_next/cuda/copy_drafts.py` | +53 | **copy-drafts 移植**（上游 PR #468 的 GLM 版）：最近 token 段在 prompt/已生成文本中出现时，整段复制为草稿；两 rank 相同 token 提相同链；`TENSORFOLD_COPY_DRAFTS=0` 可关。 |

## 修改文件（17）

### sm_80 准入与编译

| 文件 | 改动 |
|---|---|
| `src/tensorfold/cuda/build.py` | +21 −1：`TENSORFOLD_MIN_CAPABILITY` 环境变量只降不升地放宽 `MIN_CAPABILITY`（sm_80 POC：W4A16/EXL3 路径；FP8/NVFP4 仍拒绝）；拒绝信息附 Ampere 提示；cluster 目标不受影响。 |
| `src/tensorfold/cuda/kernels/qmm_prefill8.cu` | +12 −0：e4m3 MMA/cvt 内联 PTX 加 `__CUDA_ARCH__ >= 890` 守卫，sm_80 可编译（FP8 prefill 路径在 sm_80 上 trap 拒绝）。 |

### 通信（S1/S2/E-min）

| 文件 | 改动 |
|---|---|
| `src/tensorfold/cuda/comm.py` | +18 −0：`NCCLComm` 在 `TF_SHM_AG=1 && world==4` 时构建 `ShmAllGather`；`all_gather` 优先走 shm 路径，`ValueError`（尺寸/dtype 不支持）回落 NCCL。 |
| `src/tensorfold/families/glm5_next/cuda/glue.py` | +67 −4：gather 链 bf16 载荷分支（**S2**，位保持）；`allgather_sum_mix` 折入内核（E-min，`TF_FUSE_GSUM_MIX`）；fbgb 块对角融合内核。 |
| `src/tensorfold/families/glm5_next/cuda/forward.py` | +69 −32：`gather()` 的 shm+bf16 分支接线、`fold_mix`、fbgb 单 GEMM 路径、4-rank 缓冲布局。 |

### TP4（--tp 4，单机 4 卡）

| 文件 | 改动 |
|---|---|
| `src/tensorfold/cli.py` | +10 −6：serve 命令把 rank 0–3 拉起同一命令（rank 0 供 HTTP） |
| `src/tensorfold/cli_args.py` | +5 −5：`--tp {1,2,4}`、`--rank {0..3}`，帮助文本更新 |
| `src/tensorfold/families/glm5_next/cuda/engine.py` | +31 −18：`open_comm(rank, world, …)`；`split_weights(rule, world)`；**`GRAPH_ROWS=(1..8)`**（原 1..6——R=7/8 验证窗入图重放，这是 W4A16 纪录 199.4 的关键一行）；`TF_GLM_DRAFT_RING` 等常量随 4-rank 调整 |
| `src/tensorfold/families/glm5_next/cuda/weights.py` | +14 −5：`load(..., world)` 参数化（原硬编码 2）；`RankReader(ranks=world)`；`TF_FUSE_FBGB=1` 时构建 `f_b+g_b` 块对角 [2N, K+K] BF16 权重；affine/EXL3 两格式沿用上游分发 |
| `src/tensorfold/families/glm5_next/cuda/split.py` | +40 −38：`RankReader(..., ranks=world)`——分片规则从 2-rank 推广到 4-rank，**头/量化组在分片边界保持**（位精确的前提） |
| `src/tensorfold/families/glm5_next/__init__.py` | +6 −5：家族注册随上述接口同步 |

### 草稿（copy-drafts / 成本感知深度 / DFlash2）

| 文件 | 改动 |
|---|---|
| `src/tensorfold/families/glm5_next/cuda/drafter_choice.py` | +74 −4：逐轮臂选择加入 copy-drafts 优先级（命中则本轮跳过 drafter）与成本感知深度（D+F；抗分布漂移） |
| `src/tensorfold/families/glm5_next/cuda/dflash2.py` | +6 −3：DFlash2 块草稿随 world/位宽参数化 |

### 诊断与其他

| 文件 | 改动 |
|---|---|
| `src/tensorfold/families/glm5_next/cuda/decode.py` | +7 −0：`TF_DECODE_PROF=1` 在首个 eager 轮打印 torch.profiler 内核表（self_cuda_time 排序，Top-25）——nsys 三连挫后的可靠替代，W4A16 突破的定位工具 |
| `src/tensorfold/families/glm5_next/cuda/mtp.py` | +2 −2：MTP 链随 world 参数化 |
| `src/tensorfold/families/qwen4_exp/cuda/engine.py` | +57 −27：Qwen 家族引擎同一套 world/图捕获处理（保持两家族行为一致） |
| `tests/test_cuda_cli.py` | +12 −0：`--tp 4` CLI 用例 |

## 提交上游的 patch 系列（历史存档，见 patches/）

1. `0001-lower-min-capability-via-env.patch` — sm_80 准入（→ 已由上游 reland，`23f4e7f`）
2. `0002-guard-e4m3-asm-for-sm80.patch` — e4m3 PTX 守卫
3. `0003-clusters-floor-and-ampere-hint.patch` — cluster 地板 + Ampere 提示

（S1 shm all-gather 即上游 **PR #466**，当前 OPEN；`comm.py` 与上游 `python-0.6` 零漂移。）

## 已证伪方向（勿重复，详见 results/TEST-MATRIX.md）

MLX 框架路径（−24%）、专家 pf=2（噪声）、`NCCL_PROTO=LL128`（−26%）、B16 tile 加大（持平）、hc/router 融合（设计否决）、NT4 专家 tile（噪声）、Q4 dense（coding −9%）、epilogue/Z（占比 0.46ms）、split-k=1（−2.6%）、策略扫描九点（auto 七次卫冕）、跨投影 stack 融合（数学否决）、分块流水 gather（带宽饱和，+0.1%）、hc_mix 折入（图内节点开销 ≈0）。
