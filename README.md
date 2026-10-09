# TensorFold · GLM-5.3-Flash · 4× Intel CMP 170HX — 已验证部署方案

> **English TL;DR** — This repository is a complete, tested-working deployment of [TensorFold](https://github.com/ashhart/TensorFold) (v0.6.6 + our modifications) serving **GLM-5.3-Flash** on **4× Intel CMP 170HX** (GA100-class `sm_80` compute GPUs over PCIe Gen2 x4, no P2P): bit-exact 4-rank tensor parallelism, a host-staged shared-memory all-gather, bf16 gather payloads, block-diagonal fused projections, context copy-drafts, cost-aware draft depth and a widened CUDA-graph capture (verify windows R=1..8), plus MLX-affine W4A16 checkpoint support with DFlash2 + MTP speculative decoding. Measured **199.4 tok/s structured (single stream)** on the in-repo stack — **+17.2% over the EXL3-stack plateau (170.1)** — with the exactness contract `drafted == serial` (SHA-256 `0436adfa…`) maintained across quantization formats. A production reference stack on the same hardware serves 288–289 tok/s.

---

## 1. 这是什么

本仓库包含：

1. **完整可工作源码树** —— 上游 TensorFold `v0.6.6` + 我们针对 4×CMP 170HX 的**全部修改**（20 个文件改动 + 3 个新文件，见 [`cmp170hx/CHANGES.md`](cmp170hx/CHANGES.md)），与实测通过时的部署状态逐文件一致。
2. **完整差异** —— [`cmp170hx/full-changes-vs-v0.6.6.patch`](cmp170hx/full-changes-vs-v0.6.6.patch)（相对上游 v0.6.6 的全部修改，可复核）。
3. **运行/基准脚本** —— [`cmp170hx/scripts/`](cmp170hx/scripts/)：4-rank 启动器（EXL3 与 W4A16 affine 两档）、MiaAI 协议基准、扩展预编译、位精确 A/B 测试。
4. **测试结果报告** —— [`cmp170hx/results/`](cmp170hx/results/)：S 系列实验（S0 画像 / S1 shm / S2 bf16 / S3b / S5）、内核对打、W4A16 窗口、突破收据与总表 TEST-MATRIX。

上游原 README 保留在 [`docs/UPSTREAM-README.md`](docs/UPSTREAM-README.md)，GLM-5.3-Flash 配方见 [`docs/recipes/glm-5.3-flash.md`](docs/recipes/glm-5.3-flash.md)。

### 为什么 CMP 170HX 需要"另一套"方案

CMP 170HX 是 GA100（sm_80）核心的算力卡，与上游 CUDA 主线（Ada sm_89+，依赖 FP8 MMA 与 NVFP4/clusters）有三点硬冲突：

| 上游假设 | CMP 170HX 现实 | 本仓库对策 |
|---|---|---|
| 构建门槛 `MIN_CAPABILITY = 8.9`（FP8 MMA） | sm_80，无 FP8 | `TENSORFOLD_MIN_CAPABILITY=8.0` 环境变量准入（只降不升，FP8/NVFP4 路径仍拒绝） |
| NVFP4/cluster 扩展需要 sm_90+ | sm_80 | e4m3 内联 PTX 全部加架构守卫，sm_80 可编译、FP8 路径 trap 拒绝 |
| 双机 TP2（NVLink-C2C 场景） | 单机 4 卡、PCIe Gen2 x4、**无 P2P runtime** | `--tp 4`（rank 0–3）+ 主机共享内存 all-gather（S1）绕开 NCCL 小消息延迟 |

## 2. 实测结果（4×CMP 170HX · GLM-5.3-Flash · MiaAI 协议 3 轮中位）

| 配置 | structured | coding | prose | 增量 |
|---|---:|---:|---:|---|
| 串行 `--no-drafts` | 46.7 | 46.7 | 46.6 | — |
| NCCL + DFlash2（4-rank） | 102.8 | 80.8 | 61.4 | — |
| **+ S1** 主机共享内存 all-gather（`TF_SHM_AG=1`） | 124.9 | 98.1 | 75.1 | **+21.5%** |
| **+ S2** 通信载荷 bf16（`TF_BF16_GATHER=1`） | 169.8 | 132.7 | 93.6 | 累计 **+65%** |
| + fbgb 块对角融合 / 成本感知深度 / copy-drafts | 170.1 | 133.1 | 93.9 | EXL3 栈定格 |
| **W4A16 affine + `GRAPH_ROWS=(1..8)`** | **199.4** | **142.6** | **101.5** | **+17.2%（本仓库栈纪录）** |
| 生产参考（Morrowmake 1.7.2 · TP4 · 同硬件） | 288–289 | — | — | 对照 |

- **exactness（硬门槛）**：每一行配置下 `drafted == serial`（SHA-256 一致），且谱系 `0436adfa…` 跨量化格式（EXL3 4bpw ↔ W4A16 affine g64）逐位一致。
- 附带实测：W4A16 affine 的草稿接受率更高（0.776 vs 0.757；探针口径 accept 0.97 / 6.25 tok/轮）——量化保真度直接转化为投机解码收益。
- 诊断工具：`TF_DECODE_PROF=1`（首个 eager 轮的 torch.profiler 内核表，`decode.py` 内置）。本轮靠它定位了 eager 轮 shm gather 自旋占 92% 的真凶，从而一行扩展图捕获（`GRAPH_ROWS=(1..8)`）拿到 +17.2%。

## 3. 硬件 / 软件环境

| 项 | 参考值 |
|---|---|
| GPU | 4× NVIDIA CMP 170HX 64GB（GA100，**sm_80**），PCIe Gen2 x4，无 P2P（可选 ForceP2P） |
| 驱动 / CUDA | 610.43.03 / torch `2.13.0+cu130` |
| Python | 3.12 |
| 检查点 | **EXL3 4bpw**：`brandonmusic/GLM-5.3-Flash-tr3-4bpw`（HF；国内镜像 `HF_ENDPOINT=https://hf-mirror.com`）；**W4A16 affine**：MLX 4-bit g64 + 内嵌 MTP 头（约 170 GB） |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2`（DFlash2 块草稿；每 rank 仅 +0.56 GiB） |

> 权重与 drafter **不随本仓库分发**，请遵循其各自许可证。

## 4. 修改清单（全部，详见 [cmp170hx/CHANGES.md](cmp170hx/CHANGES.md)）

相对上游 v0.6.6：**17 个文件修改 + 3 个新文件**，完整 diff 见 [`cmp170hx/full-changes-vs-v0.6.6.patch`](cmp170hx/full-changes-vs-v0.6.6.patch)。

| 主题 | 文件 | 一句话 |
|---|---|---|
| sm_80 准入 | `cuda/build.py` | `TENSORFOLD_MIN_CAPABILITY` 只降不升 + Ampere 提示 |
| sm_80 可编译 | `cuda/kernels/qmm_prefill8.cu` | e4m3 内联 PTX 的架构守卫 |
| **S1** shm all-gather | `cuda/_shm_allgather.cu`（新）+ `cuda/shm_allgather.py`（新）+ `cuda/comm.py` | world=4 主机共享内存 one-shot all-gather（arrival-flag 协议，位精确，可图捕获），NCCL 兜底 |
| **S2** bf16 载荷 | `glue.py` / `forward.py` | gather 路径 bf16（位保持），`TF_BF16_GATHER=1` |
| fbgb 融合 | `glue.py` / `weights.py` | `f_b+g_b` 块对角 [2N, K+K] 单 GEMM，`TF_FUSE_FBGB=1` |
| **TP4** | `cli.py` / `cli_args.py` / `engine.py` / `weights.py` / `split.py` | `--tp 4`、rank 0–3、`RankReader(ranks=world)`（头/组边界保持） |
| 图捕获扩展 | `engine.py` | `GRAPH_ROWS=(1..8)`（原 1..6），R=7/8 验证窗入图重放 |
| copy-drafts | `glm5_next/cuda/copy_drafts.py`（新）+ `drafter_choice.py` | 上下文复制草稿移植（`TENSORFOLD_COPY_DRAFTS=0` 可关） |
| 成本感知深度 | `drafter_choice.py` | D+F 逐轮臂选择，抗分布漂移 |
| W4A16 affine | `weights.py` | MLX affine 4-bit g64 检查点加载（与 EXL3 同一引擎） |
| 诊断 | `decode.py` | `TF_DECODE_PROF=1` 内核表钩子 |
| Qwen 家族同步 | `qwen4_exp/cuda/engine.py` | 同一套 world/graph 处理 |

## 5. 快速开始

```bash
git clone https://github.com/liumorrisclaw/TensorFold-CMP170HX.git
cd TensorFold-CMP170HX
pip install -e .            # 或先跑 cmp170hx/scripts/prebuild066.py 预编译四大扩展（约 1 小时，可复用）

# 权重（按上面第 3 节自行下载；脚本默认目录见 cmp170hx/scripts/tf4_serve066.sh 头部）
#   EXL3:      ~/tensorfold-poc/weights/glm53-tr3-4bpw
#   W4A16:     ~/tensorfold-poc/weights/glm53-mlx-4bit-mtp
#   DFlash2:   ~/glm53-recipe/models/GLM-5.3-Flash-DFlash2

# EXL3 4-rank serve（dflash profile）
TF_SHM_AG=1 TF_BF16_GATHER=1 TENSORFOLD_MIN_CAPABILITY=8.0 \
  bash cmp170hx/scripts/tf4_serve066.sh dflash

# W4A16 affine（纪录配置，GRAPH_ROWS=(1..8) 已内置）
TF_SHM_AG=1 TF_BF16_GATHER=1 TENSORFOLD_MIN_CAPABILITY=8.0 \
  bash cmp170hx/scripts/tf4_serve066_mlx.sh dflash

# 基准（3 轮中位 + exactness）
python cmp170hx/scripts/tf4_ab_bench.py --profile dflash --tag my_run --exactness
```

加载约 15–25 分钟（164 GB 权重按 rank 分载；首次需构建扩展，之后由 `TORCH_EXTENSIONS_DIR` 复用）。READY 后 `curl 127.0.0.1:8899/health`。

## 6. 运行开关

| 变量 | 作用 | 默认 |
|---|---|---|
| `TENSORFOLD_MIN_CAPABILITY=8.0` | sm_80 准入（**必须**） | — |
| `TF_SHM_AG=1` | S1 主机共享内存 all-gather（world=4） | 关 |
| `TF_BF16_GATHER=1` | S2 gather 载荷 bf16 | 关 |
| `TF_FUSE_FBGB=1` | f_b+g_b 块对角融合 | 关 |
| `TF_FUSE_GSUM_MIX=1` | hc_mix 折入 gather 内核（实测持平，保留复用） | 0 |
| `TF_DECODE_PROF=1` | 首个 eager 轮内核表（torch.profiler） | 0 |
| `TENSORFOLD_COPY_DRAFTS=0` | 关闭 copy-drafts | 开 |
| `GRAPH_ROWS` | `engine.py` 常量，本仓库默认 `(1..8)` | — |

## 7. 仓库结构

```
├── src/ tests/ tools/ docs/ assets/   # TensorFold 源码树（v0.6.6 + 本仓库全部修改）
├── cmp170hx/
│   ├── CHANGES.md                     # 逐文件修改说明
│   ├── full-changes-vs-v0.6.6.patch   # 相对上游 v0.6.6 的完整 diff
│   ├── patches/                       # 当初提交上游的 3 个 patch（PR #464/#465 系）
│   ├── scripts/                       # serve / bench / 预编译 / 位精确 A/B 脚本
│   └── results/                       # 测试报告与收据（TEST-MATRIX 总表等）
└── docs/UPSTREAM-README.md            # 上游原 README（备份）
```

## 8. 上游协作

- **PR #464**（sm_80 准入）与 **PR #465**（GLM 4-rank）已由上游 reland 进 main 并署名（`23f4e7f`/`c87248d`/`f39624d`）。
- **PR #466**（shm all-gather fresh 重写，`ecf6848`）处于 OPEN 状态，等待上游 re-review。
- 本仓库的 shm all-gather 与上游 `python-0.6` 分支的 `comm.py` 零漂移，可直接合入。

## 9. 许可与致谢

- TensorFold 由 [ashhart](https://github.com/ashhart/TensorFold) 与 TensorFold contributors 开发，**Apache-2.0**（0.6.0 起；此前 MIT，见 `LICENSES/` 与 `NOTICE`）。本仓库保留原始 `LICENSE`/`NOTICE`/`THIRD_PARTY_NOTICES.md`，修改部分同样以 Apache-2.0 发布。
- 感谢上游在 PR #464/#465 review 中的快速反馈；shm all-gather 的 arrival-flag 协议由本方案设计并在 4×CMP 170HX 上实测（见 `cmp170hx/results/`）。
- GLM-5.3-Flash 权重、DFlash2 drafter 归其各自权利人所有，本仓库不含任何模型权重。
