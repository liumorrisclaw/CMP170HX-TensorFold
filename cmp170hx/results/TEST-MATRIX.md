# TEST-MATRIX：全部已做实验与结论总表（防重复测试主文档）

> **用法**：任何新想法动手前，先查此表。已有结论的项目不得重复开窗/微基准；确需复核时引用原收据行。新实验完成后必须回写本表（一行：结论 + 收据路径）。
> 覆盖窗口：2026-10-03 ~ 2026-10-09（GLM-5.3-Flash · 4×CMP 170HX · TensorFold 0.6.6 + Morrowmake 生产）。
> 维护规则：本表与 `HANDOFF-300-ROADMAP.md` 汇总表同步；冲突时以收据 JSON 为准。

---

## A. TensorFold EXL3 栈（GLM 4×170HX）——有效收益（已落地默认开）

| # | 实验 | 结论 | 收据/报告 |
|---|---|---|---|
| A1 | S1 主机共享内存 all-gather（TF_SHM_AG=1） | +21.5%（102.9→124.9），取代 NCCL | `tf4ab_dflash_copydraft_s1.json` |
| A2 | S2 通信载荷 bf16（TF_BF16_GATHER=1） | +35.8%（→169.7），逐位一致 | `tf4ab_dflash_copydraft_s2.json` |
| A3 | fbgb 块对角融合 | +0.2%，保留默认开 | `tf4ab_dflash_fbgb.json` |
| A4 | D+F 成本感知解码深度 | 持平但抗分布漂移，默认开 | `tf4ab_dflash_df_costdepth.json` |
| A5 | copy-drafts 移植 | 重复文本 2.4×；structured 口径零命中（by design） | `tf4ab_dflash_copydraft_s2.json` |
| A6 | 当前定格 | **structured 170.1 tok/s**（EXL3 栈全局最优），exactness 谱系 `0436adfa` 全程 | `tf4ab_dflash_fbgb.json` 等 |

## B. TensorFold——无收益/已关闭（**勿重复实验**）

| # | 实验 | 原预期 | 实测结论 | 收据/证据 |
|---|---|---|---|---|
| B1 | 分块流水 gather（E-P2） | +3~4% | **+0.1%，关闭**。带宽饱和：blocks=1/2/4 均 ~96µs；36.9→95.9→175.7µs @ R=1/4/8（斜率=0.62µs/KiB 物理域）；字节已在 bf16 地板 | `results/e-pipeline-refuted-2026-10-10.md`、`bench_gather_latency.py` |
| B2 | hc_mix 折入 gather 内核（E-min/E 完整版） | +1~2.5% | **持平（169.6），关闭**。图内 launch 节点开销 ≈0；位精确 21/21 + `0436adfa`；TF_FUSE_GSUM_MIX 默认 0，代码保留（=1 可复用） | `results/e-full-gsum-mix-2026-10-10.md`、`tf4ab_dflash_gsum_mix2.json`、`test_gsum_mix.py` |
| B3 | NCCL + P2P 直通（C2） | 提速 | **持平（102.9 vs 102.9）**，shm 方案胜出并落地 | `tf4ab_dflash_c2_nccl_p2p.json` |
| B4 | 换官方量化权重（P1） | 突破瓶颈 | **双向证伪**：官方 MLX 格式在自家硬件（2×GB10）仅 43~85 tok/s | `results/` P1 关闭记录、docs/recipes/glm-5.3-flash.md:122 |
| B5 | 激进解码策略 / 固定深度覆盖 | 超过 auto | **七轮全部落败**，auto 已在成本最优点 | `results/` 策略收据组 |
| B6 | B-full 多投影融合 | +0.1~0.3% | **预估低于噪声，暂缓未测**（依赖链限制，可融合对子所剩无几） | handoff 任务 B 节 |
| B7 | 生产 PP3 三卡流水 | 单流更优 | **无法稳定启动**（Xid31 MMU/CE2 驱动级），回滚 PP4 留档 | `results/m2-2026-10-03.md` 附录 |
| B8 | 月级内核重写前的软件路线 | — | **全部穷尽**：五维度闭环（内核同速/通信物理域且分块不敏感/权重双证伪/策略七次卫冕/图内融合≈0） | `HANDOFF-300-ROADMAP.md`、`kernel-challenge-final-2026-10-08.md` |

## C. affine（W4A16，MLX 4-bit）新线 —— 2026-10-09 夜首窗（新立项 300）

| # | 实验 | 结论 | 状态 |
|---|---|---|---|
| C1 | affine 路径可行性 | **零代码可 serve**（config 驱动格式分发；checkpoint 已在盘 `~/tensorfold-poc/weights/glm53-mlx-4bit-mtp`，**无需下载/转换**） | ✅ 已证 |
| C2 | qmm 边际行成本 | **+2.57 ms/行（EXL3 为 +3.4，−25%）——内核假设证实**；v1 基础 13.8 vs 14.0 | ✅ `tf4ab_dflash_mlx_affine.json` |
| C3 | affine 接受率 | 0.771（EXL3 0.757），略升 | ✅ 同上 |
| C4 | affine 位精确 | exactness **`0436adfa`**（与 EXL3 及全部历史配置逐位一致） | ✅ 同上 |
| C5 | affine 端到端 | **93.8 tok/s**；探针拆分（夜）：串行单行 **14.6 ms/tok ✅健康**（≈EXL3 v1 14.0，主前向+图重放无恙）；草稿步 **33.2 ms ❌**（verify(4行) 21.4 + draft 4.4 + **~8ms 未归因**）。每步慢于 EXL3 的部分集中在**多行验证 GPU 工作或每轮编排**，需 profile 级归因 | ❌ 待诊断 |
| C6 | 回归根因定位 | **nsys 工具课（3 次受挫）**：①`nsys profile --duration` 到期**杀死应用**（rank0 两次被带崩）；②本机 nsys 2026.1.3 的 `launch` 拒绝 `--cuda-graph-trace`/`-o`（`Sessions` 子命令也不存在）；③delay 必须≥实测 READY（4-rank 并发读盘加载 20-25min 且日志静默）。**正确姿势：扩展已全部缓存（启动≈15min 可预测）→ delay=READY+60s，duration 覆盖基准期**。**替代方案（已实施）**：decode.py 已加 `TF_DECODE_PROF=1` env 门控 torch.profiler 钩子（首个 eager 轮打印内核表，纯 Python 免重建）——下窗 `TF_DECODE_PROF=1` 起 serve，8 个 drafted 请求后从 r0 日志提取内核表 | ⏳ 钩子已就位，待下窗 |
| C7 | 修复路径 | 若 grouped MoE 慢→接 Marlin expert 段 | ⏳ **实施中**，方案已固化见下 |
| C7a | **Marlin expert 段接入作战卡（2026-10-09）** | **目标**：expert 段 7.2→~2.8ms/轮 → structured ~237，叠加 v8 深草稿 → ~250-260。**配方（已验证的调用序）**：`ops.decode_gemm(hidden, w1, s1, sorted_ids, expert_ids, ntpp, topk_w.view(-1), TOPK, n_slots, K, 2N, True, W13_ROWS=64, 0, ctr, part)` → `ops.decode_act(part, topk_ids.view(-1).int(), ksplit, slots, N=512, 10.0, h)` → `ops.decode_gemm(h, w2, s2, …, N→K, False, W2_ROWS=N//16, 0, ctr, c3)`；ops 来自 `/home/<user>/tensorfold-poc/marlin_standalone/_ampere_marlin_C.abi3.so`（torch.ops._ampere_marlin_C）。**权重格式**：w1 int32 [E,K/16,2N]、s1 bf16 [E,K/128,N]、w2 int32 [E,N/16,K]、s2 [E,N/128,K]（g128 对称，无 bias/zero）。**路由构建**：pairs=(topk_ids==E).nonzero() 按 expert 分块 8 对齐 → sorted_ids[blk*8+j]=t*TOPK+k、expert_ids[blk]=e、ntpp[e]=count；生产级需固定 shape（pad 至 56+288×7 上限，无效块 early-exit）。**bias 折叠**：affine g64 含 bias → dequant 后 Σ_k bias 得每专家常量输出偏移 B13[n13]（act 前加）、B2[hidden]（combine 时按 slot 的 expert gather 加）。**权重来源两条路**：(a) 生产 checkpoint `GLM-5.3-Flash-W4A16-MTP`（compressed-tensors g128 对称，178G 在盘）——Morrowmake 生产每次启动都把它 repack 成 Marlin（fork 的 repack 代码待定位：venv site-packages 无独立 marlin utils 文件，编译进了 `_ampere_marlin_C` 或进程内；下一步查 Morrowmake 源码 checkout/安装脚本）；(b) affine Q4 dequant→g128 对称 requant（二代量化，保真度损失小、counting 任务鲁棒）。**接线**：moe_block decode 路径 `TF_MARLIN_EXPERTS=1` 门控替换 `grouped.gate_up/down`；输出 c3 [slots,K] 经 glue.combine（ey 视图 + B2 偏移）。**验收**：expert 段微基准 ≤3ms/轮 + drafted==serial + structured ≥240。**已证锚点**：marlin_microbench M=4 对齐块 62µs/层（EXL3 同段 160µs）。**repack 已定位（2026-10-09 深夜）**：GPU op `ops.gptq_marlin_repack(b_q_weight=int32 view.T.contiguous(), size_k, size_n, num_bits=4)` + `marlin_utils.py:marlin_permute_scales(s, size_k, size_n, group_size=128)`——均在 `_ampere_marlin_C`/marlin_utils.py（vllm-src:quantization/utils/），生产启动每次都在跑同款。**转换器配方（affine g64 → marlin g128 sym）**：①按 qmm lane 布局（OFFSETS=(0,8,16,24,1,9,17,25)，nibble p of word lane → input 32v+2×(lane%4)+OFFSETS[p]）dequant affine Q4 → fp16 [n,k]；②g128 对称 requant（scale=max|w|/7 per (128k,n)）；③gptq_marlin_repack + marlin_permute_scales；④bias 折叠 B[n]=Σ_k bias（w13 在 act 前加、w2 在 combine 按 expert gather 加）。全部 GPU op、零下载 | ⏳ 实施中 |
| C8 | 理论锚点 | 每步回 ~20ms 且边际 2.57 兑现 → **~240-260 tok/s**；300 需再压边际或 x16 | 公式推算 |
| C9 | 启动工具课 | ①中途 kill 会留半成品构建态 → 下次启动串行静默重建多个扩展（qmm_v5/experts_v7/glm_kda_v2，各 5-15min，**今夜已全部重建缓存完毕，后续 affine 启动恢复 ~15min**）；②构建锁残留报错按提示删 `tf_ext066/*/lock` 即可；③串行重建时各 rank 日志静默，勿误判死亡 | 经验 |

## D. Morrowmake 生产侧

| # | 实验/事件 | 结论 |
|---|---|---|
| D1 | 生产稳态 | **structured 288.6-289.3**（多次复验，any_nan=false），288-289 为恢复验证线 |
| D2 | PP3 布局 | 见 B7（负结果，已回滚） |
| D3 | 74SM 解锁 / 1.7.2 / ECC | 已落地，构成当前 289 的组成部分 |
| D4 | W4A16-MTP checkpoint | compressed-tensors 格式（178G，TensorFold **不能直接吃**，勿再尝试直连） |

## E. 上游（ashhart/TensorFold）

| # | 事项 | 状态 |
|---|---|---|
| E1 | PR #464/#465 | 已关闭——改动经 ashhart reland（`23f4e7f`/`c87248d`/`f39624d`）进 main 并署名 |
| E2 | PR #466（shm all-gather） | fresh 重写已交（commit `ecf6848` + 收据评论 issuecomment-6049448096）；等 ashhart 复跑重合检查与 re-review。值守基线：open / head `ecf6848` / comments=3；python-0.6 `ed78d6f`；release v1.0.2 |
| E3 | 1.0.x（Zig 引擎）可用性 | GLM 合格平台仅 Metal 双 M5 Ultra；**CUDA 矿卡场景暂不可用**，python-0.6 是官方维护线；每个 release 值得读（合格清单会扩） |

## F. 工具资产（复用，勿重写）

| 工具 | 用途 | 注意 |
|---|---|---|
| `~/tensorfold-poc/test_gsum_mix.py` | 4-rank 位精确 A/B（sum/mix/copy 三内核） | 子进程须各自 CUDA_VISIBLE_DEVICES |
| `~/tensorfold-poc/bench_gather_latency.py` | gather 微基准 | 整批背靠背计时（勿逐次 event）；已含 GPU 隔离 |
| `~/tensorfold-poc/marlin_microbench.py` | Marlin expert 段 M=4 对比 | 已证与 EXL3 同速（62-67µs/层） |
| `~/tensorfold-poc/test_shm_pr.py` | PR 变体（copies-only）4-rank 验证 | 5 用例 |
| `~/tensorfold-poc/tf4_serve066_mlx.sh` | affine checkpoint serve 脚本 | 配合 TF_NSYS_DELAY/DURATION 可采样 |
| `pcie_floor.py` | PCIe 物理域曲线 | 0.62µs/KiB 出处 |

## G. 操作纪律（本表配套）

- 窗口四步：停生产（19MiB 确认）→ serve+基准 → 停 TF → 恢复生产并复验 288-289；
- 生产恢复后必须跑 `bench_decode_mia --structured`（API_KEY 从 vllm 进程 environ 取，勿硬编码）；
- vllm 改名进程（`VLLM::EngineCore/Worker_TP*`）按 `nvidia-smi --query-compute-apps=pid` 清理，`bin/vllm` 模式会漏；
- exactness 硬门槛：任何改动后 `draft_sha` 必须等于 `0436adfa…`，否则立即回滚。
