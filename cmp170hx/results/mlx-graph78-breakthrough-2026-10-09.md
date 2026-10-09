# W4A16 + GRAPH_ROWS=1-8 突破收据 — 2026-10-09 — TF 栈新纪录 199.4 tok/s（+17.2%），exactness 跨量化维持

**新目标（300，第一性原理方案）重大突破：affine/W4A16 路径 + 图捕获扩展到 R=7/8 后，TF 栈 structured 实测 199.4 tok/s——超过 EXL3 栈定格值 170.1 达 +17.2%，且 exactness 谱系 `0436adfa` 跨量化格式维持逐位一致。**

## 决定性诊断链（当夜三步）

1. **TF_DECODE_PROF 钩子（新工具）**：env 门控 torch.profiler 打印首个 eager 轮内核表——比 nsys 可靠，零时序博弈。
2. **profiler 表抓到真凶**：eager R=7 轮 CUDA 时间 197.4ms 中 **`tfshm::gather_sum` 占 92.08%（181.8ms，90 次调用 × 2.02ms）**——其余全部内核合计 <8%。根因不是 grouped MoE，是 **eager 模式下 4-rank 各自数百次 kernel launch 造成的 rank 偏斜 → shm gather 自旋等待对端arrival flag**。
3. **连带发现**：affine 路径接受率质量惊人——drafted 探针 `rounds=55 accepted=344 → 6.25 tok/轮（accept 0.97）`，W4A16 保真度高于 EXL3 4bpw 使草稿接受率大幅提升。但接受越好 → 验证窗口 R≈7 → **超出 GRAPH_ROWS=(1..6) → 每轮全 eager → gather 自旋 76ms/轮 → 只有 94.7 tok/s**。良性循环被图覆盖范围卡死。

## 修复（一行）

`engine.py: GRAPH_ROWS = (1,2,3,4,5,6)` → **`(1,2,3,4,5,6,7,8)`**。R=7/8 验证窗进入图重放：eager 自旋消失（图内 4-rank 同步收紧），轮时 76ms → ~32ms。

## 窗口结果（收据 `tf4ab_dflash_mlx_graph78.json`）

| 口径 | EXL3 栈（170.1 定格） | **affine + GRAPH_ROWS=1-8** | Δ |
|---|---:|---:|---|
| structured | 170.1 | **199.4** | **+17.2%** |
| coding | 133.1 | **142.6** | +7.1% |
| prose | 93.9 | **101.5** | +8.1% |
| drafted 探针（200tok） | — | **203.5 tok/s** | — |
| exactness | `0436adfa` | **`0436adfa`** | 跨量化逐位一致 |
| accept_ratio | 0.757 | 0.776 | +2.5% |

生产恢复复验 **288.3 tok/s**（any_nan=false，收据 `verify_graph78.json`）。

## 距 300 的剩余分解

当前 drafted 稳态 ~200 tok/s、accept 0.976（探针口径 6.25 tok/轮）。轮时 ~30ms = 图重放 R=7（~21ms，v 曲线 22.6 基础 + 边际 2.6×3）+ draft 4.4ms + 编排 ~5ms。到 300 需轮时 ≤21ms：
- 编排/采样 ~5ms：图外 Python 部分，可压缩但工程量中等；
- draft DFlash2 3.4~4.4ms：草案模型自身；
- 图内 R=7 验证 ~21ms：qmm M-tile 内核调优（BM/BN/stages 扫描）或 deeper tile；
- 或 accept 继续涨（0.976 已近顶）。

**判断：300 从"实测不可达"变为"路径明确、剩余 1.4×，全部在工程范围内"**——下一步按 qmm kernel 调优 + 编排压缩推进（另立窗口，不动生产）。

## 复现

```bash
# engine.py: GRAPH_ROWS = (1,2,3,4,5,6,7,8)
sed "s|glm53-tr3-4bpw|glm53-mlx-4bit-mtp|" tf4_serve066.sh > tf4_serve066_mlx.sh
TF_SHM_AG=1 TF_BF16_GATHER=1 bash tf4_serve066_mlx.sh dflash
venv/bin/python tf4_ab_bench.py --profile dflash --tag mlx_graph78 --exactness
```

工具课：TF_DECODE_PROF=1（decode.py 新增钩子）输出 eager 轮内核表——nsys 三连挫后的可靠替代，本轮立功。
