# W4A16 affine 路径窗口收据 — 2026-10-09（夜）— 内核假设证实，路径级回归待诊断

**新立项第一夜（目标：压验证边际行成本 → 深化草稿链 → 冲 300）。结果：曲线假设证实，但端到端出现路径级回归，需 nsys 定位。生产无损。**

## 假设与实验设计

第一性分析（见会话）：TF 与 Morrowmake 的 1.7× 差距集中在**每步接受 token 数**（3.38 vs 6.4），根变量是**验证边际行成本**（EXL3 +3.4ms/行 vs Marlin ~2.2ms/行）。TensorFold 原生支持 affine（MLX 4-bit group-64）权重的 qmm 路径——qmm.cu 是 **M-tile 化 W4A16 GEMM**（BM×BN tile + split-K，多行共享权重 tile），理论边际成本应优于 EXL3 逐行解码。官方 affine checkpoint **已在盘**（`~/tensorfold-poc/weights/glm53-mlx-4bit-mtp`，170G，config: bits=4/group=64/affine）——零代码改动，仅换 CKPT 开窗。

## 窗口结果（tf4_serve066_mlx.sh dflash，收据 `tf4ab_dflash_mlx_affine.json`）

| 指标 | EXL3（tr3-4bpw） | affine（mlx-4bit） | 判定 |
|---|---:|---:|---|
| v1 基础前向 | 14.0 ms | 13.8 ms | ≈持平 |
| 边际行成本 | +3.4 ms/行 | **+2.57 ms/行** | ✅ **−25%，内核假设证实** |
| accept_ratio | 0.757 | 0.771 | ✅ 略升 |
| drafted/step | ~4.46 | 4.07 | ❌ 反而更浅 |
| 实际每步墙钟 | 19.9 ms | **33.5 ms** | ❌ **+68%** |
| structured | 170.1 | **93.8** | ❌ 端到端减半 |
| exactness | `0436adfa` | **`0436adfa`** | ✅ 两种量化输出逐位一致 |

## 判读

1. **内核级假设证实**：qmm M-tile GEMM 的边际行成本确实比 EXL3 逐行解码低 25%——多行权重摊薄的方向正确。
2. **但存在探针/实战反转**：affine 的 v 探针（fastest-of-7）快于 EXL3，实际每步却慢 68%。EXL3 实战（19.9ms）比自己的探针（24.3ms）还快（图重放摊销），affine 实战（33.5ms）比探针（21.4ms）慢 57%——**affine 解码路径存在探针不覆盖的每轮开销**。
3. 候选根因（下窗 nsys 定位，serve 脚本已带 `TF_NSYS_DELAY/DURATION` 钩子）：affine 路径图捕获缺失/不完整（日志无 capture 行）；非 EXL3 的 grouped MoE 内核（`grouped.gate_up/down` 通用路径）在 sm_80 上慢；`decode_seconds_total` 计数器在该路径为 0（1211 rounds）本身即线索。
4. 接受率 0.771 vs 0.757：affine 保真度略优于 EXL3 4bpw，方向符合预期但幅度小（counting 任务太可预测，区分度低）。

## 生产与流程

- 窗口进出正常，生产恢复复验 **structured 288.9 tok/s**（any_nan=false，收据 `<GPU_HOST>:~/bench/verify_after_mlx_exp.json`）。
- 复现：`sed "s|glm53-tr3-4bpw|glm53-mlx-4bit-mtp|" tf4_serve066.sh > tf4_serve066_mlx.sh`；`TF_SHM_AG=1 TF_BF16_GATHER=1 bash tf4_serve066_mlx.sh dflash`。

## 下一步（下窗）

1. `TF_NSYS_DELAY=60 TF_NSYS_DURATION=180` 对 affine 路径做 nsys，定位 33.5ms 的构成（graph 缺失 vs grouped MoE vs 其他）；
2. 若为 grouped MoE：考虑把 Marlin 微基准已验证的 expert 段接给 affine 路径（marlin_standalone 已在盘、M=4 实测与 EXL3 同速）；
3. 若为图捕获缺失：补 affine 路径的 capture 支持（改动面在 decode.py 的 capture 条件）。
