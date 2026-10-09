# E 完整版收据：hc_mix 折入 gather kernel（allgather_sum_mix）— 2026-10-10 — 位精确成立，速度持平（负收益关闭）

任务 E 完整版按 HANDOFF-300-ROADMAP.md 任务 E 节规格与四条实现红线实施（<GPU_HOST> · TensorFold066 · GLM-5.3-Flash EXL3 4bpw · 4×CMP 170HX）。
**验收判定：exactness ✅（双重证实）/ structured 169.6 < 173 ❌ → 按纪律默认门关闭，170.1 栈保持默认。**

## 实现（规格逐条落地）

- 新 kernel `allgather_sum_mix`（`cuda/shm_allgather.py`，扩展名升 `tensorfold_shm_allgather_v2`）：
  phase 1/2 与 allgather_sum 完全一致（publish stripe + per-(parity,block) Flag 屏障 + seq 计数，红线 2）；
  phase 3 per unit i（r=i/dunits, d4=i%dunits, d=d4*4+j）：branch_j = Σ_rank fp32(bf16 slot) 按 rank 序累加
  → 单次 bf16 圆整 → fp32（红线 1）；4 条流 v_s = ps_s·branch + mixed_s 写回 x 原地。
- FMA 形状按 <GPU_HOST> Triton cache 实测 PTX 逐位复刻：`mixed = fma(x3,c3, fma(x2,c2, fma(x0,c0, x1*c1)))`
  （c1·x1 独立 mul 作种子，c0·x0 折入首个 FMA），`v = fma(branch, ps, mixed)`，`cvt.rn` 圆整。
- sync 位置按红线 3：4 slot 读 + 4 x 读全部进寄存器 → `__syncthreads()`（均匀迭代数保证收敛）→ 16 写。
- 接线：`forward.py gather(fold_mix)` 显式参数贯穿 `out_proj/kda_block/dsa_block/_dsa_latent/mlp_block/moe_block`，
  `layer_forward` 两处 `hc_fold` 以 `g is not None` 守卫；`TF_FUSE_GSUM_MIX` 门控；
  **mtp.py 的 `dsa_block/moe_block` 直调路径默认不折**（红线：冷路径走旧 `residual_fold`）——实测确认 mtp 直调两个块函数，参数默认 False 即旧径。

## 验收

| 关卡 | 结果 |
|---|---|
| 4-rank 位精确 A/B 单测（kernel vs `all_gather_sum` + Triton `_hc_mix`） | **7/7 用例 `torch.equal` 全等**（rows∈{1,2,3,4,6,8}×D=4096、D=256、D=1024；含部署形状 4×4096） |
| exactness（draft:true vs draft:false，greedy 400 tok） | **match=True，draft_sha = serial_sha = `0436adfa`** —— 与全部历史配置逐位一致 |
| structured（3 轮中位，MiaAI 协议） | **169.6**（169.6/169.7/169.6）< 验收线 173 |
| coding / prose | 129.6 / 102.4（对照 fbgb 133.1 / 93.9；efuse 129.2 / 102.2） |
| accept_ratio / tok_per_step | 0.757 / 3.38（与历史一致，drafter 行为未受影响） |

收据：`<GPU_HOST>:~/bench/tf4ab_dflash_gsum_mix2.json`（折入版）；
`tf4ab_dflash_gsum_mix.json`（意外首跑：缺 `TF_SHM_AG=1 TF_BF16_GATHER=1`，NCCL 回落 103.0，
**exactness 同样 `0436adfa`**——回落路径位同构设计的又一次免费实证）。

## 归因：为什么 +1-2.5% 的预估没有兑现

预估把「省 90 次 launch/步 × 3-5µs」按 eager launch 计价。实测 decode 全程跑在 CUDA graph 里，
hc_fold 的独立 kernel 是 graph **节点**而非 eager launch——replay 时节点间开销 ~0.5µs 级，
90 节点全省也只有 ~几十 µs/轮（轮耗时 ~12ms），落在 ±0.5% 噪声带内（169.6 vs 170.1）。
spart 写读往返（R×8KB×2）同理远小于估计。**五维度闭环再添一条：graph 内融合节点的收益上限 ≈ 0。**

速度指纹（169.6/129.6/102.4）与 2026-10-09 被撤销的 efuse 草稿（169.2/129.2/102.2）一致——
该草稿当时已隐约测到同一负结果，本次以位精确实证 + 根因归因正式关闭。

## 处置

- `TF_FUSE_GSUM_MIX` **默认 `0`**（2026-10-10 已部署 <GPU_HOST> 066 树）：默认栈回到已归档的 170.1 配置
  （S1 + S2 + fbgb + D+F，kernel 序列与 10-09 收据完全一致）。代码保留：位精确已双重证实，
  eager/无 graph 场景或上游采用 in-kernel mix 时可直接 `TF_FUSE_GSUM_MIX=1` 复用。
- 测试脚本：`<GPU_HOST>:~/tensorfold-poc/test_gsum_mix.py`（4-rank A/B，可复跑）。
- 生产 Morrowmake 窗口进出照常，恢复验证见 handoff（288-289 线）。
