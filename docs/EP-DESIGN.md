# 专家并行化（EP）设计 — 300 的唯一阻塞项

> **终态判定（2026-10-11 窗口 W1/W2）：工程目标全部达成，路线判定关闭。**
> EP 本身（显存、加载、路由、协议）全部验证通过；但承载它的 Marlin 专家内核是 **Morrowmake
> 的预编译二进制**，三条独立理由判死该路线：①违背"相关代码自研、不用别人二进制"的红线；
> ②端到端 exactness FAIL（bf16 下仍 FAIL，排除 fp8×EP）且同配置三轮 greedy sha 各不相同
> （C8a 通过窗口三轮全同 `0436adfa`）——该二进制端到端不可复现；③吞吐 167.2（fp8）/121.1
> （bf16）<< 全自研 affine fp8 栈 240.7（verify 行 +6.6ms → 深度求解器自动砍深度至 3.84
> tok/step）。**300 的路线回到全自研 affine fp8 栈 + 自研手段**；若需专家内核，自研重写
> （项目已具备 CUDA 能力与协议规范方法论），不再借用任何外部二进制。

## 实测收据（2026-10-11，收据 `tensorfold-sm80-poc/data/window-ep-1011/`）

| 关卡 | 结果 |
|---|---|
| U1（CPU，双机） | 分区/同一性/精确代数/lut-prefill/所有权全过 |
| U2（真机内核） | EpRoute R=1 vs R=7 **bit-exact**（\|d\|=0）；动态路由负对照正确 mismatch |
| U3 显存 | **44.3GiB/卡稳态**（行切分同路 83GB OOM；层 44 @39.1GB；mx=891MiB/层与推算一致）——affine 双载修复生效 |
| 加载健壮性 | 修出两个潜在 bug：MTP 层 prod ckpt 未打包格式（行切分因层 36 OOM 从未触达）；EpRoute ntpp 图捕获 H2D（FixedRoute 同款隐患，存档备案） |
| W1 组合窗（fp8） | structured 167.2 / coding 145.4 / prose 172.8；accept 0.884；**exactness FAIL** |
| W2 诊断窗（bf16） | structured 121.1；**exactness 仍 FAIL**；三轮 greedy sha 互异（非确定性） |

## 终态结论对三条红线的映射

- **自研红线**：Marlin .so = 别人的二进制 → 弃用。通信层 fresh 内核（我们自己的源码）不受影响。
- **位精确红线**：exactness FAIL + 运行间非确定 → 该二进制不可用于 drafted==serial 契约的栈。
- **性能证据**：C7a 的"distinct-expert DRAM 墙"结论再获端到端确认——专家段换格式不换速度；
  全自研 affine 栈仍是单流最优（240.7）。


## 0. D1 读码与实测修正（2026-10-11，重要）

实现当天读 `decode.cu`（Morrowmake ampere_marlin 内核，已存档 `reference/decode.cu`）并复核
内存账，**修正了交接文件里的三处判断**：

1. **无需重建 .so**（T5 关闭）。`blocks = ntpp[0] >> 3` 完全动态（decode.cu:120）；
   `eids[b]` 就是权重块索引（decode.cu:139,143）；`e0 < 0` 早退已在当前 P2 .so 里
   （decode.cu:140 "an unused block: no fragments, no writes"）；整专家实例化
   （w13 K=4096 N=4096 @ :352、w2 K=2048 N=4096 @ :376）与 TP 切片实例化并存。
2. **"专家显存 41→10GB"是算术错误**。72 个整专家与 288 个四分切片字节数相同
   （≈0.93GB/层，prod-mem 探针 mx=894MiB 佐证）——行切分从未多占专家显存。
3. **OOM 真因 = affine 双格式并存**。`weights.py` 的层加载循环
   `rd.prefetch(expert_names(i), up=dev)` 把 affine 全部 288 专家部件上传 GPU（~0.9GB/层），
   而 prod-marlin 路径只消费 shared——这就是交接账里 "~0.95GB/层其他权重" 的真身
   (0.9 marlin + 0.9 affine ≈ 1.85GB/层 ✓)。修复：prod-marlin 模式下 `expert_names` 只列
   shared（本次已改）。修正后预期总占用 ≈ 46-48GB，**无需 TF_CONTEXT=8192 削减即稳过**。

## 1. 问题与机会

现状 `convert_experts_prod` 是**行切分**：每卡持有全部 288 个专家的 1/4 行切片
（gate/up 取 `[:, rank*NI:(rank+1)*NI]` 行、down 取 k 维 `[:, rank*(NI//8):...]` 列），
因此每卡块数 = 289（288 路由 + 1 共享）→ 显存爆。

行切分的动机是通信：每卡算专家输出的 **k 维偏部分和**（partial sum），四卡 `allgather_sum`
按 rank 序求和即得完整专家输出。**关键观察：专家并行化后这个 fold 契约一个字都不用改。**

## 2. 架构方案

### 2.1 分片映射（静态，无协调）

```
owner(e) = e % world          # 288 个路由专家 → 每卡 72 个
共享专家（index 288）不走 marlin：保留现有 affine 列切分路径不变
```

- 每卡 marlin 块数 289 → **72**，专家显存 1.85 → ~0.46GB/层（~21GB 总），加上 affine 共享
  专家（~2.3GB）与 dense/KV → **~27GB/卡，稳过**（交接 §四.1 口径）；
- 静态 round-robin 保证块数均衡；若后续窗口实测路由热度过偏，备选=按路由直方图均衡分桶
  （先测再改，不预先优化）；
- 共享专家留 affine 的理由：①它被**所有 token** 命中，进 marlin 单块会让 owner 卡成为每层
  热点；②affine 共享权重本来就已在显存（`shared_parts`），零新增加载；③它继续以列切分
  partial 参与同一个 fold。

### 2.2 路由改造：`FixedRoute` → `EpRoute`（已实现）

- `EpRoute.build(pick16, n_slots)`：输入全局 top-8 选择，**只保留 owner==self.rank 的对**，
  映射到本地块号（`local_block = lut[e]`，72 项查找表）；
- 位精确性质与 `FixedRoute` 同构且有读码背书：块数 72 为常数 → decode.cu 的条带分解
  （`tiles = blocks * N_TILES`）逐调用不变 → 每对 fp32 累加序只依赖（块，行序）；
- **非本地对不进路由数组**（不是 e0<0 填充）：核内不出现非本地专家；
- prefill 走 `MarlinRoute(lut=…)` 动态打包（prefill 的 drafted/serial 输入恒同，路由确定即可），
  eids 同样持有本地块号；
- **残值不变式（实现关键，U1 之外的真机红线）**：非本地槽位的 `mc3/ey` 持有早前调用的有限
  残值 → EP 分支把对应槽位的 combine 权重清零（`0 × finite = 0`，槽序不变，图捕获安全）。
  缓冲区构造时为零、内核只写有限值 → 残值恒有限，不变式成立。

### 2.3 求和语义与位精确论证（重构红线 #4）

数学：`moe(x) = Σ_e w_e·expert_e(x)`，秩不变和。EP 后每卡 partial 变更为
`partial_r = Σ_{e: owner(e)=r} w_e·expert_e(x)`（卡内按专家号升序累加），
四卡 partial 仍经 `allgather_sum`（fp8 或 bf16）按 rank 0→3 求和：

1. **fold 契约不变**：同一内核、同一 rank 序、同一次发布-到达协议；通信层对 EP 无感知——
   EP 只改变 partial 的**含义**（k 切片 → 专家子集），不改变其**机制**；
2. **确定性**：路由 top-8 逐元素确定 → 每对恰好被 owner 算一次 → partial 逐元素确定；
3. **自洽契约保持**：drafted 与 serial 走同一 EpRoute、同一分片 → `drafted == serial` 硬门槛
   不变；
4. **全局谱系断点（有计划、要记录）**：EP 改变了跨 rank 累加的划分方式，全局 SHA 必然离开
   `0436adfa` 谱系。这是预期内的架构变更，不是回归——组合窗口以 `--exactness` 重新定基
   `ep-<new sha>`，收据注明断点原因与旧谱系链接。与 fp8 窗口同理（逐元素确定性 → 自洽）。

### 2.4 加载与转换

- `convert_experts_prod_ep(ckpt_dir, layer, rank, world, shared_parts)`：只转本卡 72 个专家，
  P1 生产换源通路不变（unpack→repack 零 requant 误差，drafter 兼容已证）；
  转换时间 145s → **~36-40s**（4 卡本就并行）；
- `shared_parts` 的共享专家保留 affine 常驻，**不再** repack 进块 288；
- `_prod_shard` 的 index.json 缓存在此路径上自然生效（每卡读自己的 72×6 个 key）。

### 2.5 开关与回退

`TF_EP_EXPERTS=1` 启用（默认关）。关闭时路径与快照完全一致。回退 = 去掉开关。

## 3. 测试计划（对应 TEST-MATRIX T1/T5）

| 关卡 | 内容 | 通过标准 |
|---|---|---|
| U1 单进程协议模拟 | 2/4 假 rank、288 专家随机路由：EpRoute 局部化 + partial 求和 vs 单卡全量参照 | 逐元素相对误差 = 0（同序同精度）或可归因的 fp8 量化差（bf16 通路必须为 0） |
| U2 路由不变式 | 随机路由下"对→tile 索引与他行无关"（复用 `marlin_rindep_test.py` 方法） | R=1 与 R=7 同 token bit-exact；负例（dynamic route）不匹配 |
| U3 真机 4-rank | EP 转换 45 层 + 冒烟 + 位精确 | 显存 ≤ ~30GB/卡；drafted==serial PASS |
| W1 组合窗口 | fp8 + d7 + marlin P1 + EP 全套 env | structured ≥ 290（数学投影 300.06）；exactness 新谱系定基 |

## 4. 风险登记

| 风险 | 概率 | 缓解 |
|---|---|---|
| ~~`.so` 块数硬编码/需要 pad 对~~ | — | **关闭（D1 读码）**：ntpp 动态、eids=块索引、e0<0 早退俱在，整专家实例化并存 |
| 路由热度偏斜（round-robin 计算不均） | 低 | 块数（显存）静态均衡不受影响；单 token 计算偏斜先测直方图再决定是否分桶 |
| fp8 × EP 叠加的自洽 | 低 | U1/W1 分别覆盖；fp8 通路的问题与 EP 正交（fp8 窗已单独证毕） |
| exactness 谱系断点被误判回归 | — | 本文 §2.3 已定性；收据模板注明 |
| affine 残值混入 partial | 已消 | combine 权重掩码（§2.2 残值不变式）；U2 真机复核 |

## 5. 与重构红线的关系

- 思想来源：专家并行本身是 MoE 推理的公共架构（vLLM/Morrowmake 生产即按专家切分）；本设计的
  EpRoute 局部化、affine 共享专家保留、fold 不变式论证、残值掩码为本项目自有表达；
- 改动文件（本分支）：`marlin_experts.py`（ep_local_experts/ep_lut/convert_experts_prod_ep/
  EpRoute/MarlinRoute lut 参数）、`forward.py`（Buffers EP 分支 + _marlin_block EP 路径）、
  `weights.py`（expert_names prod-marlin 跳过 + moe() EP 分支）+ `tests/test_ep_route.py`
  （U1：分区/同一性/代数/lut-prefill/所有权，CPU 全过）；
- 入库前过 `tools/overlap_check.py`（Python 逻辑文件同样送检）。

## 6. 窗口 runbook（U2/U3/W1，需停生产，用户已原则批准）

```bash
# 0) 预编译无（纯 Python 改动 + 既有 .so）；同步分支到 hym74（tar 或 git bundle，勿动原树）
# 1) 停生产：cd /home/hym/glm53-recipe && ./stop.sh；nvidia-smi 确认 19MiB
# 2) U2 路由不变式（真机，GPU）：marlin_rindep_test.py 的 EP 版 —— R=1 vs R=7 同 token bit-exact
# 3) U3 加载+冒烟：TF_MARLIN_EXPERTS=1 TF_MARLIN_SOURCE=prod TF_MARLIN_CKPT=<...> \
#    TF_MARLIN_SO=~/tensorfold-poc/marlin_standalone_p2/_ampere_marlin_C.abi3.so \
#    TF_EP_EXPERTS=1 bash tf4_serve066_mlx.sh dflash
#    看 [prod-mem-ep] 行：mx≈0.9GB/层、alloc 终值 ≈46-48GB（若 ~55GB+ → affine 双载未除净，停下排查）
# 4) W1 组合窗：加 TF_SHM_AG=1 TF_BF16_GATHER=1 TF_FP8_GATHER=1 TF_F_DEPTH=7 →
#    tf4_ab_bench.py --profile dflash --tag combo-ep --exactness
#    验收：structured ≥ 290（数学投影 300.06）；exactness 新谱系定基 ep-<sha>（断点=EP 累加划分）
# 5) 停 TF → 恢复生产 → bench_decode_mia --structured 复验 288-289
```

