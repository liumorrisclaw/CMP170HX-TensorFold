# AX300 计划 — 第一性原理分解（2026-10-11）

> 基线：`ax300/trunk` = 上游 python-0.6 HEAD `ed78d6fc`（含我们署名的 #464/#465、官方 Ampere
> 修复、#468 copy-drafts 参照）+ 我们全部已验证成果的移植（commit `523ebff`）。
> 红线：全部自研（零外部二进制）；位精确硬门槛（drafted==serial）；每个 CUDA 文件过重合门。
> 生产：暂不恢复（用户指令），GPU 全部让给开发。

## 一、目标函数与分解

```
tok/s = tok/step ÷ t_step
tok/step = Σ_{k=0..d-1} P(accept≥k)     ← drafter 质量 × 深度策略
t_step   = verify(d行) + draft + 编排    ← d7 实测 29.7ms（240.7 窗口）
verify 每行边际 ≈ 2.64ms = 专家 0.8 + gather(fp8) 0.9 + 杂 0.94
```

300 需 t_step ≈ 23.8ms @ 7.14 tok/step，即从 29.7 再削 5.9ms。

## 二、杠杆清单（全部有实测依据；✅=已在新干线，🆕=本日立项，⏳=待做）

| # | 杠杆 | 机制 | 量值 | 状态 |
|---|---|---|---|---|
| L1 | shm one-shot all-gather（fresh 协议） | 替代 NCCL ring | +21.5%（历史 A1） | ✅ |
| L2 | bf16 gather 载荷 | PCIe 字节减半 | +35.8%（A2） | ✅ |
| L3 | fp8 gather 载荷（发布侧量化） | 再减半，drafter 无损 | +20.7%（C8a → 240.7） | ✅ |
| L4 | fbgb 块对角融合 | 单 GEMM | +0.2%（A3） | ✅ |
| L5 | 成本感知深度 | argmax_d 效用 | 保持最优+抗漂移（A4） | ✅ |
| L6 | copy-drafts（GLM 版） | 重复文本加速 | 结构性收益 | ✅ |
| L7 | draft GPU 选择链（torch 两趟稳定 argsort，无损） | 实测 +0.3~0.7%（CPU 链已被流水线部分掩盖；accept 0.794→0.798 微升） | 241.4 @ 生产 | ✅ 已上线（保留，预估 +5% 未兑现——瓶颈不在候选链） |
| L8 | **自研专家内核**（本日立项） | 专家段 0.8→0.5-0.6ms/行；后续解锁 fp8 专家权重 | −2~3ms/round → **+15~25** | 🆕 见 §四 |
| L9 | 编排/commit/采样压缩 | C5 未归因 ~8ms（草稿步） | 需画像归因后定量 | ⏳ 钩子已就位（TF_DECODE_PROF，图捕获后触发条件待修） |
| L10 | block_size=16 / 更深草稿 | 架构解锁 | 大项 | 后置 |

**路线**：240.7（新干线复现验证）→ +L7 ≈ 252 → +L8 ≈ 265-275 → +L9（按画像定量）→ 300。
L9 的 8ms 是最大未知数：若归因出 3-4ms 可压缩，300 可达；若归因后无空间，则需要 L10
或接受 ~275 为当前约束集的实测上限——**每个里程碑都按实测校准，不做纸面推演**。

## 三、已关闭路线（不回头）

- Morrowmake Marlin .so：别人二进制 + exactness FAIL + 运行间非确定 + 吞吐 167（EP 窗口
  W1/W2 三重判死，收据 `data/window-ep-1011/`）——**包括"借来用用"也不行**；
- EXL3 单流五维闭环（170.1 定格）、MLX 权重、Q4 dense、策略扫描、分块流水、hc_mix 图内折叠
  （TEST-MATRIX §B 全表）。

## 四、自研专家内核（AX300-MOE，本日立项）

**为什么**：专家段是 verify 边际里最大的单项（0.8ms/行），且现在用的是上游通用 grouped
内核——只有自己写才能 ①把 affine g128 布局的读带宽压到理论值，②直接吃 fp8/自研通信的
量化载荷（外部内核给不了的协同），③完全的位精确自主权。

**方法论（与通信内核同一套，已验证有效）**：
1. 规范先行：`docs/AX300-MOE-SPEC.md` —— 布局、路由语义、位精确不变式（每输出元素单 warp
   固定 k 序 fp32 累加、drafted==serial 的 R 无关性）、与 gather/fold 的接口；
2. 单进程协议模拟位精确（torch 参照，CPU/GPU 各一）→ 真机内核位精确 → 微基准 vs 上游
   grouped 内核（同速或更快才接线）→ 窗口；
3. 每个源文件过 `tools/overlap_check.py`，零重合入库；concept credit 只写思想不写代码。

**设计要点（第一性，非参照任何实现）**：
- 输入：affine g128（word/scale/bias，rank 行切分）——与现栈权重零转换；
- 两个 GEMM 段（gate/up→SwiGLU→down）+ 路由聚合；decode R≤8：每对输出 = 单 warp 按固定
  k 序累加（位精确不变式）；
- 权重流是 DRAM 墙（C7a 实证 160µs/层级别）→ 主攻 cp.async 双缓冲深度 × tile 形状扫描；
- 与自研 shm 层的协同：expert 段输出直接以 fp8 载荷发布（省一次 bf16 round-trip）。

**里程碑**：M1 规范+位精确 harness（今日启动）→ M2 内核+单测（2-3 天）→ M3 微基准
（对齐或超过上游 grouped 160µs/层）→ M4 接线窗口（structured 增量验收）。

## 五、今日已完成 / 接下来 48 小时

- ✅ 新干线建立（`ax300/trunk` @ ed78d6f + 移植 `523ebff`），hym74 导入冒烟通过；
- ✅ EP 终态判定与收据（H6）；旧 066 树快照归档（tag `hym74-snapshot-20261011`）；
- ✅ **新干线复现验证达标（2026-10-11）：structured 240.7 / coding 171.8 / prose 116.4，
  三轮全同，exactness PASS，谱系 `0436adfa` 原样延续**——移植逐位等价，240.7 基线在新干线
  成立。收据 `data/window-ax300-1011/`。中途插曲：首轮 176 是测错树（venv 0.6.3 editable
  抢占 PYTHONPATH），教训入 TEST-MATRIX H7；
- ✅ 主线一致性：BASELINE.md（v1.0.5 = 上游最新 release 基准；引擎源码 python-0.6 HEAD，
  Zig 线无 GLM CUDA 后端——上游官方文档原文）；
- ⏳ L7 draft GPU 化（小项，先做）；L9 画像（修 TF_DECODE_PROF 触发条件）；
- ⏳ AX300-MOE M1 规范 + harness。

## 七、两步走总纲（2026-10-11 晚定稿，用户拍板）

> **第一步：保住 decode 240.7，把冷 prefill 从 ~204 提到 2000-3000 tok/s。**
> **第二步：保住 prefill 成果，把 decode 从 240 提到 300。**

### 第一步：Prefill 攻坚（当前 ~204，目标 2000-3000）

今日实测与物理账（收据 `data/window-ax300-1011/`）：
- fp8 one-shot gather 占 prefill 首块 GPU 时间 **76%**（295.5ms/389ms，单次 ~3.3ms）；
- one-shot 流量 = 每层 ~2 次 gather × 4×R×D×2 字节/秩 ≈ **2.75ms/token 通信地板 → ~400 tok/s 设计上限**（Gen2 x4）；
- bf16 路由实测 426（2.1×）但因集合互扰回滚（rank0 空转、对端自旋、decode 掉到 94）；
- 参照：Morrowmake TP4（ring allreduce+重叠）1,017；PP4（无逐层通信）5,616。

| # | 手段 | 预期 | 前置 |
|---|---|---|---|
| P1 | ~~修复集合互扰~~ **被 P1' 绕开并超越**：同族加宽（v9 NB 8/16/32）不混内核、位精确由构造保证——已上线，prefill 204→**909-926（4.5×）**，decode 242.1/0436adfa 不变 | ✅ 已上线（2026-10-11） |
| P2 | **two-shot 预填充集合**（reduce-scatter+all-gather，shm 上自研）：流量 4×→1.5× | ~1,100 | P1；概念署名 Morrowmake two-shot，内核全新自写+过重合门 |
| P3' | **fp8-over-NCCL（已上线 2026-10-12）**：prefill 大载荷（>1MiB）量化 e4m3 走 NCCL uint8，字节减半 → **冷 prefill 910→1,526-1,588（1.7×）**；decode 239.8/0436adfa 谱系原样（短 prompt 不经此路径）；长 prompt 内容谱系开自洽新支 | ✅ 已上线 |
| P4 | ~~通信/计算重叠~~ **算术判死**：prefill 计算占比仅 ~3%，重叠上限 3%。且 76% 画像实为**启动校准**（64-token drafter 标定）非服务 prefill（已修正记录）。剩余 2,000-3,000 缺口 = **PP4-prefill 架构项目**（Morrowmake 实证：TP4 物理 ~1,017，PP4 5,616）——待用户决策 | ⛔ 判死/转 PP 提案 |

验收线（每步）：decode 240.7±2 与谱系 `0436adfa` 不动 + 冷 prefill 探针（唯一 prompt）达标 + 收据归档。

### 第二步：decode 240.7 → 300

| # | 手段 | 依据 |
|---|---|---|
| D1 | L9 画像：修 TF_DECODE_PROF 对图回放的触发，量化每步 ~4ms 编排 + 深度经济学 | C5 遗留 ~8ms 未归因 |
| D2 | 草稿深度/质量：conf 截断已证非杠杆（0.05→64-69 负结果）；主攻 MTP 臂链接质量与 DFlash EDGE/NOISE 常数、分画像深度策略 | Morrowmake 6.85 vs 我方 3.3 draft/step（同一 drafter） |
| D3 | AX300-MOE 自研专家内核：verify 边际 0.8→0.5ms/行，深草稿变划算 | M1 规范+harness 已立项 |
| D4 | 架构项（后置）：block_size、lane rounds | 上游冻结，自研 |

## 六、纪律（不变）

窗口四步、生产恢复复验（本轮暂缓，用户指令）、exactness 破坏即回退、收据即时归档、
重合门零重合入库。
