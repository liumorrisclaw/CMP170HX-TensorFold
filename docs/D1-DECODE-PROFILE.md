# D1 decode 画像定论 + D2 首轮（2026-10-12）

## D1：verify 图内核表（R=6/R=7，CUPTI 归因口径，绝对值膨胀 ~10×，份额可信）

| 内核 | 份额 |
|---|---:|
| **gather_sum_fp8（90 次/步）** | **93.26%** |
| expert_kernel 两型 | 2.16% + 0.95% |
| qmm_kernel | 1.24% |
| chain/hc/router/… 其余全部 | <2.4% |

**推论**：
1. decode 每步同样是通信骨架主导（45 层 × 2 gather）；
2. **D3 AX300-MOE 被数据降权**（专家段 2.2%，砍半也只值 ~1%）；
3. 300 路径 = **深位接受率（D2）+ 骨架/偏斜削减**（R=1 时单次 gather 传输仅 ~15µs，墙钟大头是 90 次屏障事件的偏斜/发射 + 主机步间空隙——即 C5 遗留 ~4-8ms）。

## D2 首轮（负结果入册）

- 发现 `m_policy = DepthPolicy(3, fixed, conf=0.35)` **硬编码**——MTP 臂从未获准深于 3；
- env 门控化（TF_M_DEPTH/TF_M_CONF，默认不变）+ m7 A/B：**逐位零变化**（241.4/accept 0.802/tok-step 3.32/0436adfa）；
- 归因：贪心解码 DrafterChoice `first="f"` 起 f 臂，m 臂深度策略不上场 → **MTP 臂非贪心路径杠杆**；
- f 臂深位（d7 已默认）接受率坍缩是墙：tok/step 3.3 @ d4 与 d7 相同 → P(accept@5+)≈0。

## 下一步（按数据优先级）

1. **步相位分解仪表**（TF_STEP_PROF）：100 步内打 replay/drafter/commit/host-gap 各自墙钟 → 定位 C5 的 4-8ms 主机空隙（v1.0.5 Mac 线的"round-ahead commit"思想 = CUDA 侧同款机会）；
2. 骨架削减候选：主机空隙流水化（提交/采样与 GPU 重叠）、gather 事件合并可行性论证；
3. D2 残余：DFlash 链 EDGE/NOISE 常数微调（低预期）；D3 冻结待 1/2 出清。

## 步账本 v2（2026-10-12 定稿，stages_ms + post 拆分，生产实测）

结构化 400 tok / 55 轮（sha 279e2fb，241 tok/s）：

| 相位 | ms/轮 | 说明 |
|---|---:|---|
| forward | **24.5** | verify 图（93% = 90 次 gather：~0.12ms 传输 + ~0.15ms 偏斜/发射） |
| draft | **4.62** | **drafter 本体**（DFlash block 3.37 + 链 ~1.2） |
| post | **0.13** | backlog/taps 拷贝 + follow + on_tokens（round-ahead-commit 的猎物**不存在**） |
| sample+commit | 0.73 | 噪声 |
| 合计 | ~30.0 | 7.3 tok/轮 = 241 tok/s，账目 99% 闭合 |

**C5 关闭**：无未归因主机空隙。**D2 关闭**：7.3 tok/轮 @ 0.983 逐 token 接受（345/351）已是 Morrowmake 级。

## 300 的两个残余项目（数据定价）

1. **drafter 通信优化**：4.62ms 里 drafter 自身也是 TP4-per-layer-gather 设计（小模型多层 × gather 骨架）→ 目标 2-3ms；与对家（draft≈0，forward 25.5 全包）对齐后 7.3/25.5 ≈ 286；
2. **verify 骨架偏斜削减**：24.5 → ~22（90 事件 × 偏斜 0.15ms 的削减——流内锁步/发射流水）；
两者叠加 ≈ 7.3/24 ≈ **304**。

## drafter 通信优化项目结论（2026-10-12，负结果入册）

- 实现：`_row` 的 copies-gather+torch 三连加 → `all_gather_sum_fp8` 单内核（TF_DRAFT_FP8，10 次/块，字节减半、launch 减 3/4；drafter 数值自由变化不影响输出谱系——草稿只提议）；
- A/B：structured **243.7（+2.7）**、coding **175.3（−6.0）**、prose **114.7（−2.0）**；draft 臂 4.62→**3.90ms**（仅省 0.72ms）；accept 0.802→0.814；shas/0436adfa/exactness 全保；
- **判定：净负，回滚**（TF_DRAFT_FP8 默认关，代码留树）。**drafter 是计算/环形注意力约束非通信约束**（gathers 仅占其成本 ~16%，块体 3.37ms 主导）；
- 剩余 300 路径更新：① verify 骨架偏斜削减（24.5→~22，90 事件 × 0.15ms）；② drafter 环形/块计算优化（DRAFT_RING 尺寸 × 接受率权衡——高风险高回报）；两者之外 TP4 约束集已穷尽软杠杆。

## verify 骨架削减项目结论（2026-10-12，地板定论）

| 探针/手段 | 结果 |
|---|---|
| 锁频 1500MHz + persistence | **零变化**（241.2）——时钟非源 |
| two-shot@decode 尺度（v12 模式 2） | **−17%**（199.4）——双屏障 × 90 事件 > 字节节省；与 P2-prefill 负结果在 decode 端对称闭合 |
| （前置）P2/偏斜账 | one-shot 0.27ms/gather @R8 = 196KB 传输物理 + 协议地板 |

**定论：one-shot fp8 的 90 事件骨架是 TP4 该架构的实用下限**（三探针闭合）。decode 剩余未试项均为接受率轮盘类（m 臂混合策略、DRAFT_RING 经济学），期望值不确定。

## 会话级 300 总账（截至本节）

- 生产：**241.6/181.4/116.9** + prefill **1,551-1,587**（自研栈，0436adfa，exactness PASS）；
- 已关闭：C5 主机空隙、D2 深度、D3 专家、drafter-comm、时钟、two-shot（两端）、P4 重叠（3% 天花板）；
- 剩余正路：**PP4-prefill**（prefill 2,000-3,000，待立项）；decode 侧 TP4 软杠杆穷尽，241 即当前约束集实测最优（对家 TP4 同骨架 288 的差额 = 其 drafter≈0，我们的 4.6ms 臂对应到 ~265 的理论上限仅在臂被砍且内容轮盘友好时成立）。

## m 臂混合实验（2026-10-12，轮盘判负）

- TF_FIRST_ARM=m + TF_M_DEPTH=5（first 臂已 env 门控，默认不变）；
- **0436adfa 保持 + exactness PASS**（换臂不触输出谱系——预测兑现）；
- 三数轮盘：structured 236.7（−4.9）/ coding 163.5（**−17.3**）/ prose 125.0（**+8.4**）——净负（525.2 vs 537.9）；
- 判定：回滚。m 臂对 prose 有真实增益（+7%）但 coding 代价过大；**decode 实验矩阵至此全部跑完**。

## decode 300 最终结案陈词

TP4 + 本架构的实测全局最优 = **240.5/180.8/116.6**（0436adfa）。每一条软杠杆均有收据定价。
超越路径只剩架构级：PP4（prefill/decode 重排）或 lane rounds 聚合口径——均为待立项项目。
