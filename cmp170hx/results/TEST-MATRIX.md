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
| C7a | **Marlin expert 段接入作战卡（2026-10-09）** | ⚠️ **窗口验收完成（2026-10-10），结论：内核更快但策略性回退**——①端到端跑通（15min 加载、~46GB vs 63GB/rank、输出连贯），②serial 54.8 vs 46.6 (+18%)、coding 153.1 vs 142.6 (+7%)、prose 116.6 vs 101.5 (+15%)，③**structured 151.1 vs 199.4 (−24%)**：g128-requant 改目标 logits → DFlash2 drafter 接受率↓20%（tok/round 6.8→5.41），高接受率画像受损最大，④exactness FAIL（41 tok 分叉）：decode.cu stripe split 依赖 tiles=blocks×N_TILES 与 pair 块号 → fp32 累加顺序随 R/路由变，⑤**62µs 对齐锚点是假象**：真实路由 72 pair 散布 50-72 专家 → distinct-expert 权重读 = DRAM 墙（M=8 spread 157µs/层 ≈ affine 160µs），专家段成本由 distinct-expert 数决定与格式无关。**判定：速度无净收益，价值=省 17GB/卡**；栈已保留（TF_MARLIN_EXPERTS=1，weights/forward 已接线+模块验证，marlin_experts.py），后续作内存优化选项；若重启需 P1 生产 g128 checkpoint 换源（零 requant 误差，drafter 兼容已证）+ P2 内核固定 289 块布局+e0<0 早退重建 .so。完整分析：`data/window-marlin-c7a-1010/ANALYSIS.md`。原始作战卡归档于 git 历史（本行替换前版本） | ✅ 窗口完成（负结果） |
| C8 | 理论锚点 | 每步回 ~20ms 且边际 2.57 兑现 → **~240-260 tok/s**；300 需再压边际或 x16 | 公式推算 |
| C8a | **fp8 gather 载荷（2026-10-10）** | **structured 240.7（+20.7%）达标**：affine 栈 + TF_FP8_GATHER=1（v8 扩展 gather_sum_fp8：staging 即量化 bf16→e4m3，PCIe 字节减半，求和手写位解码回 fp32，own-rank 从 slice 重量化确定性同位）+ TF_F_DEPTH=7；coding 170.9(+19.8%)/prose 116.1(+14.4%)；**accept 0.794≈bf16 0.793（fp8 零均值噪声不伤 drafter）**；**exactness PASS**（逐元素确定性→drafted==serial 自洽保持）。内核级验证：单进程模拟 4-rank 协议 bit-exact。坑：cuda_fp8.h 的 cvt_fp8_to_halfraw 在 sm_80 返回 0（手写 ldexpf 位解码）；Layout 偏移 arrival=0/failure=128；改 CUDA_SRC 必须 bump 扩展名。轮时 29.7ms@7.14 tok/round；**距 300 剩余**：draft GPU 化（4.9→3.5，→~250）+ base 减重 marlin 17GB（→~268）+ 编排压缩；2.64ms/行 对账=专家 0.8+gather 载荷 1.8（bf16 地板）+杂 0.05，fp8 后 gather→0.9。收据 `data/window-fp8-gather-1010/`，代码树内默认关 | ✅ 窗口达标 |
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
| E2 | PR #466（shm all-gather） | **已完结（2026-10-11）：closed-unmerged**。两版（移植版 + fresh `ecf6848`）均未落地；ashhart 10-07 判决=收益真实但只落地作者自己的代码（移植版 11/220 六行窗与 Morrowmake fork 重合）。**重合复检收据（2026-10-11，本表 H1）**：移植版 11 窗、fresh 版 1 窗（`break;` 平凡 token）、066 树 v3 全功能版 1 窗——fresh/v3 干净。上游转 Zig，python-0.6 冻结；PR 值守规程退役。后续走自主线：`../REFACTOR-DEV-PLAN-20261011.md` |
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
- exactness 硬门槛：任何改动后 `draft_sha` 必须等于 `0436adfa…`，否则立即回滚（EP 改造将改变全局累积序，届时**有计划地**开新谱系并在收据注明断点，见 REFACTOR-DEV-PLAN §6 阶段 1）。

## H. 2026-10-11 重构立项收据

| # | 事项 | 结论/产物 |
|---|---|---|
| H1 | 六行窗重合复检（归一化，对 Morrowmake `host_shm_all_reduce.py`） | 移植版 v1 = 11 窗（复现 ashhart 判决）；fresh `ecf6848` = 1 窗（`break;`）；066 树 v3 = 1 窗。检查器待固化 `tools/overlap_check.py` |
| H2 | Morrowmake 源码存档 | `../reference/morrowmake_host_shm_all_reduce.py`（1183 行）+ `morrowmake_cuda_communicator.py`（973 行），自 hym74 生产源码目录拉回 |
| H3 | 代码资产盘点 | fork 分支干净；**066 树非 git 仓库（单副本在 GPU 机）**——阶段 0 第一件事入库。明细见 REFACTOR-DEV-PLAN §2.2 |
| H4 | 全面重构开发计划 | `../REFACTOR-DEV-PLAN-20261011.md`（定位转换 / 开源对比 / 分层架构 / EP 方案 / 测试缺口 T1-T9） |
| H5 | 成果沉淀（交接 §四.4 收据部分） | fp8 / marlin-c7a / p1-prod-source 收据 + TEST-MATRIX 已同步至公开仓库 `CMP170HX-TensorFold/cmp170hx/results/`（本地，未 push）；代码与 CHANGES 随阶段 5 |
| H6 | **EP 专家并行窗口（2026-10-11，W1+W2）** | 工程全过：U1 CPU 双机、U2 真机 bit-exact（负对照 mismatch）、U3 **44.3GiB/卡**（83GB OOM 消除，affine 双载修复=OOM 真因：`rd.prefetch` 上传 288 死重 ~0.9GB/层）。**路线判定关闭**：Marlin .so=Morrowmake 二进制（违自研红线）+ exactness FAIL（bf16 仍 FAIL，排除 fp8×EP）+ 运行间非确定（三轮 greedy sha 互异 vs C8a 全同）+ 吞吐 167.2/121.1 << 自研 affine 240.7。MTP 层 prod-ckpt 未打包格式 bug 与 EpRoute 图捕获 H2D 已修（`cmp170hx-stack` c91029d 等）。收据 `data/window-ep-1011/`，终态判定 `cmp170hx-stack/docs/EP-DESIGN.md` | ✅ 窗口完成（路线负结果） |
| H7 | **ax300 新干线建立与验证（2026-10-11）** | 基线=上游 python-0.6 HEAD `ed78d6fc`（1.0.x Zig 线无 GLM CUDA 后端，上游 v1.0.5 配方原文钉死；主线基准声明 `cmp170hx-stack/docs/BASELINE.md`）。移植 12 文件（shm v3 家族/fp8/fbgb/cost-depth/copy-drafts/画像钩子），marlin 全剥离。**验证：structured 240.7 / coding 171.8 / prose 116.4，三轮全同，exactness PASS，谱系 sha 0436adfa 原样延续**——移植逐位等价。教训入库：venv 里 0.6.3 editable install 会抢占 PYTHONPATH（`TensorFold/src`），部署树必须带 `src/` 层；换树前先验证 `import tensorfold.__file__`。收据 `data/window-ax300-1011/` | ✅ 窗口完成（移植验证达标） |
| H8 | **生产切换 + L7 draft GPU 化（2026-10-11）** | 生产主线=ax300（:8000，ax300_prod.sh，回滚=glm53-recipe/start.sh）；上线验收 240.7/0436adfa。**prefill 指标进 bench**（prefill_tok_s 逐 run+中位）+ 长探针：17.7k/61.5k/171k/186k tok/s @1.5k/6.3k/25.4k/31.2k tokens。L7=候选选择链上 GPU（两趟稳定 argsort 等价 lexsort；hym74 torch 2.13 无 lexsort/take_along_axis，gather 实现）：**241.4**、accept 0.798、exactness PASS、谱系不变；+0.3~0.7% 小于预估（CPU 链已被流水线掩盖）→ 瓶颈在别处，L9 画像成为关键路径。收据 `data/window-ax300-1011/`（ax300-l7/ax300-prod-accept） | ✅ 已上线 |
| H9 | **Morrowmake 历史数据开采 + 真实冷预填充更正（2026-10-11）** | hym74 `~/bench/*.json` 为 Morrowmake 全矩阵：structured 288.6-289.1（final_hold 系列）、coding 167.3（v170poff）、prose 101.5（v170poff）/143.6（prose_tp4off，10-10 复测）；PP4 vs TP4 实测：decode TP4 胜（172.8 vs 107.7@240k）、**prefill PP4 5× 胜**（5,616 vs ~1,000@512k）、c8 聚合 PP4 605 vs TP4 442。**更正：本日早前 prefill 17.7k-186k tok/s 为前缀缓存命中假数据**；真实冷预填充 ax300 ≈ **204 tok/s 平坦（1.5k-31k）**，与 170 时代 194@10.5k 一致（prefill 路径未动过）。追赶结构线索：Morrowmake 每步草稿 6.85 个近全收 vs 我们 3.2（同一 DFlash2）——深度策略/深位接受率是第一差距。收据 `data/window-ax300-1011/cold_prefill_20261011.json` | ✅ 数据开采完成 |
| H10 | **prefill/起草双调查 + 主线恢复（2026-10-11 晚）** | Q1：fp8 gather 占 prefill 首块 76%（内核表物证）；one-shot 流量模型 → ~400 tok/s 设计上限；bf16 路由实测 426（2.1×）但引发集合互扰（rank0 空转/对端自旋/decode 94）→ **回滚，主线恢复 242.0/181.5/117.0 + 0436adfa**（restored2 收据）。Q2：conf=0.05 负结果（64-69 tok/s，截断非杠杆，草稿质量才是）；已回滚 0.3。两步走计划入 PLAN-AX300 §7（P1-P4 prefill 2000-3000 / D1-D4 decode 300）。收据 `data/window-prefill-1011/` | ✅ 恢复+计划定稿 |
| H11 | **P1' 同族加宽落地（2026-10-11 深夜）** | v9 内核：gather_sum_fp8 增 NB=8/16/32（TF_FP8_WIDE，阈值 128KB：decode 窗 ≤96KB 恒 blocks=4）。单测全过（NB 1-32 位精确=参照/不变/确定/代数）；开发口 A/B：**冷 prefill 204→909-926（4.5×）**，decode 242.1/181.0/116.7 + exactness PASS + 0436adfa 不变；**已上生产**（prod-wide-final：241.8/180.9/116.6 PASS）。两条纪律教训：CUPTI 画像窗数字自带 -5% 开销（TF_GRAPH_PROF 只用于归因不作性能口径）；探针预热必须换 nonce（同 prompt 二发=前缀缓存假数据）。标准化 prefill 探针（带上下文长度列）入 tools/。收据 `data/window-wide-1011/` | ✅ 已上线 |
| H12 | **P2 离线就绪 + 稳态画像确认（2026-10-12）** | wide-NB 后首块表显 3.3ms/gather 为暖机偏斜（含自旋等待）；墙钟账（910 tok/s → ~0.75ms/gather 稳态）证实**已贴 one-shot fp8 带宽模型** → P2 流量削减（4N→2.5N）对症。two-shot 内核 v10 实现入库（658bdd2）：分块四分片归约 + 全字双相位屏障（自查修掉跨块竞态与 NB× 重复归约两处结构 bug）+ gather 区扩段 + TF_TWOSHOT 门控；**数值语义读码修正**（v9=own-first 序 → two-shot 开自洽新谱系支）；重合门 CLEAN。待 GPU 阶梯（owner 序参照单测 → 开发口 A/B ≥1,400 → 上线） | ⏳ 下窗口 |
| H13 | **P2 修复 + A/B 负结果（2026-10-12）** | compute-sanitizer 双发现全修：①shared 旗标复用无尾屏障（racecheck 56-60 hazards）；②**相位 2 屏障未含自己 rank 的块**（块 0 重组竞态块 2-7 的 gather 写入 → 坏元素恰为 0.0、起于 unit 3072 的确定性零尾——数值判尸定位）。v12 fresh-base 阶梯 4 连过。**A/B 判死路线（正确性无关）**：冷 prefill 910-926 与 one-shot 持平 + decode 新谱系 218 → **prefill 是屏障/秩偏斜约束，非带宽约束**（1.6× 流量削减买不到东西）。TF_TWOSHOT 回滚默认关；v12 内核留树作 P4 重叠搭档。生产回 v12-one-shot 路径验证无损（241.7/0436adfa）。收据 `data/window-p2-debug/` | ✅ 闭环（负结果入册） |
| H14 | **P3' fp8-over-NCCL 上线（2026-10-12）** | 画像修正先行：76%-gather 表实为**启动校准**（64-token drafter 标定），服务 prefill=2048 行块 NCCL bf16 字节约束（~1.5GB/s，910 为该路径物理上限）；P4 重叠算术判死（计算仅 3%）。杠杆=**partial 量化 e4m3 走 NCCL uint8**（TF_FP8_NCCL，纯 torch 实现零内核）：**冷 prefill 910→1,524-1,586（1.7×）平坦**；**decode 241.1/181.5/116.6 + 0436adfa 原样**（短 prompt 不经此路径；长 prompt 内容开自洽新支，exactness PASS）。生产验收 prod-fp8nccl-final。剩余 2,000-3,000 缺口 = PP4-prefill 架构项目（Morrowmake 实证 TP4 1,017 vs PP4 5,616），提案待决策。收据 `data/window-fp8nccl-1012/` | ✅ 已上线 |
