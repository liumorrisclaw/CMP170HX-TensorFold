# PR #466 fresh 重写收据 — 2026-10-10 — 上游邀约已响应，clean-room 内核落分支

**上游 review 结论（ashhart，10-07 08:37）**：S1 shm all-gather "+21.5% 是 PCIe-only 节点的真实收益，
receipt 干净"；但 CUDA 源与 Morrowmake vLLM fork 逐行重合（220 个六行窗中 11 个：flag acquire/release、
block barrier、copy loops），按"只落地作者自己的代码"的规矩不能照原样落地——**明确邀约：
"If you'd like to write the kernel fresh, we'll gladly take it."**

## fresh 重写（本日完成，commit `ecf6848` → fork `shm-allgather` → PR #466 更新）

- 到达协议全新表达：generation-tagged arrival words（[stage][block][rank] 打包数组，无 128B Flag 结构），
  thread 0 经 `cuda::atomic_ref<uint32_t, thread_scope_system>` release store 发布（前有 `__threadfence_system()`），
  每 peer 一线程并行 acquire 轮询（PCIe 检测保持单次往返），64-bit sticky failure words（atomic_ref relaxed）。
- staging 布局、内核体（gather_copies）、launcher、注释、docstring 全部重写；模块文档明示
  "concept credit: Morrowmake's host-shm all-reduce；实现为本文件自有表达，欢迎同款重合检查复跑"。
- PR 范围对齐原审稿（copies-only，285 行文件的原位替换）；sum/mix 折叠变体留在本地 066 树，不入本 PR。
- 评论：`ashhart/TensorFold#466`（issuecomment-6049448096）。

## 验证收据

| 关卡 | 结果 |
|---|---|
| 4-rank 位精确（PR 变体，5 用例） | i32 rank-id 置放模式 + bf16/f32 种子抽取（含部署形状 8×4096）**全部 `torch.equal`** |
| 4-rank 位精确（066 树 v3 全功能版，7 形状 × sum/mix/copy） | **21/21 全等**（ALL_BITWISE_EQUAL） |
| 端到端窗口（v3 全功能版，GLM-5.3-Flash EXL3 TP4） | structured **168.8** / coding 129.1 / prose 102.0 tok/s（ported 版同数学带 169.2-170.1，噪声带内）；exactness **`0436adfa`** 与全部历史配置逐位一致；accept_ratio 0.7565 不变 |
| 微基准（v3 vs v2） | R=1/4/8 = 38.2/96.7/177.8 µs vs 36.9/95.9/175.7 µs（<2%，噪声带内） |

调参过程中的两次实测回退（留档供复跑者）：leader 串行轮询 3 peer（+26µs/调用）→ 改回每 peer 一线程并行；
`announce()` 全员 fence+store（1024 次冗余 host-mapped store，+28µs）→ 门控为 tid0 单 store。两修后恢复同速。

## 附带事件：生产恢复异常（已闭环：宿主重启恢复 289.0）

窗口结束后生产 Morrowmake 恢复**连续四次同型失败**：TP worker 过 NCCL init 后停摆
（530-618MiB @100% util，权重不加载），rank 0 侧正常。排查记录：

- 第 1-3 次失败曾归因于孤儿进程毒化（vllm setproctitle 改名 `VLLM::EngineCore/Worker_TP*`，
  stop.sh 与常规 pkill 模式全部漏杀；按 nvidia-smi compute-apps pid 清杀后 GPU 回 19MiB）；
- **第 4 次在真·干净态（0 进程、19MiB、/dev/shm 8.3M 正常、无 Xid 可见）下仍同点卡死**——
  楔死在驱动层，进程手段穷尽。前情：本日多次 4-rank 测试/微基准进程在生产上下文之上反复
  建毁 CUDA context + host-registered shm 突发流量。
- 与本机 M2 实录（Xid 31 CE2 / nv_open_q / "重启清驱动后正常"）同家族；GPU reset 因 Xorg 持有不可用。
- **终态（用户授权宿主重启后）**：重启 → start.sh 一次成功 → structured **289.0 tok/s**
  （288.96-289.27，any_nan=false，accept_ratio 0.935，6.4 tok/step）——生产完全恢复，
  收据 `<GPU_HOST>:~/bench/verify_after_reboot.json`。
- 复盘教训（入纪律）：生产宿主上跑 4-rank 实验进程须节流；vllm 改名进程的清理必须按
  compute-apps pid 或 `VLLM::` 前缀匹配，`bin/vllm` 模式会漏。

## 三出口决策影响

②号出口从"被动等待 review"变为"我方义务已尽，球在上游"：fresh 内核 + 收据已交，等 ashhart 复跑重合检查与 re-review。
①号出口（接受 170.1 定格）不受影响，全部归档完好。
