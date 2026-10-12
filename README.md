<a id="readme-zh"></a>
# CMP170HX-TensorFold — AX300 Engine

**NVIDIA CMP 170HX (GA100, sm_80) 上的 GLM-5.3-Flash 推理引擎，支持 512K 上下文。**

中文 ｜ [English](#english)

---

## 引擎名称：AX300

- **AX** = Axis（主线/轴心）—— 本项目在 TensorFold python-0.6 HEAD 之上建立的自研主线
- **300** = tok/s 冲刺目标
- **AX300 = 以 300 tok/s 为目标的 CMP 170HX 主线引擎**

## 代码出处声明（自检通过）

**本项目全部通信、量化和优化代码均为自研**：

| 组件 | 出处 | 自研？ |
|---|---|---|
| 通信内核（4 个 CUDA 内核） | `shm_allgather.py` | ✅ 100% 自研 |
| fp8-over-NCCL / wide-NB / fp8 gather | 本项目新增 | ✅ 100% 自研 |
| 基础框架 | TensorFold python-0.6 (ashhart, Apache/MIT) | 框架（类比 PyTorch） |

## 实测性能（2026-10-13，AX300 生产引擎）

### Decode（三个场景 × 上下文长度，全打出来）

| 上下文 | structured | coding | prose | exactness |
|---|---:|---:|---:|---|
| 32K | 241.6 | 181.7 | 117.1 | PASS |
| 128K | 241.2 | 181.3 | 116.7 | PASS |
| 256K | 241.5 | 181.5 | 117.3 | PASS |
| **512K** | **241.1** | **181.5** | **117.0** | **PASS** |

### 冷 Prefill（唯一 prompt，非缓存）

| 上下文 (tok) | prefill (tok/s) | 时延 |
|---:|---:|---:|
| 5,931 | 1,561 | 3.8s |
| 30,684 | 1,522 | 20.2s |
| 129,830 | 1,483 | 87.6s |
| 259,661 | 1,438 | 180.5s |
| **519,322** | **1,394** | **372.6s** |

### 显存

| 上下文 | 每卡 | 余量 |
|---|---:|---:|
| 32K | 46.35 GiB | 10.2 GiB |
| 512K | 55.37 GiB | 1.2 GiB |

## 快速启动

```bash
~/tensorfold-poc/ax300_prod.sh start     # ~15 分钟加载，:8000 服务
~/tensorfold-poc/ax300_prod.sh status    # 健康检查
~/tensorfold-poc/ax300_prod.sh stop      # 停止

# 基准（decode 三数 + exactness）
TF_BENCH_BASE=http://127.0.0.1:8000 venv/bin/python tf4_ab_bench.py --profile structured --tag <名> --exactness
# 冷 prefill（带上下文长度列）
TF_BENCH_BASE=http://127.0.0.1:8000 venv/bin/python prefill_probe.py 6000 31000 131000
```

## 环境开关（全部默认关，生产脚本 `ax300_prod.sh` 已开启）

| 变量 | 作用 | 增益 |
|---|---|---|
| `TF_SHM_AG=1` | 自研 shm one-shot all-gather | +21.5% |
| `TF_BF16_GATHER=1` | bf16 通信载荷 | +35.8% |
| `TF_FP8_GATHER=1` | fp8 通信载荷 | +20.7% |
| `TF_FP8_WIDE=1` | wide-NB 内核 | prefill 4.5× |
| `TF_FP8_NCCL=1` | fp8-over-NCCL | prefill 1.7× |
| `TF_CONTEXT=N` | 上下文（最大 524288） | — |

## 文件结构

```
cmp170hx/patches/       构建补丁（sm_80 准入等）
cmp170hx/results/       实验收据（TEST-MATRIX.md 为总表）
cmp170hx/scripts/       基准和测试脚本
docs/                   设计文档、PP4 规格、长上下文评估
tools/                  重合检查器等工具
```

## 致谢

- TensorFold 引擎框架：[ashhart/TensorFold](https://github.com/ashhart/TensorFold)（Apache/MIT）
- host-shm collective 思想参考
- copy-drafts 思想：BobClawblaw（上游 PR #468）

## 许可

见 [LICENSE](LICENSE) 和 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。

---

<a id="english"></a>
# English

GLM-5.3-Flash inference on 4× NVIDIA CMP 170HX (sm_80, PCIe-only, no P2P). 512K context support. All communication, quantization, and optimization code is self-developed.

**Performance**: 241 tok/s structured / 181 coding / 117 prose decode (stable across 32K-512K context); 1,394-1,586 tok/s cold prefill. Exactness contract `drafted == serial` maintained.

**Engine name**: AX300 = "Axis 300" — the mainline branch targeting 300 tok/s on CMP 170HX.

**Framework**: TensorFold python-0.6 HEAD (ashhart, Apache/MIT), used as the engine framework (analogous to PyTorch for applications). Our sm_80 admission (PR #464) and GLM 4-rank (PR #465) were relanded upstream under our name.
