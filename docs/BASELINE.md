# 上游基线声明（主线一致性，2026-10-11）

**主线基准 = 上游最新 release `v1.0.5`；引擎源码 = `python-0.6` HEAD `ed78d6fc`。**

## 为什么引擎源码不是 1.0.x（上游官方文档原文，v1.0.5 tag 的 GLM 配方）

> "This recipe describes the Python 0.6.6 engine, maintained on `python-0.6`. Its CUDA EXL3
> reader, two-rank flags and DFlash2 policies **do not apply to the native Zig binary**. The
> native binary serves GLM on two M5 Ultra Macs ... **it has no GLM CUDA backend yet**."
>
> "An unpinned install selects the native default branch, **which does not serve this CUDA
> recipe**."

仓库结构证据：`v1.0.5`（main）已无 `src/`——Python 引擎整体退出主线，main 是纯 Zig 引擎仓
（build.zig）。GLM-5.3-Flash 在 NVIDIA GPU 上只能由 python-0.6 线服务——这是上游自己的
设计，连官方配方都把安装钉在 python-0.6 的 0.6.6 修订版上。

**因此本仓的"最新版本" = python-0.6 HEAD（比官方配方钉的 0.6.6 还新 12 个提交，含官方
Ampere 修复与我们署名的 #464/#465），即该部署能跑的最新上游代码。**

## 主线一致性动作

1. 上游 remote 已跟踪 `main`（v1.0.5）与 `python-0.6`；每次上游 release 检查
   README「qualified models」表中 GLM+CUDA 是否晋级——**一旦 Zig 引擎获得 GLM CUDA 后端，
   立即评估整体迁移**（届时本仓全部成果按新引擎 API 重新表达）；
2. 1.0.x 中可借鉴的思想（M5 张量单元 prompt unpack、轻量 prompt state、并发 lane 等）
   按"思想署名 + 自己表达"红线移植到本线，逐项过重合门；
3. 本文件随每次上游 release 复查更新。
