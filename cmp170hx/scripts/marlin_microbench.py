#!/usr/bin/env python3
"""Micro-bench: Morrowmake's prebuilt sm_80 Marlin MoE decode kernels on GLM expert shapes.

Measures the 3-kernel sequence (w13 fused gateup + act + w2 down) at world=4
geometry (K=4096, N=512, E=288, topk=8, M tokens) and compares with the
per-layer budget of the TF EXL3 stack (~62 us/layer expert segment).

Run on <GPU_HOST> GPU0 (brief SM contention with production, seconds total).
"""
import importlib.util
import sys
import time

import torch

SO = "/home/<user>/tensorfold-poc/marlin_standalone/_ampere_marlin_C.abi3.so"
spec = importlib.util.spec_from_file_location("_ampere_marlin_C", SO)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ops = torch.ops._ampere_marlin_C
print("ABI:", mod.build_info())

torch.cuda.set_device(0)
E, TOPK, K, N = 288, 8, 4096, 512
M = 4                                   # decode rows
slots = M * TOPK                        # 32
n_slots = slots // 8                    # 4 aligned blocks of 8
assert slots % 8 == 0

# weights: marlin packed [E][K/16][2N] int32 (w13: gate+up fused -> 2N), scales [E][K/128][N] bf16
w1 = torch.randint(-2**31, -1, (E, K // 16, 2 * N), dtype=torch.int32, device="cuda")
s1 = (torch.randn(E, K // 128, N, device="cuda") * 0.01).to(torch.bfloat16)
w2 = torch.randint(-2**31, -1, (E, N // 16, K), dtype=torch.int32, device="cuda")
s2 = (torch.randn(E, N // 128, K, device="cuda") * 0.01).to(torch.bfloat16)
print(f"w1 {tuple(w1.shape)} {w1.numel()*4/2**20:.0f}MiB | w2 {tuple(w2.shape)} {w2.numel()*4/2**20:.0f}MiB")

# routing: 4 tokens, each picks 8 distinct experts round-robin -> every expert gets 4 slots
topk_ids = torch.stack([(torch.arange(TOPK) + t * E // M) % E for t in range(M)]).int().cuda()  # disjoint ranges
topk_w = torch.rand(M, TOPK, dtype=torch.float32, device="cuda") + 0.1
hidden = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

# block-8 alignment (each expert owns exactly one aligned block here)
pairs = (topk_ids[:, None, :] == torch.arange(E, device="cuda")[None, :, None]).nonzero()  # [pairs, 3]: t, k, e
n_slots = ((pairs.shape[0] + 7) // 8)
sorted_ids = torch.zeros(n_slots * 8, dtype=torch.int32, device="cuda")
expert_ids = torch.zeros(n_slots, dtype=torch.int32, device="cuda")
ntpp = torch.zeros(E, dtype=torch.int32, device="cuda")
for blk in range(n_slots):
    chunk = pairs[blk * 8:(blk + 1) * 8]
    e = int(chunk[0, 2])
    expert_ids[blk] = e
    ntpp[e] = chunk.shape[0]
    for j, (_, t, k) in enumerate(chunk.tolist()):
        sorted_ids[blk * 8 + j] = t * TOPK + k
print(f"pairs {pairs.shape[0]}, blocks {n_slots}, slots {slots}")

ctr = torch.zeros(2, dtype=torch.int32, device="cuda")
part = torch.zeros(1, n_slots, 2 * N, dtype=torch.float32, device="cuda")
h = torch.zeros(slots, N, dtype=torch.bfloat16, device="cuda")
c3 = torch.zeros(slots, K, dtype=torch.bfloat16, device="cuda")
out = torch.zeros(M, K, dtype=torch.bfloat16, device="cuda")
W13_ROWS, W2_ROWS = 64, N // 16

def layer():
    ksplit = ops.decode_gemm(hidden, w1, s1, sorted_ids, expert_ids, ntpp, topk_w.view(-1),
                             TOPK, n_slots, K, 2 * N, True, W13_ROWS, 0, ctr, part)
    ops.decode_act(part, topk_ids.view(-1).int(), ksplit, slots, N, 10.0, h)
    ops.decode_gemm(h, w2, s2, sorted_ids, expert_ids, ntpp, topk_w.view(-1),
                    TOPK, n_slots, N, K, False, W2_ROWS, 0, ctr, c3)

# correctness-ish smoke: does it run and produce nonzero
layer(); torch.cuda.synchronize()
print(f"smoke: part norm {part.norm().item():.3f}, c3 norm {c3.norm().item():.3f}")

for iters in (100, 400):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        layer()
    torch.cuda.synchronize()
    dt = (time.time() - t0) / iters * 1000
    print(f"{iters} iters: {dt:.3f} ms/layer-sequence")
print(f"TF EXL3 expert segment reference: ~60-67 us/layer (~2.7-3.0 ms/step over 45 layers)")
