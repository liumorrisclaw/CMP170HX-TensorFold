#!/usr/bin/env python3
"""Validate the production g128 packed -> gptq_marlin_repack -> marlin kernel path
against the direct dequant reference. No requant: the packed values ARE the
quantization, so the match should be tight (bf16 rounding only)."""
import os, sys
os.environ["TF_MARLIN_SO"] = "/home/hym/tensorfold-poc/marlin_standalone_p2/_ampere_marlin_C.abi3.so"
os.environ["TENSORFOLD_MIN_CAPABILITY"] = "8.0"
sys.path.insert(0, "/home/hym/tensorfold-poc/TensorFold066/src")
import torch
from safetensors import safe_open
from tensorfold.families.glm5_next.cuda import marlin_experts as me

DEV = "cuda:0"
torch.manual_seed(0)
ext = me._ext()

F = '/home/hym/glm53-recipe/models/GLM-5.3-Flash-W4A16-MTP/model-00001-of-00009.safetensors'
pre = 'model.language_model.layers.3.mlp.experts.0'
with safe_open(F, framework='pt', device='cpu') as f:
    gp = f.get_tensor(pre + '.gate_proj.weight_packed')   # [2048, 512] int32
    gs = f.get_tensor(pre + '.gate_proj.weight_scale')    # [2048, 32] bf16
    up = f.get_tensor(pre + '.up_proj.weight_packed')
    us = f.get_tensor(pre + '.up_proj.weight_scale')

# rank-0 shard: rows [0:512] (the contiguous split)
gp, gs, up, us = gp[:512], gs[:512], up[:512], us[:512]
NI = gp.shape[0]          # 512
K = gp.shape[1] * 8       # 4096
print(f"shard packed {tuple(gp.shape)} scale {tuple(gs.shape)} -> n={NI} k={K}")

# repack: unpack the sequential nibbles -> q [1024, 4096] -> .T -> the torch marlin repack
fused_scale = torch.cat([gs, us], 0)                      # [1024, 32]
fused_q = torch.cat([me._unpack_q(gp), me._unpack_q(up)], 0)   # [1024, 4096] int8 0..15
w1p = me._repack_batch(fused_q.t().contiguous().unsqueeze(0), K, 2 * NI).to(DEV)
print("repacked:", tuple(w1p.shape))
assert tuple(w1p.shape) == (1, K // 16, 2 * 2 * NI), w1p.shape
s1p = me._permute_scales(fused_scale.t().contiguous()).unsqueeze(0).to(DEV)
w1p = w1p.unsqueeze(0)

# the dequant reference (sequential nibbles, offset+8)
def deq(packed, scale):
    n, k8 = packed.shape
    w = packed.to(torch.int64) & 0xFFFFFFFF
    nib = ((w[..., None] >> (torch.arange(8) * 4)) & 0xF).reshape(n, k8 * 8)
    return ((nib - 8).float() * scale.float().repeat_interleave(128, dim=1))

W13_ref = torch.cat([deq(gp, gs), deq(up, us)], 0)        # [1024, 4096]
W13_qref = fused_q.t().contiguous()                       # [4096, 1024] the packed q, for the exactness cross-check
print("reference W13:", tuple(W13_ref.shape), "scale range:",
      (fused_scale.min().item(), fused_scale.max().item()))

# the marlin run: E=1, TOPK=16 pair space, M=1
M, P = 1, 16
topk = torch.zeros(1, 16, dtype=torch.int64); topk[0, 1:] = -1
def build_route(topk_ids):
    """The reference route builder (host): expert-blocked, 8-aligned; invalid pairs dropped."""
    E = 289
    flat = topk_ids.view(-1).long()
    valid = (flat >= 0).nonzero().flatten()
    e_sorted = flat[valid]
    order = valid[torch.argsort(e_sorted, stable=True)]
    e_sorted = e_sorted[torch.argsort(e_sorted, stable=True)]
    counts = torch.bincount(e_sorted, minlength=E)
    blocks_per_e = (counts + 7) // 8
    blk_off = torch.cumsum(blocks_per_e, 0) - blocks_per_e
    total_blocks = int(blocks_per_e.sum())
    n_slots = topk_ids.numel()
    sorted_ids = torch.full((total_blocks * 8,), n_slots, dtype=torch.int64)
    pos_in_e = torch.arange(len(order)) - torch.repeat_interleave(torch.cumsum(counts, 0)[e_sorted] - counts[e_sorted], torch.ones_like(order))
    pos = (blk_off[e_sorted] + pos_in_e // 8) * 8 + pos_in_e % 8
    sorted_ids[pos] = order
    expert_ids = torch.full((total_blocks,), -1, dtype=torch.int64)
    nz = (blocks_per_e > 0).nonzero().flatten()
    expert_ids[: total_blocks] = torch.repeat_interleave(nz, blocks_per_e[nz])
    ntpp = torch.tensor([total_blocks * 8], dtype=torch.int64)
    return sorted_ids, expert_ids, ntpp

sorted_ids, expert_ids, ntpp = [t.to(torch.int32) for t in build_route(topk)]
hidden = torch.randn(1, K, device=DEV, dtype=torch.bfloat16, generator=torch.Generator(DEV).manual_seed(7))
ctr = torch.zeros(2, dtype=torch.int32, device=DEV)
part = torch.zeros(P, 2 * NI, dtype=torch.float32, device=DEV)
ks = ext.decode_gemm(hidden, w1p, s1p, sorted_ids.to(DEV), expert_ids.to(DEV), ntpp.to(DEV),
                     torch.ones(1, 16, device=DEV), 16, P, K, 2 * NI, True, 64, 0, ctr, part)
torch.cuda.synchronize()

x32 = hidden.float().cpu()[0]
y_ref = W13_ref.to(torch.bfloat16).float() @ x32           # [2048]
d = (part[0].cpu() - y_ref).abs().max().item()
rel = d / y_ref.abs().max().item()
print(f"part max|d| = {d:.5f} (rel {rel:.5f})  -> {'PASS' if rel < 0.02 else 'FAIL'}")
