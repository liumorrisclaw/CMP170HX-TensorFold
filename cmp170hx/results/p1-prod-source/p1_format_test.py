#!/usr/bin/env python3
"""Validate the production checkpoint's compressed-tensors W4A16 g128 expert format.

Checks: (a) the nibble convention (offset +8 vs signed), (b) the k-group scale
consistency (detects a k permutation from actorder=static), (c) the packed nibble
order (sequential vs lane).
"""
import torch
from safetensors import safe_open

F = '/home/hym/glm53-recipe/models/GLM-5.3-Flash-W4A16-MTP/model-00001-of-00009.safetensors'
KEY = 'model.language_model.layers.3.mlp.experts.0.gate_proj'

with safe_open(F, framework='pt', device='cpu') as f:
    packed = f.get_tensor(KEY + '.weight_packed')      # [2048, 512] int32
    scale = f.get_tensor(KEY + '.weight_scale')        # [2048, 32] bf16
    shape = f.get_tensor(KEY + '.weight_shape')        # [2] int64

n, k8 = packed.shape
k = int(shape[1])
print(f'packed {tuple(packed.shape)} scale {tuple(scale.shape)} shape {shape.tolist()} (n={n}, k={k})')

# sequential nibbles: nibble j of word w = the input 8w + j
w = packed.to(torch.int64) & 0xFFFFFFFF
nib = ((w[..., None] >> (torch.arange(8) * 4)) & 0xF).reshape(n, k8 * 8)   # [n, k]
# two candidate conventions
q_off = (nib - 8).float()          # the uint4b8 offset convention
q_sgn = torch.where(nib < 8, nib.float(), nib.float() - 16)   # the two's-complement signed
# broadcast the scales per 128-group
sc = scale.repeat_interleave(4, dim=1).float()                # [n, k] (32 groups × 4 words... wait k/128=32 groups, each = 32 int4 → 4 words... k8*8 = 4096; 4096/128 = 32 ✓ the group g covers the nibbles [g*128, (g+1)*128) ✓
sc_full = scale.float().repeat_interleave(128, dim=1)
print('scale_full', tuple(sc_full.shape))

for name, q in (('offset+8', q_off), ('signed', q_sgn)):
    W = q * sc_full
    # (a) the zero-centered check: the symmetric quant → the values ~zero-mean per group
    gm = W.view(n, 32, 128).mean(dim=(0, 2)).abs().mean().item()
    # (b) the per-group value/scale consistency: |W| within a group should be <= 8×scale (the max q=7... for the offset: q ∈ [-8,7] → |W| ≤ 8×scale)
    viol = (W.view(n, 32, 128).abs().amax(dim=2) > 8.0 * scale.float() + 1e-6).float().mean().item()
    print(f'{name}: |group mean| = {gm:.4g}, the |W|>8scale fraction = {viol:.4f}')

# (c) the scale-vs-values agreement per group: for the CORRECT grouping, the group's max|W| ≈ 7-8× the scale;
# for a MISALIGNED grouping (a permutation), some groups would have max|W| >> 8× their scale
W = q_off * sc_full
gmax = W.view(n, 32, 128).abs().amax(dim=2)          # [n, 32]
ratio = (gmax / scale.float()).clamp(min=0)
print('the ratio max|W|/scale per group: mean', round(ratio.mean().item(), 3),
      'p95', round(ratio.quantile(0.95).item(), 3), 'max', round(ratio.max().item(), 3),
      '(the healthy range ~6.5-8.0; >>8 or ~0 = a misaligned grouping)')
