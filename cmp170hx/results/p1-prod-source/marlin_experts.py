"""Marlin W4A16 routed experts (``TF_MARLIN_EXPERTS=1``).

Converts the checkpoint's affine group-64 experts to the g128-symmetric Marlin
layout at load — the per-group bias folds into the requantized weights — and
runs the prebuilt sm_80 decode kernels for decode windows and prefill chunks
alike. The affine packed experts are not kept alongside: both formats on one
card next to the KV cache do not fit.
"""

import os
from dataclasses import dataclass

import torch

SO = os.environ.get("TF_MARLIN_SO",
                    os.path.expanduser("~/tensorfold-poc/marlin_standalone/_ampere_marlin_C.abi3.so"))
TOPK = 8                                   # the router's top_k (GLM)
PAIR = 2 * (TOPK + 1)                      # the 16-pair space the kernels index

_ops = None


def _ext():
    global _ops
    if _ops is None:
        torch.ops.load_library(SO)
        _ops = torch.ops._ampere_marlin_C
    return _ops


_TABLES = None


def _tables(dev):
    """The packed layout's (k, n) offset per lane and nibble: k 2c/2c+1/2c+8/2c+9, n g/g+8 (n + 16j + 64t)."""
    global _TABLES
    if _TABLES is None:
        K16 = torch.zeros(32, 8, dtype=torch.int64)
        N16 = torch.zeros(32, 8, dtype=torch.int64)
        for l in range(32):
            c, g = l % 4, l // 4
            for i, (ko, no) in enumerate([(2*c, g), (2*c+8, g), (2*c, g+8), (2*c+8, g+8),
                                          (2*c+1, g), (2*c+9, g), (2*c+1, g+8), (2*c+9, g+8)]):
                K16[l, i], N16[l, i] = ko, no
        _TABLES = (K16, N16)
    return _TABLES[0].to(dev), _TABLES[1].to(dev)


def _permute_scales(s):
    """[.., K/128, N] -> n-permuted within each 64-col chunk (the kernels' scale layout)."""
    perm = torch.tensor([i + 8 * j for i in range(8) for j in range(8)], device=s.device)
    shape = s.shape
    return s.reshape(-1, 64)[:, perm].reshape(shape).contiguous()


def _requant_g128(W, gs=128):
    """[.., n, k] fp32 -> (q int8 in 1..15, scales [.., k/gs, n] bf16); scale = max|w|/7 per group."""
    *lead, n, k = W.shape
    w = W.reshape(*lead, n, k // gs, gs)
    scale = (w.abs().amax(-1) / 7.0).clamp_min(1e-12)
    q = (w / scale.unsqueeze(-1)).round().clamp(-7, 7) + 8
    return q.reshape(*lead, n, k).to(torch.int8), scale.transpose(-1, -2).contiguous().to(torch.bfloat16)


def _repack(q_kn, size_k, size_n):
    """One matrix: q [size_k, size_n] int (0..15) -> packed int32 [size_k//16, (size_n//64)*128]."""
    K16, N16 = _tables(q_kn.device)
    q = q_kn.to(torch.int64)
    R16, T = size_k // 16, size_n // 64
    shifts = torch.arange(8, device=q.device, dtype=torch.int64) * 4
    rr = torch.arange(R16, device=q.device)
    tt = torch.arange(T, device=q.device)
    cols = tt.view(-1, 1) * 128 + torch.arange(32, device=q.device).view(1, -1) * 4
    out = torch.zeros(R16, T * 128, dtype=torch.int32, device=q.device)
    for j in range(4):
        idx_k = rr.view(-1, 1, 1) * 16 + K16
        idx_n = tt.view(-1, 1, 1) * 64 + 16 * j + N16
        vals = q[idx_k.unsqueeze(1).expand(-1, T, -1, -1), idx_n.unsqueeze(0).expand(R16, -1, -1, -1)]
        out[:, (cols + j).reshape(-1)] = ((vals << shifts) & 0xFFFFFFFF).sum(-1).to(torch.int32).reshape(R16, -1)
    return out.contiguous()


def _repack_batch(q_bnk, size_k, size_n, chunk=36):
    """[B, size_k, size_n] -> [B, size_k//16, (size_n//64)*128] int32."""
    outs = []
    for i in range(0, q_bnk.shape[0], chunk):
        q = q_bnk[i:i + chunk].to(torch.int64)
        K16, N16 = _tables(q.device)
        B = q.shape[0]
        R16, T = size_k // 16, size_n // 64
        shifts = torch.arange(8, device=q.device, dtype=torch.int64) * 4
        rr = torch.arange(R16, device=q.device)
        tt = torch.arange(T, device=q.device)
        cols = tt.view(-1, 1) * 128 + torch.arange(32, device=q.device).view(1, -1) * 4
        out = torch.zeros(B, R16, T * 128, dtype=torch.int32, device=q.device)
        for j in range(4):
            idx_k = rr.view(-1, 1, 1) * 16 + K16
            idx_n = tt.view(-1, 1, 1) * 64 + 16 * j + N16
            vals = q[:, idx_k.unsqueeze(1), idx_n.unsqueeze(0, 1)]
            out[:, :, (cols + j).reshape(-1)] = ((vals << shifts) & 0xFFFFFFFF).sum(-1).to(torch.int32)
        outs.append(out)
    return torch.cat(outs)


@dataclass
class MarlinExperts:
    """One layer's experts in marlin layout: E routed + the shared expert last."""

    w13: torch.Tensor        # [E+1, K/16, 4*NI] int32 (the packed layout's linear size)
    s13: torch.Tensor        # [E+1, K/128, 2*NI] bf16 (permuted)
    w2: torch.Tensor         # [E+1, NI/16, 2*K] int32
    s2: torch.Tensor         # [E+1, NI/128, K] bf16 (permuted)
    ni: int                  # the rank's expert width
    hidden: int
    ctr: torch.Tensor        # [2] int32 work counters (the kernels reset each other's)

    def gemm13(self, a, sorted_ids, eids, ntpp, topk_w, out):
        """w13 projection of a [R, hidden] into out [R*16, 2*ni] fp32; returns the act ksplit."""
        return _ext().decode_gemm(a, self.w13, self.s13, sorted_ids, eids, ntpp,
                                  topk_w.reshape(-1), 16, a.shape[0] * 16, self.hidden, 2 * self.ni,
                                  True, 64, 0, self.ctr, out)

    def act(self, part, ids16, ksplit, out, limit):
        """clamp-silu-mul of the w13 partials into out [R*16, ni] bf16."""
        _ext().decode_act(part, ids16.reshape(-1), ksplit, ids16.numel(), self.ni, float(limit), out)

    def gemm2(self, h, sorted_ids, eids, ntpp, topk_w, out):
        """down projection of h [R*16, ni] into out [R*16, hidden] bf16 (unweighted)."""
        return _ext().decode_gemm(h, self.w2, self.s2, sorted_ids, eids, ntpp,
                                  topk_w.reshape(-1), 16, h.shape[0], self.ni, self.hidden,
                                  False, self.ni // 16, 0, self.ctr, out)


def convert_experts(gate, up, down):
    """Stacked raw parts (words [E+1, NI, K/8], scales, biases [E+1, NI, K/64]) -> MarlinExperts."""
    from tensorfold.families.glm5_next.cuda.qmm import dequantize

    words_g, scales_g, biases_g = gate
    E, NI, k8 = words_g.shape
    K = k8 * 8
    W13 = torch.cat([dequantize(words_g, scales_g, biases_g), dequantize(up[0], up[1], up[2])], dim=1).float()
    q13, s13 = _requant_g128(W13)
    del W13
    Wd = dequantize(down[0], down[1], down[2]).float()
    q2, s2 = _requant_g128(Wd)
    del Wd
    w13 = _repack_batch(q13, K, 2 * NI)
    del q13
    w2 = _repack_batch(q2, NI, K)
    del q2
    ctr = torch.zeros(2, dtype=torch.int32, device=words_g.device)
    return MarlinExperts(w13, _permute_scales(s13), w2, _permute_scales(s2), NI, K, ctr)


class MarlinRoute:
    """Sync-free block-8 route builder; buffers sized for ``rows`` rows of PAIR slots."""

    def __init__(self, rows, dev):
        self.max_pairs = rows * PAIR
        self.max_blocks = rows * (TOPK + 1)          # every valid pair on its own expert
        self.sorted = torch.full((self.max_blocks * 8 + self.max_pairs,), -1,
                                 dtype=torch.int32, device=dev)
        self.eids = torch.full((self.max_blocks + self.max_pairs,), -1, dtype=torch.int32, device=dev)
        self.ntpp = torch.zeros(1, dtype=torch.int32, device=dev)
        self.arange = torch.arange(self.max_pairs, device=dev)

    def build(self, pick16, n_slots):
        """pick16 [R, PAIR] int32 (valid expert ids then -1s) -> (sorted, eids, ntpp); no host sync."""
        flat = pick16.reshape(-1).int()
        P = flat.numel()
        inv = flat < 0
        counts = torch.zeros(290, dtype=torch.int32, device=flat.device)
        counts.scatter_add_(0, flat.clamp_min(0).long(), torch.ones_like(flat))
        counts[0] -= inv.to(torch.int32).sum()
        blocks_pe = (counts + 7) // 8
        blk_off = torch.cumsum(blocks_pe, 0) - blocks_pe
        self.ntpp[0] = blocks_pe.sum() * 8
        key = torch.where(inv, torch.full_like(flat, 290), flat)
        order = key.argsort(stable=True)
        e_sorted = flat[order]
        gs = torch.cumsum(counts, 0) - counts
        ranks = self.arange[:P] - gs[e_sorted.clamp_min(0)]
        pos = (blk_off[e_sorted.clamp_min(0)] + ranks // 8) * 8 + ranks % 8
        pos = torch.where(e_sorted >= 0, pos, (self.max_blocks * 8 + self.arange[:P]).int()).long()
        self.sorted[: self.max_blocks * 8].fill_(n_slots)
        self.sorted[pos] = torch.where(e_sorted >= 0, order.int(), torch.zeros_like(order.int()))
        self.eids.fill_(-1)
        self.eids[pos // 8] = e_sorted.clamp_min(0)
        return self.sorted, self.eids, self.ntpp
