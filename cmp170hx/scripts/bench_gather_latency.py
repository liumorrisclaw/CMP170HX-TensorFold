#!/usr/bin/env python3
"""4-rank per-call latency microbench for the shm gather kernels (Task E premise check).

Question (handoff Task E P2 "chunked pipelined gather", priced +3-4% / days): how much
wall time does one gather cost at decode shapes, and does block concurrency (the
already-shipped chunking: NB blocks x per-block flag barrier) matter? If blocks=1 ~
blocks=4, the cost is one irreducible PCIe flight and staggering buys ~0.

Measures, CUDA events, median of 300 calls after 50 warmup, per (kernel, rows, blocks):
  sum  = allgather_sum   (S2 deployed kernel)
  mix  = allgather_sum_mix (E-min kernel, NB=4 = the pipelined overlap)
Rank 0 reports; ranks stay in lockstep via the store. Runs beside production (~0.5 GB/rank).

Run on <GPU_HOST>:  ~/tensorfold-poc/venv/bin/python bench_gather_latency.py
"""
import os
import statistics
import subprocess
import sys

SRC = "/home/<user>/tensorfold-poc/TensorFold066/src"
SITE = "/home/<user>/glm53-recipe/venv/lib/python3.12/site-packages"
PORT = 29519
D = 4096
CASES = [("sum", 1, 4), ("sum", 4, 1), ("sum", 4, 2), ("sum", 4, 4), ("sum", 8, 4), ("mix", 4, 4), ("mix", 8, 4)]
WARM, ITERS = 50, 300


def run_rank(rank: int, port: int) -> int:
    import torch
    import torch.distributed as dist

    sys.path.insert(0, SRC)
    sys.path.insert(0, SITE)
    from tensorfold.cuda.shm_allgather import ShmAllGather

    torch.cuda.set_device(0)
    store = dist.TCPStore("127.0.0.1", port, world_size=4, is_master=(rank == 0))
    results = {}
    for ci, (kind, rows, blocks) in enumerate(CASES):
        shm = ShmAllGather(rank, 4, store, blocks=blocks)
        g = torch.Generator(device="cuda").manual_seed(4242 + ci)
        part = torch.randn(rows, D, device="cuda").to(torch.bfloat16)
        x = torch.randn(rows, 4 * D, device="cuda").to(torch.bfloat16)
        post = torch.rand(rows, 4, device="cuda")
        comb = torch.rand(rows, 16, device="cuda")
        out = torch.empty(rows, D, dtype=torch.bfloat16, device="cuda")

        def call():
            if kind == "sum":
                shm.all_gather_sum(part.reshape(-1), out.reshape(-1))
            else:
                shm.all_gather_sum_mix(part.reshape(-1), x, post, comb)

        for _ in range(WARM):
            call()
        torch.cuda.synchronize()
        store.set(f"go_{ci}_r{rank}", "1")
        store.wait([f"go_{ci}_r{r}" for r in range(4)])
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(ITERS):
            call()
        e1.record()
        torch.cuda.synchronize()
        results[(kind, rows, blocks)] = e0.elapsed_time(e1) * 1000.0 / ITERS   # us/call, batch-amortized
        store.set(f"done_{ci}_r{rank}", "1")
        store.wait([f"done_{ci}_r{r}" for r in range(4)])
        shm.close()

    if rank == 0:
        print(f"{'kernel':>6} {'rows':>4} {'blocks':>6} {'us/call':>9}", flush=True)
        for (kind, rows, blocks), us in results.items():
            print(f"{kind:>6} {rows:>4} {blocks:>6} {us:>9.2f}", flush=True)
        s4 = results[("sum", 4, 4)]
        s1 = results[("sum", 4, 1)]
        print(f"\nblocks=4 vs blocks=1 (R=4): {s4:.2f} vs {s1:.2f} us "
              f"-> concurrency buys {100 * (s1 - s4) / s1:.1f}%", flush=True)
        for r_in in (1, 4, 8):
            per_round = 90 * results[("sum", r_in, 4)]
            print(f"90 gathers/round @R={r_in}: {per_round / 1000:.2f} ms "
                  f"({100 * per_round / 12300:.1f}% of a 12.3 ms round)", flush=True)
        print("RESULT GATHER_BENCH_DONE", flush=True)
    return 0


def main() -> int:
    if len(sys.argv) == 3:
        return run_rank(int(sys.argv[1]), int(sys.argv[2]))
    env = dict(os.environ)
    env.update(PYTHONPATH=f"{SITE}:{SRC}", TENSORFOLD_MIN_CAPABILITY="8.0",
               TORCH_EXTENSIONS_DIR=os.path.expanduser("~/tensorfold-poc/tf_ext066"))
    rc = 0
    procs = []
    for r in range(4):
        cenv = dict(env)
        cenv["CUDA_VISIBLE_DEVICES"] = str(r)
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), str(r), str(PORT)], env=cenv))
    for p in procs:
        code = p.wait()
        if code:
            rc = rc or code
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
